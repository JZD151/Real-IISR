import time
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import Real_IISR, VQVAE, VectorQuantizer
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModel
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
import wandb
import torch.nn.functional as F
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from torchvision.utils import save_image
from models.src.loss import ThermalOrderConsistencyLoss

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp, var: DDP, 
        text_encoder, clip_vision, exp_name,
        var_opt: AmpOptimizer, label_smooth: float, wandb_flag=False,
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer
        self.var_wo_ddp: Real_IISR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        self.wandb_flag = wandb_flag
        self.exp_name = exp_name

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.text_encoder = text_encoder
        self.clip_vision = clip_vision
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = []
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        self.start_token = self.L - patch_nums[-1] * patch_nums[-1]
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        
        self.thermalOrderConsistencyLoss = ThermalOrderConsistencyLoss(reduction="mean").to(device)
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        CEL_mean, CEL_tail, acc_mean, acc_tail, mse_loss, order_loss, total_loss = 0, 0, 0, 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for batch in ld_val:

            inp_B3HW = batch["HR"].to(dist.get_device(), non_blocking=True)
            lr_inp = batch["LR"].to(dist.get_device(), non_blocking=True)
            label_B = batch["label_B"].to(dist.get_device(), non_blocking=True)
            B, V = inp_B3HW.shape[0], self.vae_local.vocab_size
            gt_idx_Bl, idx_N_list = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl[0:len(self.patch_nums)], dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(idx_N_list)

            with torch.no_grad():
                self.var_wo_ddp.forward
                logits_BLV, hr_b = self.var(x_BLCv_wo_first_l, label_B, lr_inp, text_hidden=None, lr_inp_scale = None)
            gt_BL = torch.cat((gt_BL[:, :-self.var_wo_ddp.last_level_pns], gt_BL[:, -self.var_wo_ddp.last_level_pns:]), dim=1)
            CEL_mean += self.val_loss(logits_BLV.contiguous().data.view(-1, V), gt_BL.view(-1)) * B
            CEL_tail += self.val_loss(logits_BLV.contiguous().data[:, self.start_token:].reshape(-1, V), gt_BL[:, self.start_token:].reshape(-1)) * B
            acc_mean += (logits_BLV.contiguous().data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.contiguous().data[:, self.start_token:].argmax(dim=-1) == gt_BL[:, self.start_token:]).sum() * (100 / (gt_BL.shape[1]-self.start_token))
            pred_BL = logits_BLV.argmax(dim=-1).long().to(next(self.vae_local.parameters()).device)  # [B, L]
            
            v_patch_nums = tuple(self.vae_local.quantize.v_patch_nums)
            per_scale_L = [int(pn) * int(pn) for pn in v_patch_nums]
            need_L = sum(per_scale_L)

            if pred_BL.shape[1] != need_L:
                if pred_BL.shape[1] == per_scale_L[-1]:
                    B, L = pred_BL.shape
                    pn = int(v_patch_nums[-1])
                    h_BChw = self.vae_local.quantize.embedding(pred_BL).transpose(1, 2).view(
                        B, self.vae_local.Cvae, pn, pn
                    )
                    sr_img = self.vae_local.fhat_to_img(h_BChw).add_(1).mul_(0.5)
                else:
                    raise ValueError(
                        f"Length of idx mismatch: got {pred_BL.shape[1]}, need {need_L} "
                        f"(scales={v_patch_nums}, per-scale={per_scale_L})"
                    )
            else:
                ms_idx_Bl = list(torch.split(pred_BL, per_scale_L, dim=1))
                sr_img = self.vae_local.idxBl_to_img_lora(ms_idx_Bl, hr_b, same_shape=True, last_one=True).add_(1).mul_(0.5)
                
            hr_img = (inp_B3HW + 1)/2
            lr_img = (lr_inp + 1)/2
            order_loss += self.thermalOrderConsistencyLoss(tar = hr_img, sr = sr_img)  * B
            mse_loss += F.mse_loss(sr_img, hr_img)  * B
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = CEL_mean.new_tensor([CEL_mean.item(), CEL_tail.item(), acc_mean.item(), acc_tail.item(), order_loss.item(), mse_loss.item(), CEL_mean + 0.8 * order_loss + 0.2 * mse_loss, tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        CEL_mean, CEL_tail, acc_mean, acc_tail, order_loss, mse_loss, total_loss, _ = stats.tolist()
        return CEL_mean, CEL_tail, acc_mean, acc_tail, order_loss, mse_loss, total_loss , tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger,
        inp_B3HW: FTen, lr_inp: Union[ITen, FTen], label_B,
        text, prog_si: int, prog_wp_it: float, lr, wd,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = inp_B3HW.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        with torch.no_grad():
            _, idx_N_list = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(idx_N_list[0:len(self.patch_nums)], dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(idx_N_list)

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV, hr_b = self.var(x_BLCv_wo_first_l, label_B, lr_inp, text_hidden=None, lr_inp_scale = None)
            
            logits_loss = self.train_loss(logits_BLV.contiguous().view(-1, V), gt_BL.view(-1)).view(B, -1)

            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            logits_loss = logits_loss.mean(dim=-1).mean()
            pred_BL = logits_BLV.argmax(dim=-1).long().to(next(self.vae_local.parameters()).device)  # [B, L]
            
            v_patch_nums = tuple(self.vae_local.quantize.v_patch_nums)
            per_scale_L = [int(pn) * int(pn) for pn in v_patch_nums]
            need_L = sum(per_scale_L)

            if pred_BL.shape[1] != need_L:
                if pred_BL.shape[1] == per_scale_L[-1]:
                    B, L = pred_BL.shape
                    pn = int(v_patch_nums[-1])
                    h_BChw = self.vae_local.quantize.embedding(pred_BL).transpose(1, 2).view(
                        B, self.vae_local.Cvae, pn, pn
                    )
                    sr_img = self.vae_local.fhat_to_img(h_BChw).add_(1).mul_(0.5)     # [-1,1] -> [0,1]
                else:
                    raise ValueError(
                        f"Length of idx mismatch: got {pred_BL.shape[1]}, need {need_L} "
                        f"(scales={v_patch_nums}, per-scale={per_scale_L})"
                    )
            else:
                ms_idx_Bl = list(torch.split(pred_BL, per_scale_L, dim=1))
                sr_img = self.vae_local.idxBl_to_img_lora(ms_idx_Bl, hr_b, same_shape=True, last_one=True).add_(1).mul_(0.5)

            hr_img = (inp_B3HW + 1)/2
            lr_img = (lr_inp + 1)/2
            order_loss = self.thermalOrderConsistencyLoss(tar = hr_img, sr = sr_img)
            mse_loss = F.mse_loss(sr_img, hr_img)
            loss = logits_loss + 0.8 * order_loss + 0.2 * mse_loss
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            CELmean = self.val_loss(logits_BLV.contiguous().view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                CELtail = acc_tail = -1
            else:               # not in progressive training
                CELtail = self.val_loss(logits_BLV.contiguous().data[:, self.start_token:].reshape(-1, V), gt_BL[:, self.start_token:].reshape(-1)).item()
                acc_tail = (pred_BL[:, self.start_token:] == gt_BL[:, self.start_token:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(CELm=CELmean, CELt=CELtail, Accm=acc_mean, Acct=acc_tail, order_loss=order_loss.item(), mse_loss=mse_loss.item(), total_loss=loss.item(),tnm=grad_norm)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
