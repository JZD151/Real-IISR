import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
from torch.nn import functional as F
import dist
from models.basic_var import AdaLNBeforeHead
from models.basic_var import AdaLNSelfAttn_RoPE, precompute_freqs_cis, precompute_freqs_cis_cross, precompute_freqs_cis_zeros
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer
import scipy.stats as stats
import torch.utils.checkpoint as checkpoint
from .src.Heatmap import compute_thermal_response_map
from .src.fusion import AFF, iAFF
from .src.sobel import sobel_torch
from .src.cross_attention import CrossAttention

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (128, 256, 512, 1024),
        return_rgbs: bool = False,
    ):
        super().__init__()

        self.return_rgbs = return_rgbs
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        if return_rgbs:
            self.to_rgbs = nn.Conv2d(channel_out, 32, kernel_size=3, padding=1)

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)


    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding, inplace=True)

        out_rgbs = []
        for i, block in enumerate(self.blocks):
            embedding = block(embedding)
            embedding = F.silu(embedding, inplace=True)

        if self.return_rgbs:
            out_rgbs = self.to_rgbs(embedding)

        embedding = self.conv_out(embedding)

        return [embedding, out_rgbs] if self.return_rgbs else [embedding, None]


class Real_IISR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, controlnet_depth=6, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[-1] ** 2 * 2 + 1
        self.begin_ends = []
        context_token = self.first_l
        self.context_token = context_token
        self.begin_ends.append((0, context_token))
        cur = context_token
        self.L = sum(pn**2 for pn in self.patch_nums)
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.last_level_pns = self.patch_nums[-1] ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        
        # 1. input (word) embedding
        quant: VectorQuantizer = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer] = (quant,)
        self.vae = vae_local
        self.vq  = vae_local.quantize
        for p in self.vq.parameters():
            p.requires_grad_(False)

        for n, p in self.vq.named_parameters():
            if any(k in n for k in ['U', 'V', 'lora_gate']):
                p.requires_grad_(True)
        self.con_embedding = ControlNetConditioningEmbedding(self.C, 3, (32, 128, 256, 512, 1536) , return_rgbs=False)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        self.heatmap_embedding = ControlNetConditioningEmbedding(self.C, 1, (32, 128, 256, 512, 1536) , return_rgbs=False)
        self.edge_embedding = ControlNetConditioningEmbedding(self.C, 3, (32, 128, 256, 512, 1536) , return_rgbs=False)
        self.fusion_block = AFF(channels=self.C)
        self.cross_attention = CrossAttention(self.C, self.num_heads)
         
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)     
        

        self.class_emb = nn.Embedding(self.num_classes, self.C)                              
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)

        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))                        
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        rope_patch_nums =  (self.patch_nums[-1], self.patch_nums[-1], self.patch_nums[0], self.patch_nums[1]) +  self.patch_nums[2:]
        self.freqs_cis = precompute_freqs_cis(
            self.C // num_heads, rope_patch_nums
        )

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.controlnet_depth = controlnet_depth
        self.interval = int(np.ceil(self.depth / self.controlnet_depth))
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn_RoPE(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            ) 
            for block_idx in range(depth)
        ])
        
        print(
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + context_token - 1, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L + context_token - 1, self.L + context_token - 1
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )
        print(attn_bias_for_masking.shape)
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        
        self.cb_lora_rank = self.vq.lora_rank
        self.cb_cond = nn.Sequential(
            nn.LayerNorm(self.C, elementwise_affine=False),
            nn.SiLU(inplace=False),
            nn.Linear(self.C, 4*self.cb_lora_rank),
            nn.SiLU(inplace=False),
            nn.Linear(4*self.cb_lora_rank, self.cb_lora_rank)
        )

    def _h_from_sos(self, sos_BLC: torch.Tensor) -> torch.Tensor:
        g = sos_BLC.mean(dim=1)
        return self.cb_cond(g)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()


    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, text_hidden, lr_inp, negative_text, label_B,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, 
        more_smooth=False, lr_inp_scale=None, tile_flag=False,
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        lr_cond, _ = self.con_embedding(lr_inp)
        lr_cond = lr_cond.view(B, self.C, -1).permute(0,2,1)
        assert lr_cond is not None

        sos = lr_cond
        
        embedded_heatmap,_ = self.heatmap_embedding(compute_thermal_response_map(lr_inp.mean(dim=-3, keepdim=True)))
        embedded_edge,_ = self.edge_embedding(sobel_torch(lr_inp))
        fusion_feature = self.fusion_block(embedded_heatmap, embedded_edge)
        fusion_feature = fusion_feature.view(B, self.C, -1).permute(0,2,1)
        fusion_feature_enhanced, _ = self.cross_attention(sos, fusion_feature)
        sos = torch.cat((sos, fusion_feature_enhanced), dim=1)
        sos = sos.repeat(2, 1, 1)
        assert sos.shape[1] == self.context_token - 1
        
        cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes-1)), dim=0))
        sos = torch.cat((sos, cond_BD.unsqueeze(1)), dim=1)
        h_Br = self._h_from_sos(sos[:B])
        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos.expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        self.freqs_cis = self.freqs_cis.to(dist.get_device())
        
        ms_idx_Bl = []
        
        cur_Lr = 1
        if lr_inp_scale is not None:
            next_token_map[:, -1, :] = next_token_map[:, -1, :] + self.word_embed(lr_inp_scale[:, 0]).repeat(2,1)
        for b in self.blocks: 
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            if si > 0:
                freqs_cis = self.freqs_cis[cur_L:cur_L + pn*pn]
                cur_L += pn * pn
            else:
                freqs_cis = self.freqs_cis[0:self.context_token]
                cur_L += self.context_token

            cond_BD_or_gss = cond_BD
            x = next_token_map
            for i, b in enumerate(self.blocks):
                AdaLNSelfAttn_RoPE.forward
                x = b(x=x, cond_BD=cond_BD_or_gss,  freqs_cis=freqs_cis, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            ms_idx_Bl.append(idx_Bl)
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                if lr_inp_scale is not None:
                    next_token_map = next_token_map + self.word_embed(lr_inp_scale[:, cur_Lr:cur_Lr + self.patch_nums[si+1] ** 2])
                    cur_Lr += self.patch_nums[si+1] ** 2
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: 
            b.attn.kv_caching(False)
        

        if tile_flag:
            assert False, "Unable to reparameterize the codebook!"
            return f_hat
        else:
            return self.vae_proxy[0].idxBl_to_img_lora(ms_idx_Bl, h_Br, same_shape=True, last_one=True).add_(1).mul_(0.5)

    def forward(self, x_BLCv_wo_first_l: torch.Tensor, label_B, lr_inp, text_hidden,
        lr_inp_scale = None,
    ) -> torch.Tensor:  # returns logits_BLV
        bg, ed = (
            self.begin_ends[self.prog_si]
            if self.prog_si >= 0
            else (0, self.L + self.context_token - 1)
        )
        B = x_BLCv_wo_first_l.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            lr_cond, _ = self.con_embedding(lr_inp)
            sos = lr_cond.view(B, self.C, -1).permute(0,2,1)
            
            embedded_heatmap, _ = self.heatmap_embedding(compute_thermal_response_map(lr_inp.mean(dim=-3, keepdim=True)))
            embedded_edge, _ = self.edge_embedding(sobel_torch(lr_inp))
            fusion_feature = self.fusion_block(embedded_heatmap, embedded_edge)
            fusion_feature = fusion_feature.view(B, self.C, -1).permute(0,2,1)
            fusion_feature_enhanced, _ = self.cross_attention(sos, fusion_feature)
            sos = torch.cat((sos, fusion_feature_enhanced), dim=1)
            
            cond_BD = self.class_emb(label_B)
            sos = torch.cat((sos, cond_BD.unsqueeze(1)), dim=1)
            sos = sos.expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                if lr_inp_scale is not None:
                    x_BLC[:, -self.L:, :] = x_BLC[:, -self.L:, :] + self.word_embed(lr_inp_scale.float())
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
    
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = cond_BD
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        self.freqs_cis = self.freqs_cis.to(x_BLC.device)
        freqs_cis = self.freqs_cis.repeat(B, 1, 1)
        
        AdaLNSelfAttn_RoPE.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss,  
                    freqs_cis=freqs_cis, attn_bias=attn_bias)
        x_BLC_logits = self.get_logits(x_BLC.float(), cond_BD)
        x_BLC = x_BLC_logits[:, self.context_token - 1 :, :]
        
        return x_BLC, self._h_from_sos(sos[:B])
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            if hasattr(self.head_nm, 'ada_lin'):
                self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
                if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                    self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            if isinstance(sab, AdaLNSelfAttn_RoPE):
                sab: AdaLNSelfAttn_RoPE
                sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                    nn.init.ones_(sab.ffn.fcg.bias)
                    nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                if hasattr(sab, 'ada_lin'):
                    sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                    sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                    if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                        sab.ada_lin[-1].bias.data.zero_()
                elif hasattr(sab, 'ada_gss'):
                    sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                    sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
            elif isinstance(sab, BaseBlock_RoPE):
                if sab.paca_flag:
                    for sab in [sab.self_attn]:
                        sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                        sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                        if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                            nn.init.ones_(sab.ffn.fcg.bias)
                            nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                        if hasattr(sab, 'ada_lin'):
                            sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                            sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                            if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                                sab.ada_lin[-1].bias.data.zero_()
                        elif hasattr(sab, 'ada_gss'):
                            sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                            sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
