import os
import sys
import glob
import argparse
import numpy as np
import yaml
from PIL import Image
import torch.nn.functional as F
import safetensors.torch
import time
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
import dist
import torch
from torchvision import transforms
import torch.utils.checkpoint
from utils import arg_util, misc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizer, CLIPImageProcessor
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from dataloader.testdataset import TestDataset
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import pyiqa
from skimage import io
from models import Real_IISR, VQVAE, build_var
from tqdm import tqdm

def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

logger = get_logger(__name__, log_level="INFO")

def main(args: arg_util.Args):
    vae_ckpt =  args.vae_model_path
    var_ckpt = args.var_test_path
    args.depth = 24

    vae, var = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, controlnet_depth=args.depth,
        device=dist.get_device(), patch_nums=args.patch_nums, control_patch_nums =args.patch_nums,
        num_classes=1 + 1, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    vae = var.vae
    model_state = torch.load(var_ckpt, map_location='cpu')
    state_dict = model_state['trainer']['var_wo_ddp']
    model_keys = set(var.state_dict().keys())
    state_keys = set(state_dict.keys())
    extra_keys = state_keys - model_keys
    if extra_keys:
        print(f"[Warning] Skip loading extra parameters: {sorted(extra_keys)}")
        for k in extra_keys:
            state_dict.pop(k)
    missing_keys = model_keys - state_keys
    if missing_keys:
        raise KeyError(f"[Error] Missing parameters for keys: {sorted(missing_keys)}")
    var.load_state_dict(state_dict, strict=True)

    vae.eval(), var.eval()


    img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
                

    dataset_val = TestDataset("testset", image_size=args.data_load_reso, tokenizer=None, resize_bak=True)
    ld_val = DataLoader(
        dataset_val, num_workers=8, pin_memory=True,
        batch_size=4, sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )

    for batch in ld_val:
        lr_inp = batch["LR"].to(args.device, non_blocking=True)
        label_B = batch["label_B"].to(args.device, non_blocking=True)
        B = lr_inp.shape[0]

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                recon_B3HW = var.autoregressive_infer_cfg(B=B, cfg=1, top_k=1, top_p=0.75,
                                                    text_hidden=None, lr_inp=lr_inp, negative_text=None, label_B=label_B, lr_inp_scale = None,
                                                    more_smooth=False)
                recon_B3HW = numpy_to_pil(pt_to_numpy(recon_B3HW))

        for idx in range(B):
            image = recon_B3HW[idx]
            if True: 
                validation_image = Image.open(batch['path'][idx].replace("/HR","/LR")).convert("RGB")
                validation_image = validation_image.resize((512, 512))
                image = adain_color_fix(image, validation_image)

            folder_path, ext_path = os.path.split(batch['path'][idx])
            output_name = folder_path.replace("/LR", "/VARPrediction/").replace("/HR", "/VARPrediction/")
            os.makedirs(output_name, exist_ok=True)
            image.save(os.path.join(output_name, ext_path))
    return True


if __name__ == "__main__":
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main(args)
