import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Optional
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable, Tuple, List

from typing import Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalOrderConsistencyLoss(nn.Module):
    def __init__(
        self,
        offsets: Optional[Iterable[Tuple[int, int]]] = None,
        patch_size: int = 8,
        threshold: float = 0.1,
        reduction: str = "mean",
        detach_target: bool = True,
    ):
        super().__init__()
        if offsets is None:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.offsets: List[Tuple[int, int]] = list(offsets)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2 and patch_size[0] > 0 and patch_size[1] > 0
        self.patch_size: Tuple[int, int] = (int(patch_size[0]), int(patch_size[1]))

        self.threshold = float(threshold)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.detach_target = detach_target

    @staticmethod
    def _shift_with_border(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        y = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        if dy > 0:
            y[:, :, :dy, :] = x[:, :, :1, :].expand(-1, -1, dy, -1)
        elif dy < 0:
            y[:, :, dy:, :] = x[:, :, -1:, :].expand(-1, -1, -dy, -1)
        if dx > 0:
            y[:, :, :, :dx] = x[:, :, :, :1].expand(-1, -1, -1, dx)
        elif dx < 0:
            y[:, :, :, dx:] = x[:, :, :, -1:].expand(-1, -1, -1, -dx)
        return y

    @staticmethod
    def to_intensity(x, mode="avg"):
        if x.size(1) == 1:
            return x
        if mode == "avg":
            return x.mean(dim=1, keepdim=True)
        elif mode == "y":
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            return 0.299 * r + 0.587 * g + 0.114 * b
        else:
            raise ValueError("Unknown intensity mode")

    def _pool_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.patch_size
        return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))

    def forward(self, tar: torch.Tensor, sr: torch.Tensor) -> torch.Tensor:
        assert tar.shape == sr.shape and tar.dim() == 4, "tar/sr must be [N,C,H,W] with same shape"

        tar_i = self.to_intensity(tar, mode="avg")
        sr_i  = self.to_intensity(sr,  mode="avg")

        tar_p = self._pool_to_patches(tar_i)
        sr_p  = self._pool_to_patches(sr_i)

        if self.detach_target:
            tar_p = tar_p.detach()

        losses = []
        for (dy, dx) in self.offsets:
            tar_n = self._shift_with_border(tar_p, dy, dx)
            sr_n  = self._shift_with_border(sr_p, dy, dx)

            d_tar = tar_p - tar_n
            d_sr  = sr_p  - sr_n

            if self.threshold > 0:
                weight = (d_tar.abs() >= self.threshold).float()
            else:
                weight = 1.0

            term = F.relu(-(d_sr * d_tar)) * weight
            losses.append(term)

        loss = torch.stack(losses, dim=0).mean(dim=0)

        if self.reduction == "mean":
            per_img = loss.flatten(1).sum(dim=1)
            return per_img.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
