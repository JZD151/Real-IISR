import torch
import torch.nn.functional as F


def compute_thermal_response_map(ir_image, alpha=3.0, eps=1e-5):
    B, _, H, W = ir_image.shape
    mean = ir_image.view(B, -1).mean(dim=1, keepdim=True).view(B, 1, 1, 1)
    std = ir_image.view(B, -1).std(dim=1, keepdim=True).view(B, 1, 1, 1)

    normalized = (ir_image - mean) / (std + eps)

    T_heat = torch.sigmoid(alpha * normalized)

    return T_heat
