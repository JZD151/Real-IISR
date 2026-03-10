import torch
import torch.nn.functional as F

def sobel_torch(x):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1,  2,  1],
                            [0,  0,  0],
                            [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    gx = F.conv2d(x, sobel_x.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sobel_y.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    
    grad = torch.sqrt(gx ** 2 + gy ** 2)
    return grad

