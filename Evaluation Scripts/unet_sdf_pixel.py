import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
from einops import rearrange
import math 

def positionalencoding_xy_batched(xy, d_model):
    """
    Sinusoidal positional encoding for batched (x, y) point sequences.

    Args:
        xy (Tensor): shape [B, N, 2], where each entry is (x, y)
        d_model (int): total embedding size, must be divisible by 4

    Returns:
        Tensor: [B, N, d_model] positional embeddings
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")

    B, N, _ = xy.shape
    d_half = d_model // 2
    div_term = torch.exp(torch.arange(0., d_half, 2, device=xy.device) * -(math.log(10000.0) / d_half))  # [d_half/2]
    # xy = (xy / 2.2) * (2 * math.pi)
    pos_x = xy[:, :, 0].unsqueeze(-1)  # [B, N, 1]
    pos_y = xy[:, :, 1].unsqueeze(-1)  # [B, N, 1]

    pe_x = torch.zeros(B, N, d_half, device=xy.device)
    pe_y = torch.zeros(B, N, d_half, device=xy.device)

    pe_x[:, :, 0::2] = torch.sin(pos_x * div_term)
    pe_x[:, :, 1::2] = torch.cos(pos_x * div_term)

    pe_y[:, :, 0::2] = torch.sin(pos_y * div_term)
    pe_y[:, :, 1::2] = torch.cos(pos_y * div_term)

    return torch.cat([pe_x, pe_y], dim=-1)  # [B, N, d_model]
    
class UNet_SDF_Intermediate(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_SDF_Intermediate, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = (DoubleConv(n_channels, 64))
        self.down0 = (Down(64, 128))
        self.down1 = (Down(128, 256))
        self.down2 = (Down(256, 512))
        self.up1 = (Up(512, 256// factor, bilinear))
        self.up2 = (Up(256, 128// factor, bilinear))
        self.up3 = (Up(128, 64// factor, bilinear))
        self.outc = (OutConv(64, 128))
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down0(x1)
        x3 = self.down1(x2)
        x4 = self.down2(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        per_pixel_feature = nn.ReLU()(self.outc(x))
        
        B, C, H, W = x4.shape

        x4_flat = x4.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        # Generate grid of (x, y) positions normalized between 0 and 1
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing='ij'
        )
        pos = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        pos = pos.view(-1, 2).unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]

        # Apply sinusoidal positional encoding (like transformer)
        pos_embed = positionalencoding_xy_batched(pos, C)  # [B, H*W, C]

        # Add positional embeddings to features
        x4_pos = x4_flat + pos_embed  # [B, H*W, C]
        return per_pixel_feature, x4_pos
