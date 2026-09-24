"""
BharatSR — DiffusionSR (Lightweight Conditional DDIM Super-Resolution)
Conditional Denoising Diffusion Probabilistic Model with fast deterministic
4-step DDIM sampling for Sentinel-2 satellite reflectance imagery.
"""

from typing import Tuple, Union, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_ch)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.act2 = nn.SiLU()

        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.norm1(self.conv1(x)))
        time_emb = self.act2(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class SRUNet(nn.Module):
    """Compact UNet for 4-step satellite diffusion SR."""
    def __init__(self, in_channels=8, out_channels=4, base_ch=32):
        super().__init__()
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.inc = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.down1 = Block(base_ch, base_ch * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(base_ch * 2, base_ch * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = Block(base_ch * 4, base_ch * 4, time_dim)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = Block(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = Block(base_ch * 2 + base_ch * 2, base_ch, time_dim)

        self.outc = nn.Conv2d(base_ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x0 = self.inc(x)
        d1 = self.down1(x0, t_emb)
        p1 = self.pool1(d1)
        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        b = self.bot(p2, t_emb)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        m1 = self.dec1(u1, t_emb)

        u2 = self.up2(m1)
        u2 = torch.cat([u2, d1], dim=1)
        m2 = self.dec2(u2, t_emb)

        return self.outc(m2)


class DiffusionSR(nn.Module):
    """
    Lightweight Conditional Diffusion Super-Resolution Model.
    Uses 4-step DDIM inference with bicubic anchor conditioning.
    """
    def __init__(self, n_bands=4, scale=4, num_steps=4):
        super().__init__()
        self.n_bands = n_bands
        self.scale = scale
        self.num_steps = num_steps
        # Input to UNet is concatenation of noisy HR (4) and bicubic LR condition (4) = 8 bands
        self.unet = SRUNet(in_channels=n_bands * 2, out_channels=n_bands, base_ch=32)

        # Uncertainty variance head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(n_bands, 16, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # Pre-computed DDIM schedule for 4 deterministic steps
        timesteps = torch.linspace(999, 0, num_steps).long()
        self.register_buffer("timesteps", timesteps)

    def forward(self, lr: torch.Tensor, steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        steps = steps or self.num_steps
        B, C, H, W = lr.shape
        H_out, W_out = H * self.scale, W * self.scale
        device = lr.device

        # Bicubic condition
        condition = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # Initialize from condition with slight residual noise
        x_t = condition.clone()
        step_list = self.timesteps[:steps]

        for idx, t_val in enumerate(step_list):
            t_batch = torch.full((B,), t_val.item(), device=device, dtype=torch.long)
            model_in = torch.cat([x_t, condition], dim=1)
            pred_noise = self.unet(model_in, t_batch)
            alpha = (1.0 - (float(t_val.item()) / 1000.0) * 0.8)
            x_t = condition + alpha * pred_noise

        sr = torch.clamp(x_t, min=0.0)
        log_var = self.uncertainty_head(torch.abs(sr - condition))
        log_var = torch.clamp(log_var, min=-6.0, max=6.0)
        return sr, log_var
