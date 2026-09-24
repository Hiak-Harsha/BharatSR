"""
BharatSR — HAT Super-Resolution (Hybrid Attention Transformer)
Combines channel attention from RCAN with shifted-window self-attention.
Tailored for 4-band Sentinel-2 satellite reflectance (B2, B3, B4, B8).
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.models_ml.swinir_sr import WindowAttention, window_partition, window_reverse, Mlp, DropPath


class ChannelAttention(nn.Module):
    """Channel attention mechanism for global spectral band context."""
    def __init__(self, num_feat, squeeze_factor=4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class HAB(nn.Module):
    """Hybrid Attention Block: Window Self-Attention + Channel Attention."""
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.ca = ChannelAttention(dim, squeeze_factor=4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x

        x_2d = self.norm1(x).transpose(1, 2).view(B, C, H, W)
        ca_out = self.ca(x_2d).flatten(2).transpose(1, 2)

        x_norm = self.norm1(x).view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x_norm.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_norm

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x_win = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_win = shifted_x

        if pad_r > 0 or pad_b > 0:
            x_win = x_win[:, :H, :W, :].contiguous()

        sa_out = x_win.view(B, H * W, C)
        x = shortcut + self.drop_path(sa_out + ca_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG)."""
    def __init__(self, dim, depth, num_heads, window_size=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        H, W = x_size
        res = x
        for blk in self.blocks:
            res = blk(res, x_size)
        B, L, C = res.shape
        res_2d = res.transpose(1, 2).view(B, C, H, W)
        res_2d = self.conv(res_2d)
        return x + res_2d.flatten(2).transpose(1, 2)


class HAT_SR(nn.Module):
    """Hybrid Attention Transformer for 4-Band Satellite Imagery."""
    def __init__(
        self,
        n_bands: int = 4,
        embed_dim: int = 48,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 4,
        scale: int = 4,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        depths = depths or [4, 4, 4]
        num_heads = num_heads or [4, 4, 4]
        self.n_bands = n_bands
        self.scale = scale
        self.predict_uncertainty = predict_uncertainty

        # 1. First convolution
        self.conv_first = nn.Conv2d(n_bands, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction via RHAGs
        self.rhags = nn.ModuleList([
            RHAG(dim=embed_dim, depth=depths[i], num_heads=num_heads[i], window_size=window_size)
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. Sub-pixel upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        # 4. Reconstruction & uncertainty
        self.conv_last = nn.Conv2d(embed_dim, n_bands, 3, 1, 1)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
                nn.PReLU(),
                nn.Conv2d(embed_dim // 2, 1, 3, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        H, W = x.shape[2], x.shape[3]
        bicubic_base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        fea = self.conv_first(x)
        b, c, h, w = fea.shape
        tokens = fea.flatten(2).transpose(1, 2)

        for rhag in self.rhags:
            tokens = rhag(tokens, (h, w))
        tokens = self.norm(tokens)
        fea_deep = tokens.transpose(1, 2).view(b, c, h, w)
        fea_body = self.conv_after_body(fea_deep) + fea

        up = self.upsample(fea_body)
        sr = bicubic_base + self.conv_last(up)

        if self.predict_uncertainty:
            log_var = self.uncertainty_head(up)
            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
            return sr, log_var
        return sr
