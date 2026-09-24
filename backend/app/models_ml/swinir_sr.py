"""
BharatSR — SwinIR Super-Resolution (Transformer Architecture)
Adapted from Liang et al. "SwinIR: Image Restoration Using Swin Transformer"
Tailored for 4-band Sentinel-2 satellite reflectance (B2, B3, B4, B8).

Key adaptations:
- Native 4-band multispectral input/output
- Physical reflectance preserving — no ImageNet normalization
- Shifted-window self-attention with relative position encoding
- Heteroscedastic uncertainty head for physics-anchored confidence
- Residual anchor: SR = bicubic(LR) + learned deep feature residual
"""

import math
from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    except ImportError:
        def to_2tuple(x):
            return (x, x) if isinstance(x, int) else tuple(x)

        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            with torch.no_grad():
                return tensor.normal_(mean, std).clamp_(a, b)

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # Coordinate grid for relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with alternating standard and shifted window attention."""
    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = None
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
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
        res = res.transpose(1, 2).view(B, C, H, W)
        res = self.conv(res)
        res = res.flatten(2).transpose(1, 2)
        return x + res


class SwinIR_SR(nn.Module):
    """
    SwinIR adapted for 4-Band Sentinel-2 Super-Resolution with Uncertainty Head.
    """
    def __init__(
        self,
        n_bands: int = 4,
        embed_dim: int = 60,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 4,
        scale: int = 4,
        mlp_ratio: float = 2.0,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        depths = depths or [6, 6, 6]
        num_heads = num_heads or [6, 6, 6]
        self.n_bands = n_bands
        self.scale = scale
        self.predict_uncertainty = predict_uncertainty
        self.embed_dim = embed_dim

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(n_bands, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction via Residual Swin Transformer Blocks
        self.rstb_layers = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                input_resolution=(32, 32),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. High quality upsampling head (Sub-pixel PixelShuffle)
        if scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            )
        elif scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.PReLU(),
            )

        # 4. Reconstruction Head (reflectance output)
        self.conv_last = nn.Conv2d(embed_dim, n_bands, 3, 1, 1)

        # 5. Clamped heteroscedastic uncertainty head
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
                nn.PReLU(),
                nn.Conv2d(embed_dim // 2, 1, 3, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        H, W = x.shape[2], x.shape[3]
        x_size = (H, W)

        # Residual anchor: bicubic baseline
        bicubic_base = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # Shallow feature
        fea = self.conv_first(x)

        # RSTB deep features
        b, c, h, w = fea.shape
        tokens = fea.flatten(2).transpose(1, 2)
        for rstb in self.rstb_layers:
            tokens = rstb(tokens, x_size)
        tokens = self.norm(tokens)
        fea_deep = tokens.transpose(1, 2).view(b, c, h, w)
        fea_body = self.conv_after_body(fea_deep) + fea

        # Upsample
        up = self.upsample(fea_body)

        # Reconstructed reflectance: bicubic anchor + residual
        sr = bicubic_base + self.conv_last(up)

        if self.predict_uncertainty:
            log_var = self.uncertainty_head(up)
            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
            return sr, log_var
        return sr
