"""
BharatSR — Residual Channel Attention Network (RCAN) (Phase 5)
Adapted for 4-band satellite imagery (RGB + NIR) with optional uncertainty head.

Architecture:
- Shallow feature extraction: Conv(4 -> n_feats)
- Residual in Residual (RIR): n_resgroups Residual Groups, each with n_resblocks RCABs
- Channel Attention (CA) in each RCAB exploits inter-band and inter-feature dependencies
- Sub-pixel convolution upsampler (PixelShuffle 4x)
- Dual-head:
    - Primary head: Super-resolved physical reflectance image (4 bands)
    - Uncertainty head (optional): Per-pixel log-variance map (1 band)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) Layer.
    Squeezes spatial context via global pooling, excites channel weights via MLP.
    """

    def __init__(self, num_features: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.conv_du(self.avg_pool(x))
        return x * weights


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, num_features: int, reduction: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """Residual Group (RG) containing multiple RCABs with a shortcut connection."""

    def __init__(self, num_features: int, num_blocks: int = 3, reduction: int = 8):
        super().__init__()
        modules = [RCAB(num_features, reduction) for _ in range(num_blocks)]
        modules.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)


class RCAN(nn.Module):
    """
    Complete RCAN adapted for satellite super-resolution.

    Args:
        n_bands: Number of input/output spectral bands (default 4 for R, G, B, NIR)
        n_feats: Number of intermediate feature channels (default 36 for fast CPU inference)
        n_resgroups: Number of residual groups (default 3)
        n_resblocks: Number of RCABs per group (default 3)
        scale: Upsampling factor (default 4)
        predict_uncertainty: If True, outputs (sr, log_var)
    """

    def __init__(
        self,
        n_bands: int = 4,
        n_feats: int = 36,
        n_resgroups: int = 3,
        n_resblocks: int = 3,
        reduction: int = 8,
        scale: int = 4,
        predict_uncertainty: bool = True,
        residual_learning: bool = True,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.scale = scale
        self.predict_uncertainty = predict_uncertainty
        self.residual_learning = residual_learning

        # 1. Shallow feature extraction
        self.head = nn.Conv2d(n_bands, n_feats, kernel_size=3, padding=1, bias=True)

        # 2. Deep feature extraction (RIR)
        modules = [ResidualGroup(n_feats, n_resblocks, reduction) for _ in range(n_resgroups)]
        modules.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

        # 3. Sub-pixel convolution upsampler
        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale),
        )

        # 4. Primary reconstruction head (Learned residual or direct reflectance)
        self.sr_head = nn.Conv2d(n_feats, n_bands, kernel_size=3, padding=1, bias=True)

        # 5. Uncertainty head (predicts bounded log-variance map)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // 2, 1, kernel_size=3, padding=1, bias=True),
            )
        else:
            self.uncertainty_head = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input LR tensor (B, C, H, W)
        Returns:
            sr: Super-resolved tensor (B, C, H*scale, W*scale)
            log_var: Uncertainty tensor (B, 1, H*scale, W*scale) if predict_uncertainty=True
        """
        feats_shallow = self.head(x)
        feats_deep = self.body(feats_shallow) + feats_shallow
        feats_up = self.upsampler(feats_deep)

        residual = self.sr_head(feats_up)

        if self.residual_learning:
            # Residual learning: bicubic(LR) anchor + learned high-frequency residual
            x_up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
            sr = x_up + residual
        else:
            sr = residual

        if self.predict_uncertainty and self.uncertainty_head is not None:
            log_var = self.uncertainty_head(feats_up)
            # Numerically clamp log_variance to prevent infinite uncertainty or collapsed precision
            log_var = torch.clamp(log_var, -6.0, 6.0)
            return sr, log_var

        return sr


def profile_model(model: nn.Module, input_size=(1, 4, 64, 64)) -> dict:
    """Compute parameter count, latency, output range, and estimated FLOPs."""
    import time
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(*input_size)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model(x)
        latency = (time.time() - t0) * 1000  # ms

    sr = out[0] if isinstance(out, tuple) else out
    # Approximate FLOPs for conv2d layers
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # FLOPs = 2 * C_in * C_out * K_h * K_w * H_out * W_out
            h_out = input_size[2] * (model.scale if hasattr(model, "scale") else 1)
            w_out = input_size[3] * (model.scale if hasattr(model, "scale") else 1)
            flops += 2 * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * h_out * w_out

    return {
        "parameters": num_params,
        "trainable_parameters": trainable_params,
        "cpu_latency_ms": round(latency, 2),
        "output_shape": list(sr.shape),
        "estimated_mflops": round(flops / 1e6, 2),
    }


if __name__ == "__main__":
    # Smoke test
    model = RCAN(n_bands=4, n_feats=36, n_resgroups=3, n_resblocks=3, scale=4, predict_uncertainty=True, residual_learning=True)
    stats = profile_model(model)
    print(f"RCAN Satellite Model initialized: {stats['parameters']:,} parameters")
    print(f"Estimated MFLOPs: {stats['estimated_mflops']} MFLOPs | Latency: {stats['cpu_latency_ms']} ms")

    x = torch.randn(2, 4, 32, 32)
    sr, log_var = model(x)
    print(f"Input shape:       {x.shape}")
    print(f"SR Output shape:   {sr.shape} (Expected: [2, 4, 128, 128])")
    print(f"LogVar shape:      {log_var.shape} (Expected: [2, 1, 128, 128])")
    assert sr.shape == (2, 4, 128, 128)
    assert log_var.shape == (2, 1, 128, 128)
    print("[PASS] RCAN architecture test passed.")

    print("RCAN architecture test PASSED [OK]")
