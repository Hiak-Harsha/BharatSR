"""
BharatSR — Loss Functions and Metrics (Phase 2+)

Loss functions:
- L1 reconstruction loss (Phase 2 baseline)
- Spectral consistency loss (Phase 5)
- Heteroscedastic NLL loss for uncertainty (Phase 5)

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SAM (Spectral Angle Mapper) — measures spectral fidelity
- Downsample consistency error

CRITICAL: All metrics operate on physical reflectance values [0, ~1+].
No ImageNet normalization anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ========================
# LOSS FUNCTIONS
# ========================

class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss on reflectance values."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        return self.l1(sr, hr)


class SpectralConsistencyLoss(nn.Module):
    """
    Downsample SR output back to LR resolution and compare against original LR input.
    This enforces that the SR output is physically consistent with the input.

    Phase 5 loss — placeholder interface available from Phase 2.
    """

    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, sr, lr_original):
        """
        Args:
            sr: super-resolved output (B, C, H*scale, W*scale)
            lr_original: original LR input (B, C, H, W)
        """
        # Downsample SR back to LR resolution
        sr_downsampled = F.interpolate(
            sr, size=lr_original.shape[2:],
            mode='bilinear', align_corners=False
        )
        return F.l1_loss(sr_downsampled, lr_original)


class UncertaintyLoss(nn.Module):
    """
    Heteroscedastic negative log-likelihood loss for uncertainty estimation.
    The model predicts mean (SR image) and log-variance (uncertainty).

    NLL = 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var

    Phase 5 loss.
    """

    def forward(self, mean, log_var, target):
        precision = torch.exp(-log_var)
        loss = 0.5 * precision * (target - mean) ** 2 + 0.5 * log_var
        return loss.mean()


class BharatSRCombinedLoss(nn.Module):
    """
    Combined Physics-Constrained Loss for BharatSR RCAN.
    Combines:
    1. Heteroscedastic NLL loss on predicted reflectance and log-variance
    2. Spectral consistency loss (downsampled SR vs LR input)
    3. Direct L1 regularization
    """

    def __init__(self, scale_factor: int = 4, spectral_weight: float = 0.1, l1_weight: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight
        self.uncertainty_loss = UncertaintyLoss()
        self.spectral_loss = SpectralConsistencyLoss(scale_factor)
        self.l1 = nn.L1Loss()

    def forward(self, sr, log_var, hr, lr):
        # 1. Uncertainty NLL loss
        nll = self.uncertainty_loss(sr, log_var, hr)

        # 2. Spectral consistency loss
        spec = self.spectral_loss(sr, lr)

        # 3. Direct L1 loss
        l1 = self.l1(sr, hr)

        total = nll + self.spectral_weight * spec + self.l1_weight * l1
        return total, {"nll": nll.item(), "spectral": spec.item(), "l1": l1.item()}


# ========================
# METRICS
# ========================

def compute_psnr(sr, hr, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical good values for SR: 28-35 dB.

    Args:
        sr, hr: numpy arrays (C, H, W) or (H, W)
        max_val: maximum possible value (1.0 for reflectance)
    """
    mse = np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)


def compute_ssim(sr, hr, win_size=7):
    """
    Structural Similarity Index.
    Higher is better. Range [0, 1]. Good SR: > 0.8.

    Simplified implementation operating on each band independently.
    """
    if sr.ndim == 3:
        # Multi-band: average SSIM across bands
        return np.mean([compute_ssim(sr[b], hr[b], win_size) for b in range(sr.shape[0])])

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)

    # Use a simple uniform window
    kernel_size = win_size
    pad = kernel_size // 2

    # Compute means using uniform filter
    from scipy.ndimage import uniform_filter
    mu_sr = uniform_filter(sr, size=kernel_size)
    mu_hr = uniform_filter(hr, size=kernel_size)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = uniform_filter(sr ** 2, size=kernel_size) - mu_sr_sq
    sigma_hr_sq = uniform_filter(hr ** 2, size=kernel_size) - mu_hr_sq
    sigma_sr_hr = uniform_filter(sr * hr, size=kernel_size) - mu_sr_hr

    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

    return float(np.mean(ssim_map))


def compute_sam(sr, hr):
    """
    Spectral Angle Mapper (SAM).
    Measures the spectral angle between predicted and reference spectra.
    Lower is better. Units: degrees. Good SR: < 5°.

    This metric distinguishes "looks sharp" from "is spectrally correct."

    Args:
        sr, hr: numpy arrays (C, H, W)
    Returns:
        Mean SAM in degrees across all pixels.
    """
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)

    # Compute per-pixel spectral angle
    # dot product along spectral dimension
    dot = np.sum(sr * hr, axis=0)
    norm_sr = np.sqrt(np.sum(sr ** 2, axis=0))
    norm_hr = np.sqrt(np.sum(hr ** 2, axis=0))

    # Avoid division by zero
    denom = norm_sr * norm_hr
    denom = np.maximum(denom, 1e-10)

    cos_angle = np.clip(dot / denom, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(np.mean(angle_deg))


def compute_downsample_consistency(sr, lr_original, scale_factor=4):
    """
    Downsample-consistency error.
    Downsample the SR output and compare pixel-wise against the original LR input.

    This is the key explainability metric: if SR is consistent,
    downsampling it should closely reproduce the original input.

    Args:
        sr: (C, H_hr, W_hr) super-resolved output
        lr_original: (C, H_lr, W_lr) original LR input
    Returns:
        Mean absolute error between downsampled SR and original LR.
    """
    # Downsample SR to LR resolution via area averaging
    # Fast vectorized area averaging in NumPy
    c, h_hr, w_hr = sr.shape
    _, h_lr, w_lr = lr_original.shape

    h_trim = min(h_lr * scale_factor, h_hr)
    w_trim = min(w_lr * scale_factor, w_hr)
    sr_trimmed = sr[:, :h_trim, :w_trim]

    h_out = h_trim // scale_factor
    w_out = w_trim // scale_factor

    sr_down = sr_trimmed[:, :h_out * scale_factor, :w_out * scale_factor].reshape(
        c, h_out, scale_factor, w_out, scale_factor
    ).mean(axis=(2, 4))

    # Match target LR shape if needed
    if sr_down.shape != lr_original.shape:
        pad_h = lr_original.shape[1] - sr_down.shape[1]
        pad_w = lr_original.shape[2] - sr_down.shape[2]
        if pad_h > 0 or pad_w > 0:
            sr_down = np.pad(sr_down, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')

    mae = np.mean(np.abs(sr_down - lr_original))
    return float(mae), sr_down


def compute_all_metrics(sr, hr, lr_original=None, scale_factor=4):
    """
    Compute all metrics for a single image.

    Args:
        sr: (C, H, W) super-resolved output
        hr: (C, H, W) high-resolution ground truth
        lr_original: (C, H_lr, W_lr) original LR input (optional)
    Returns:
        dict of metric name → value
    """
    metrics = {
        "psnr_db": compute_psnr(sr, hr),
        "ssim": compute_ssim(sr, hr),
        "sam_degrees": compute_sam(sr, hr),
    }

    if lr_original is not None:
        dc_error, _ = compute_downsample_consistency(sr, lr_original, scale_factor)
        metrics["downsample_consistency_mae"] = dc_error

    return metrics


if __name__ == "__main__":
    # Quick test with random data
    np.random.seed(42)
    hr = np.random.rand(4, 256, 256).astype(np.float32) * 0.8
    sr = hr + np.random.randn(4, 256, 256).astype(np.float32) * 0.05
    lr = np.random.rand(4, 64, 64).astype(np.float32) * 0.8

    metrics = compute_all_metrics(sr, hr, lr)

    print("Metrics test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nMetrics computation OK [OK]")
