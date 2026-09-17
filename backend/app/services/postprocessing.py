"""
BharatSR — Postprocessing Service
Metrics computation and result formatting for API responses.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.losses import (
    compute_psnr, compute_ssim, compute_sam,
    compute_downsample_consistency, compute_all_metrics
)


def compute_inference_metrics(
    sr: np.ndarray,
    hr: np.ndarray = None,
    lr_original: np.ndarray = None,
    scale_factor: int = 4,
) -> dict:
    """
    Compute all relevant metrics for an inference result.

    Args:
        sr: (C, H, W) super-resolved output
        hr: (C, H, W) ground truth (if available)
        lr_original: (C, H_lr, W_lr) original LR input
    Returns:
        Dictionary of metrics with descriptions
    """
    metrics = {}

    # Always compute downsample consistency (doesn't need ground truth)
    if lr_original is not None:
        dc_error, sr_downsampled = compute_downsample_consistency(
            sr, lr_original, scale_factor
        )
        metrics["downsample_consistency"] = {
            "value": round(dc_error, 6),
            "unit": "MAE",
            "description": "Pixel-wise error between downsampled SR and original LR. "
                          "Lower = more physically consistent.",
            "quality": "good" if dc_error < 0.01 else "fair" if dc_error < 0.05 else "poor"
        }

    # Compute full metrics if ground truth is available
    if hr is not None:
        psnr = compute_psnr(sr, hr)
        ssim = compute_ssim(sr, hr)
        sam = compute_sam(sr, hr)

        metrics["psnr"] = {
            "value": round(psnr, 2),
            "unit": "dB",
            "description": "Peak Signal-to-Noise Ratio. Higher = better reconstruction. "
                          "Good SR: 28-35 dB.",
            "quality": "good" if psnr > 30 else "fair" if psnr > 25 else "poor"
        }

        metrics["ssim"] = {
            "value": round(ssim, 4),
            "unit": "",
            "description": "Structural Similarity Index. Range [0,1]. Higher = better "
                          "structural preservation. Good SR: > 0.8.",
            "quality": "good" if ssim > 0.85 else "fair" if ssim > 0.7 else "poor"
        }

        metrics["sam"] = {
            "value": round(sam, 2),
            "unit": "degrees",
            "description": "Spectral Angle Mapper. Lower = better spectral fidelity. "
                          "Measures if colors/bands are correct, not just sharp. "
                          "Good SR: < 5°.",
            "quality": "good" if sam < 5 else "fair" if sam < 10 else "poor"
        }

    # Image statistics (always available)
    metrics["sr_stats"] = {
        "min": round(float(sr.min()), 4),
        "max": round(float(sr.max()), 4),
        "mean": round(float(sr.mean()), 4),
        "bright_pixels_pct": round(float(np.sum(sr > 1.0) / sr.size * 100), 2),
    }

    return metrics


def format_metrics_summary(metrics: dict) -> str:
    """Format metrics as a human-readable summary string."""
    lines = []
    for key, val in metrics.items():
        if isinstance(val, dict) and "value" in val:
            unit = val.get("unit", "")
            lines.append(f"{key}: {val['value']} {unit}")
    return " | ".join(lines)
