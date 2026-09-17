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

        from training.losses import compute_spectral_mae, compute_gradient_similarity, compute_hallucination_and_correctness
        spec_mae = compute_spectral_mae(sr, hr)
        grad_sim = compute_gradient_similarity(sr, hr)
        halluc_metrics = compute_hallucination_and_correctness(sr, hr)

        metrics["psnr"] = {
            "value": round(psnr, 2),
            "unit": "dB",
            "description": "Peak Signal-to-Noise Ratio. Higher = better reconstruction.",
            "quality": "good" if psnr > 30 else "fair" if psnr > 25 else "poor"
        }

        metrics["ssim"] = {
            "value": round(ssim, 4),
            "unit": "",
            "description": "Structural Similarity Index. Range [0,1]. Higher = better structural preservation.",
            "quality": "good" if ssim > 0.80 else "fair" if ssim > 0.7 else "poor"
        }

        metrics["sam"] = {
            "value": round(sam, 2),
            "unit": "degrees",
            "description": "Spectral Angle Mapper. Lower = better spectral fidelity. Target: < 5.0°.",
            "quality": "good" if sam < 5.0 else "fair" if sam < 10.0 else "poor"
        }

        metrics["spectral_mae"] = {
            "value": round(spec_mae, 6),
            "unit": "reflectance",
            "description": "Mean Absolute Error across all 4 spectral bands.",
            "quality": "good" if spec_mae < 0.02 else "fair" if spec_mae < 0.05 else "poor"
        }

        metrics["gradient_similarity"] = {
            "value": round(grad_sim, 4),
            "unit": "",
            "description": "Spatial gradient edge similarity between SR and HR.",
            "quality": "good" if grad_sim > 0.85 else "fair" if grad_sim > 0.70 else "poor"
        }

        metrics["hallucination_fidelity"] = {
            "false_edge_rate": halluc_metrics["false_edge_rate"],
            "missing_edge_rate": halluc_metrics["missing_edge_rate"],
            "high_freq_hallucination_rate": halluc_metrics["high_freq_hallucination_rate"],
            "correctness_score": halluc_metrics["correctness_score"],
            "consistency_score": halluc_metrics["consistency_score"],
            "synthesis_score": halluc_metrics["synthesis_score"],
            "description": "Quantitative hallucination vs fidelity assessment."
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
