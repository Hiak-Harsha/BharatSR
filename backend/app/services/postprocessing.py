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

import math
from training.losses import (
    compute_psnr, compute_ssim, compute_sam,
    compute_downsample_consistency, compute_all_metrics
)


def _safe_float(val, ndigits=None):
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or np.isnan(f) or math.isinf(f) or np.isinf(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    except Exception:
        return None


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
        dc_val = _safe_float(dc_error, 6)
        metrics["downsample_consistency"] = {
            "value": dc_val,
            "unit": "MAE",
            "description": "Pixel-wise error between downsampled SR and original LR. "
                          "Lower = more physically consistent.",
            "quality": "good" if dc_val is not None and dc_val < 0.01 else "fair" if dc_val is not None and dc_val < 0.05 else "poor"
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

        is_psnr_inf = math.isinf(psnr) or np.isinf(psnr) or psnr > 100
        psnr_val = _safe_float(psnr, 2) if not is_psnr_inf else None
        ssim_val = _safe_float(ssim, 4)
        sam_val = _safe_float(sam, 2)
        spec_mae_val = _safe_float(spec_mae, 6)
        grad_sim_val = _safe_float(grad_sim, 4)

        metrics["psnr"] = {
            "value": psnr_val,
            "unit": "dB",
            "description": "Peak Signal-to-Noise Ratio. Higher = better reconstruction.",
            "quality": "perfect" if is_psnr_inf else ("good" if psnr_val is not None and psnr_val > 30 else "fair" if psnr_val is not None and psnr_val > 25 else "poor"),
            "note": "Undefined — output identical to reference (zero error / infinite PSNR)." if is_psnr_inf else None
        }

        metrics["ssim"] = {
            "value": ssim_val,
            "unit": "",
            "description": "Structural Similarity Index. Range [0,1]. Higher = better structural preservation.",
            "quality": "good" if ssim_val is not None and ssim_val > 0.80 else "fair" if ssim_val is not None and ssim_val > 0.7 else "poor"
        }

        metrics["sam"] = {
            "value": sam_val,
            "unit": "degrees",
            "description": "Spectral Angle Mapper. Lower = better spectral fidelity. Target: < 5.0°.",
            "quality": "good" if sam_val is not None and sam_val < 5.0 else "fair" if sam_val is not None and sam_val < 10.0 else "poor"
        }

        metrics["spectral_mae"] = {
            "value": spec_mae_val,
            "unit": "reflectance",
            "description": "Mean Absolute Error across all 4 spectral bands.",
            "quality": "good" if spec_mae_val is not None and spec_mae_val < 0.02 else "fair" if spec_mae_val is not None and spec_mae_val < 0.05 else "poor"
        }

        metrics["gradient_similarity"] = {
            "value": grad_sim_val,
            "unit": "",
            "description": "Spatial gradient edge similarity between SR and HR.",
            "quality": "good" if grad_sim_val is not None and grad_sim_val > 0.85 else "fair" if grad_sim_val is not None and grad_sim_val > 0.70 else "poor"
        }

        metrics["hallucination_fidelity"] = {
            "false_edge_rate": _safe_float(halluc_metrics.get("false_edge_rate"), 4),
            "missing_edge_rate": _safe_float(halluc_metrics.get("missing_edge_rate"), 4),
            "high_freq_hallucination_rate": _safe_float(halluc_metrics.get("high_freq_hallucination_rate"), 4),
            "correctness_score": _safe_float(halluc_metrics.get("correctness_score"), 4),
            "consistency_score": _safe_float(halluc_metrics.get("consistency_score"), 4),
            "synthesis_score": _safe_float(halluc_metrics.get("synthesis_score"), 4),
            "description": "Quantitative hallucination vs fidelity assessment."
        }

    # Image statistics (always available)
    metrics["sr_stats"] = {
        "min": _safe_float(sr.min(), 4),
        "max": _safe_float(sr.max(), 4),
        "mean": _safe_float(sr.mean(), 4),
        "bright_pixels_pct": _safe_float(np.sum(sr > 1.0) / sr.size * 100, 2),
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
