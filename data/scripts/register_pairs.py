"""
BharatSR — Remote-Sensing Sub-Pixel Image Registration & Quality Verification Tool
Quantifies spatial and spectral alignment between Sentinel-2 LR imagery and High-Resolution reference imagery.

CRITICAL SCIENTIFIC PRINCIPLES:
- Never claim sub-pixel co-registration without quantitatively measuring alignment RMSE.
- Measures:
  - Sub-pixel translation (dx, dy in pixels)
  - Registration RMSE in pixels
  - Registration RMSE in meters (RMSE_m = RMSE_px * GSD)
- Rejects poorly aligned pairs exceeding tolerance (e.g. > 0.5 LR pixels).
- Documents:
  - Band correspondence (e.g. B2 Blue, B3 Green, B4 Red, B8 NIR)
  - Reflectance scaling differences
  - Resampling artifacts
  - Spectral response function mismatch
  - Temporal acquisition mismatch
- Outputs registration report: reports/registration_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import correlate2d

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.losses import degrade_canonical_4x, compute_sam


def estimate_subpixel_translation(
    lr_band: np.ndarray,
    hr_band_degraded: np.ndarray,
    upsample_factor: int = 10,
) -> Tuple[float, float, float]:
    """
    Sub-pixel translation estimation via 2D cross-correlation peak fitting.
    Returns:
        (dx, dy, peak_correlation)
    """
    # Mean-center
    ref = hr_band_degraded - np.mean(hr_band_degraded)
    target = lr_band - np.mean(lr_band)

    corr = correlate2d(ref, target, mode="same")
    mid_y, mid_x = corr.shape[0] // 2, corr.shape[1] // 2

    # Find peak index
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    dx_int = float(peak_x - mid_x)
    dy_int = float(peak_y - mid_y)

    # Sub-pixel parabolic interpolation around peak
    try:
        y0 = max(0, min(peak_y, corr.shape[0] - 1))
        x0 = max(0, min(peak_x, corr.shape[1] - 1))

        if 0 < x0 < corr.shape[1] - 1:
            alpha = corr[y0, x0 - 1]
            beta = corr[y0, x0]
            gamma = corr[y0, x0 + 1]
            denom_x = 2.0 * (alpha - 2.0 * beta + gamma)
            delta_x = (alpha - gamma) / denom_x if abs(denom_x) > 1e-7 else 0.0
        else:
            delta_x = 0.0

        if 0 < y0 < corr.shape[0] - 1:
            alpha = corr[y0 - 1, x0]
            beta = corr[y0, x0]
            gamma = corr[y0 + 1, x0]
            denom_y = 2.0 * (alpha - 2.0 * beta + gamma)
            delta_y = (alpha - gamma) / denom_y if abs(denom_y) > 1e-7 else 0.0
        else:
            delta_y = 0.0

        dx = dx_int + float(np.clip(delta_x, -0.5, 0.5))
        dy = dy_int + float(np.clip(delta_y, -0.5, 0.5))
    except Exception:
        dx, dy = dx_int, dy_int

    peak_val = float(np.max(corr) / (np.sqrt(np.sum(ref ** 2) * np.sum(target ** 2)) + 1e-7))
    return dx, dy, peak_val


def compute_registration_rmse(
    lr: np.ndarray,
    hr: np.ndarray,
    scale_factor: int = 4,
    gsd_m: float = 10.0,
) -> Dict[str, float]:
    """
    Measures registration error between LR image and independent HR reference.
    Degrades HR to LR scale using canonical area-averaging and computes alignment metrics.
    """
    hr_deg = degrade_canonical_4x(hr, scale_factor=scale_factor)

    # Trim to matching dimensions if needed
    min_h = min(lr.shape[1], hr_deg.shape[1])
    min_w = min(lr.shape[2], hr_deg.shape[2])
    lr_c = lr[:, :min_h, :min_w]
    hr_c = hr_deg[:, :min_h, :min_w]

    # Measure across NIR (band 3) and Red (band 2) which have sharpest land-cover gradients
    dx_nir, dy_nir, corr_nir = estimate_subpixel_translation(lr_c[3], hr_c[3])
    dx_red, dy_red, corr_red = estimate_subpixel_translation(lr_c[2], hr_c[2])

    dx_mean = (dx_nir + dx_red) / 2.0
    dy_mean = (dy_nir + dy_red) / 2.0

    rmse_px = float(np.sqrt(dx_mean ** 2 + dy_mean ** 2))
    rmse_m = float(rmse_px * gsd_m)

    # Residual intensity MAE and SAM spectral angle
    pixel_mae = float(np.mean(np.abs(hr_c - lr_c)))
    sam_deg = float(compute_sam(hr_c, lr_c))

    return {
        "dx_pixels": round(dx_mean, 4),
        "dy_pixels": round(dy_mean, 4),
        "rmse_pixels": round(rmse_px, 4),
        "rmse_meters": round(rmse_m, 2),
        "peak_cross_correlation": round(float((corr_nir + corr_red) / 2.0), 4),
        "downsample_residual_mae": round(pixel_mae, 6),
        "spectral_angle_error_deg": round(sam_deg, 2),
    }


def register_dataset_pairs(
    dataset_npz_path: Path,
    output_report_path: Optional[Path] = None,
    max_allowed_rmse_px: float = 0.5,
    gsd_m: float = 10.0,
) -> Dict[str, any]:
    """
    Registers and validates LR/HR scene pairs in a dataset archive.
    Rejects pairs exceeding max_allowed_rmse_px.
    """
    data = np.load(str(dataset_npz_path), allow_pickle=True)
    lr_patches = data["lr"].astype(np.float32)
    hr_patches = data["hr"].astype(np.float32)
    n_pairs = len(lr_patches)

    results = []
    accepted_count = 0
    rejected_count = 0

    for idx in range(min(n_pairs, 20)):
        reg = compute_registration_rmse(lr_patches[idx], hr_patches[idx], gsd_m=gsd_m)
        is_accepted = bool(reg["rmse_pixels"] <= max_allowed_rmse_px)
        reg["pair_index"] = idx
        reg["status"] = "ACCEPTED" if is_accepted else "REJECTED"
        if is_accepted:
            accepted_count += 1
        else:
            rejected_count += 1
        results.append(reg)

    mean_rmse_px = float(np.mean([r["rmse_pixels"] for r in results])) if results else 0.0
    mean_rmse_m = float(np.mean([r["rmse_meters"] for r in results])) if results else 0.0

    report = {
        "dataset_file": str(dataset_npz_path.name),
        "evaluation_count": len(results),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "mean_registration_rmse_px": round(mean_rmse_px, 4),
        "mean_registration_rmse_m": round(mean_rmse_m, 2),
        "max_allowed_rmse_px": max_allowed_rmse_px,
        "spectral_band_correspondence": {
            "B2": "Blue (490nm)",
            "B3": "Green (560nm)",
            "B4": "Red (665nm)",
            "B8": "NIR (842nm)",
        },
        "scientific_limitations": {
            "spectral_mismatch": "Sensors may have distinct Spectral Response Functions (SRF) causing minor radiometric variance.",
            "temporal_mismatch": "Acquisition time delta can introduce illumination, solar zenith, and phonological variations.",
            "resampling": "Bicubic / cubic convolution resampling of reference grids introduces high-frequency smoothing.",
        },
        "pairs_detail": results,
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved registration report to {output_report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register and quantify spatial alignment for BharatSR LR/HR pairs")
    parser.add_argument("--data", type=str, default="data/processed/test.npz", help="Path to .npz dataset")
    parser.add_argument("--out", type=str, default="reports/registration_report.json", help="Path to output report")
    parser.add_argument("--max-rmse", type=float, default=0.5, help="Maximum allowed RMSE in pixels")
    args = parser.parse_args()

    data_file = PROJECT_ROOT / args.data
    out_file = PROJECT_ROOT / args.out

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        sys.exit(1)

    rep = register_dataset_pairs(data_file, out_file, max_allowed_rmse_px=args.max_rmse)
    print("============================================================")
    print(" BharatSR Sub-Pixel Co-Registration Verification")
    print("============================================================")
    print(f" Evaluated:             {rep['evaluation_count']} pairs")
    print(f" Accepted:              {rep['accepted_count']} pairs (RMSE <= {rep['max_allowed_rmse_px']} px)")
    print(f" Rejected:              {rep['rejected_count']} pairs")
    print(f" Mean Registration RMSE: {rep['mean_registration_rmse_px']:.4f} pixels ({rep['mean_registration_rmse_m']:.2f} m)")
    print("============================================================")
