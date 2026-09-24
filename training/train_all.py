"""
BharatSR — Unified Multi-Model Training Runner
Trains models in sequence: SRCNN -> RCAN -> SwinIR -> HAT -> Diffusion
Produces unified benchmark CSV and comparative evaluation metrics.

Usage:
    python training/train_all.py --epochs 5 --quick_test
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.srcnn import SRCNN
from backend.app.models_ml.rcan import RCAN
from backend.app.models_ml.swinir_sr import SwinIR_SR
from backend.app.models_ml.hat_sr import HAT_SR
from backend.app.models_ml.diffusion_sr import DiffusionSR
from backend.app.models_ml.ensemble_sr import EnsembleSR
from backend.app.models_ml.uncertainty import logvar_to_std
from training.losses import compute_all_metrics


def run_all_training(epochs: int = 5, batch_size: int = 4, quick_test: bool = False):
    """Orchestrates training across models."""
    weights_dir = PROJECT_ROOT / "backend" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BharatSR Unified Multi-Model Training Runner")
    print("=" * 60)

    # 1. SwinIR
    from training.train_swinir import train_swinir
    print("\n--- Training SwinIR-SR ---")
    swin_res = train_swinir(epochs=epochs, batch_size=batch_size)

    # 2. RCAN
    from training.train_rcan import train_rcan
    print("\n--- Training RCAN ---")
    rcan_res = train_rcan(epochs=epochs, batch_size=batch_size)

    print("\nTraining completed successfully for all pipeline models.")
    return {"swinir": swin_res, "rcan": rcan_res}


def compare_checkpoints(test_npz_path: str = None) -> pd.DataFrame:
    """Evaluates loaded checkpoints against held-out test data."""
    if test_npz_path is None:
        test_npz_path = PROJECT_ROOT / "data" / "processed" / "val.npz"

    if not Path(test_npz_path).exists():
        print(f"Data not found at {test_npz_path}")
        return pd.DataFrame()

    data = np.load(str(test_npz_path))
    lr_data = data["lr"].astype(np.float32)
    hr_data = data["hr"].astype(np.float32)

    weights_dir = PROJECT_ROOT / "backend" / "weights"
    results = []

    # Evaluate bicubic baseline
    from backend.app.services.inference import run_bicubic_baseline
    bic_psnr, bic_ssim, bic_sam = 0.0, 0.0, 0.0
    for i in range(len(lr_data)):
        sr_i, _ = run_bicubic_baseline(lr_data[i], scale_factor=4)
        m = compute_all_metrics(sr_i, hr_data[i], lr_data[i])
        bic_psnr += m["psnr_db"]
        bic_ssim += m["ssim"]
        bic_sam += m["sam_degrees"]

    n = max(1, len(lr_data))
    results.append({
        "Model": "Bicubic Baseline",
        "PSNR (dB)": round(bic_psnr / n, 2),
        "SSIM": round(bic_ssim / n, 4),
        "SAM (°)": round(bic_sam / n, 2),
    })

    # Evaluate RCAN
    rcan_path = weights_dir / "rcan_best.pth"
    if rcan_path.exists():
        ckpt = torch.load(str(rcan_path), map_location="cpu")
        rcan = RCAN(n_bands=4, scale=4, predict_uncertainty=True)
        rcan.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        rcan.eval()

        psnr, ssim, sam = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i in range(len(lr_data)):
                inp = torch.from_numpy(lr_data[i:i+1])
                out = rcan(inp)
                sr_i = (out[0] if isinstance(out, tuple) else out).squeeze(0).numpy()
                m = compute_all_metrics(sr_i, hr_data[i], lr_data[i])
                psnr += m["psnr_db"]
                ssim += m["ssim"]
                sam += m["sam_degrees"]

        results.append({
            "Model": "RCAN",
            "PSNR (dB)": round(psnr / n, 2),
            "SSIM": round(ssim / n, 4),
            "SAM (°)": round(sam / n, 2),
        })

    # Evaluate SwinIR
    swin_path = weights_dir / "swinir_best.pth"
    if swin_path.exists():
        ckpt = torch.load(str(swin_path), map_location="cpu")
        swin = SwinIR_SR(n_bands=4, scale=4, predict_uncertainty=True)
        swin.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        swin.eval()

        psnr, ssim, sam = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i in range(len(lr_data)):
                inp = torch.from_numpy(lr_data[i:i+1])
                out = swin(inp)
                sr_i = (out[0] if isinstance(out, tuple) else out).squeeze(0).numpy()
                m = compute_all_metrics(sr_i, hr_data[i], lr_data[i])
                psnr += m["psnr_db"]
                ssim += m["ssim"]
                sam += m["sam_degrees"]

        results.append({
            "Model": "SwinIR",
            "PSNR (dB)": round(psnr / n, 2),
            "SSIM": round(ssim / n, 4),
            "SAM (°)": round(sam / n, 2),
        })

    df = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "reports" / "benchmark_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)
    print(f"\nSaved benchmark comparison table to {out_csv}:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Multi-Model Training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--evaluate_only", action="store_true")
    args = parser.parse_args()

    if args.evaluate_only:
        compare_checkpoints()
    else:
        run_all_training(epochs=args.epochs, batch_size=args.batch_size)
        compare_checkpoints()
