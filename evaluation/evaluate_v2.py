"""
BharatSR v2 — Comprehensive Model Evaluation
Evaluates all pipeline models on held-out test set.

Metrics computed:
- PSNR (dB), SSIM (band-weighted), SAM (deg)
- Downsample Consistency MAE, Spectral MAE
- Edge diagnostics: false_edge_rate, missing_edge_rate, hallucination_rate, correctness_score, synthesis_score
- Uncertainty calibration: ECE, 68%/95% coverage

Outputs:
- reports/benchmark_v2.csv
- reports/benchmark_v2.json
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.inference import model_registry, run_bicubic_baseline
from training.losses import compute_all_metrics
from evaluation.uncertainty_calibration_v2 import evaluate_uncertainty_comprehensive


def evaluate_v2(data_path: str = None) -> pd.DataFrame:
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "val.npz"
        if (PROJECT_ROOT / "data" / "processed" / "test.npz").exists():
            data_path = PROJECT_ROOT / "data" / "processed" / "test.npz"

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Data not found at {data_path}. Creating synthetic test patches...")
        from data.scripts.prepare_synthetic import create_synthetic_dataset
        create_synthetic_dataset(data_path.parent, num_samples=8, patch_size=32)

    data = np.load(str(data_path))
    lr_arr = data["lr"].astype(np.float32)
    hr_arr = data["hr"].astype(np.float32)
    n_samples = len(lr_arr)

    print(f"\n=== BharatSR v2 Comprehensive Benchmark ({n_samples} scenes) ===")

    models = ["bicubic"]
    for m in ["srcnn", "rcan", "swinir", "hat", "diffusion", "ensemble"]:
        if model_registry.get_model(m) is not None:
            models.append(m)

    benchmark_rows = []

    for model_id in models:
        psnr_list, ssim_list, sam_list, dc_list, sp_mae_list = [], [], [], [], []
        correctness_list, hallucination_list = [], []
        ece_list = []

        print(f"Evaluating model: {model_id}...")
        for i in range(n_samples):
            lr_i = lr_arr[i]
            hr_i = hr_arr[i]

            if model_id == "bicubic":
                sr_i, _ = run_bicubic_baseline(lr_i, scale_factor=4)
                unc_i = None
            else:
                m = model_registry.get_model(model_id)
                with torch.no_grad():
                    inp = torch.from_numpy(lr_i[None]).to(model_registry.device)
                    out = m(inp)
                    if isinstance(out, tuple):
                        sr_t, lv_t = out
                        sr_i = sr_t.squeeze(0).cpu().numpy()
                        unc_i = np.exp(0.5 * np.clip(lv_t.squeeze(0).cpu().numpy(), -6, 6))
                    else:
                        sr_i = out.squeeze(0).cpu().numpy()
                        unc_i = None

            metrics = compute_all_metrics(sr_i, hr_i, lr_i)
            psnr_list.append(metrics["psnr_db"])
            ssim_list.append(metrics["ssim"])
            sam_list.append(metrics["sam_degrees"])
            dc_list.append(metrics.get("downsample_consistency_mae", 0.0))
            sp_mae_list.append(metrics["spectral_mae"])
            correctness_list.append(metrics.get("correctness_score", 0.0))
            hallucination_list.append(metrics.get("hallucination_rate", 0.0))

            if unc_i is not None:
                calib = evaluate_uncertainty_comprehensive(sr_i, hr_i, unc_i)
                ece_list.append(calib["ece"])

        benchmark_rows.append({
            "Model": model_id.upper(),
            "PSNR (dB)": round(float(np.mean(psnr_list)), 2),
            "SSIM": round(float(np.mean(ssim_list)), 4),
            "SAM (°)": round(float(np.mean(sam_list)), 2),
            "DC MAE": round(float(np.mean(dc_list)), 5),
            "Spectral MAE": round(float(np.mean(sp_mae_list)), 5),
            "Correctness": round(float(np.mean(correctness_list)), 3),
            "Hallucination Rate": round(float(np.mean(hallucination_list)), 3),
            "ECE": round(float(np.mean(ece_list)), 4) if len(ece_list) > 0 else "N/A",
        })

    df = pd.DataFrame(benchmark_rows)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "benchmark_v2.csv"
    json_path = reports_dir / "benchmark_v2.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(benchmark_rows, f, indent=2)

    print(f"\nBenchmark completed! Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print("\n" + df.to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BharatSR v2 Models")
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    evaluate_v2(data_path=args.data_path)
