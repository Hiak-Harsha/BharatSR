"""
BharatSR — Model Evaluation CLI
Evaluates Bicubic, SRCNN, and RCAN models across held-out test scenes.
Usage: python evaluation/evaluate.py [--data_path data/processed/test.npz]
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import zoom

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.srcnn import SRCNN
from backend.app.models_ml.rcan import RCAN
from training.losses import compute_all_metrics


def run_evaluation(data_path: Path):
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run 'python scripts/prepare_data.py' first.")
        return

    data = np.load(str(data_path))
    lr_arr = data["lr"].astype(np.float32)
    hr_arr = data["hr"].astype(np.float32)
    n_samples = len(lr_arr)
    n_bands = lr_arr.shape[1]
    scale_factor = 4

    print(f"\nEvaluating BharatSR Models on {n_samples} held-out test patches ({data_path.name})...")
    print(f"Canonical degradation operator: 4x4 area average aligned with LR grid.")

    models_to_test = {}

    # 1. Bicubic Baseline
    models_to_test["Bicubic"] = None

    # 2. SRCNN Baseline
    srcnn_path = PROJECT_ROOT / "backend" / "weights" / "srcnn_best.pth"
    if srcnn_path.exists():
        srcnn = SRCNN(n_bands=n_bands)
        chk = torch.load(str(srcnn_path), map_location="cpu", weights_only=False)
        srcnn.load_state_dict(chk["model_state_dict"])
        srcnn.eval()
        models_to_test["SRCNN"] = srcnn

    # 3. RCAN Model
    rcan_path = PROJECT_ROOT / "backend" / "weights" / "rcan_best.pth"
    if rcan_path.exists():
        chk = torch.load(str(rcan_path), map_location="cpu", weights_only=False)
        rcan = RCAN(
            n_bands=n_bands,
            n_feats=chk.get("n_feats", 36),
            n_resgroups=chk.get("n_resgroups", 3),
            n_resblocks=chk.get("n_resblocks", 3),
            scale=scale_factor,
            predict_uncertainty=True,
        )
        rcan.load_state_dict(chk["model_state_dict"])
        rcan.eval()
        models_to_test["RCAN"] = rcan

    results = {}
    metric_keys = ["psnr_db", "ssim", "sam_degrees", "downsample_consistency_mae", "spectral_mae", "hallucination_rate", "correctness_score"]

    for name, model in models_to_test.items():
        metrics_accum = {k: [] for k in metric_keys}

        for i in range(n_samples):
            lr_i = lr_arr[i]
            hr_i = hr_arr[i]

            if model is None:
                # Bicubic
                c, h_lr, w_lr = lr_i.shape
                sr_i = np.zeros((c, h_lr * scale_factor, w_lr * scale_factor), dtype=np.float32)
                for b in range(c):
                    sr_i[b] = zoom(lr_i[b], zoom=scale_factor, order=3)
                sr_i = np.clip(sr_i, 0.0, None)
            else:
                with torch.no_grad():
                    lr_t = torch.from_numpy(lr_i).unsqueeze(0)
                    out = model(lr_t)
                    if isinstance(out, tuple):
                        sr_t = out[0]
                    else:
                        sr_t = out
                    sr_i = sr_t.squeeze(0).numpy()

            m = compute_all_metrics(sr_i, hr_i, lr_i, scale_factor)
            for k in metric_keys:
                if k in m:
                    metrics_accum[k].append(m[k])

        results[name] = {
            k: {
                "mean": round(float(np.mean(metrics_accum[k])), 4),
                "std": round(float(np.std(metrics_accum[k])), 4),
                "median": round(float(np.median(metrics_accum[k])), 4),
            }
            for k in metric_keys
        }

    # Print clean comparison table
    print(f"\n{'='*95}")
    print(f"{'Model':<12} | {'PSNR (dB)':<14} | {'SSIM':<12} | {'SAM (°)':<10} | {'DC-MAE':<10} | {'Correctness':<12} | {'Hallucination':<12}")
    print(f"{'-'*95}")
    for name, m in results.items():
        psnr_str = f"{m['psnr_db']['mean']} ± {m['psnr_db']['std']}"
        ssim_str = f"{m['ssim']['mean']} ± {m['ssim']['std']}"
        sam_str = f"{m['sam_degrees']['mean']}°"
        dc_str = f"{m['downsample_consistency_mae']['mean']}"
        corr_str = f"{m['correctness_score']['mean']}"
        hall_str = f"{m['hallucination_rate']['mean']}"
        print(f"{name:<12} | {psnr_str:<14} | {ssim_str:<12} | {sam_str:<10} | {dc_str:<10} | {corr_str:<12} | {hall_str:<12}")
    print(f"{'='*95}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BharatSR models")
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "test.npz"))
    args = parser.parse_args()
    run_evaluation(Path(args.data_path))
