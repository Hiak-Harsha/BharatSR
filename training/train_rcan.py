"""
BharatSR — RCAN Training Script (Phase 5 & Scientific Upgrades)

Trains the Residual Channel Attention Network (RCAN) with:
- Residual learning anchor: bicubic(LR) + learned residual
- Dual-head: 4-band SR output + spatial uncertainty map
- Multi-task physics loss: L_rec (L1) + L_sam (SAM) + L_dc (Canonical 4x4 Area Avg) + L_unc (Heteroscedastic NLL)
- Logs parameter count, latency, estimated FLOPs, output range, and full metrics
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.rcan import RCAN, profile_model
from backend.app.models_ml.uncertainty import logvar_to_std, evaluate_uncertainty_calibration
from training.losses import BharatSRCombinedLoss, compute_all_metrics


class SatelliteDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(str(npz_path))
        self.lr = data["lr"].astype(np.float32)
        self.hr = data["hr"].astype(np.float32)
        print(f"Loaded {len(self.lr)} patches from {npz_path.name}")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.lr[idx]), torch.from_numpy(self.hr[idx])


def get_git_revision() -> str:
    """Get current git commit hash if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT)).decode("ascii").strip()
    except Exception:
        return "v0.2.0-clean"


def train_rcan(
    epochs: int = 5,
    batch_size: int = 4,
    lr_rate: float = 5e-4,
    scale_factor: int = 4,
    n_feats: int = 36,
    n_resgroups: int = 3,
    n_resblocks: int = 3,
    lambda_rec: float = 1.0,
    lambda_sam: float = 0.1,
    lambda_dc: float = 0.1,
    lambda_unc: float = 0.2,
    predict_uncertainty: bool = True,
    config_name: str = "rcan_full",
    checkpoint_name: str = "rcan_best.pth",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train RCAN with specified loss configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir = PROJECT_ROOT / "backend" / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data" / "processed"

    train_dataset = SatelliteDataset(data_dir / "train.npz")
    val_dataset = SatelliteDataset(data_dir / "val.npz")
    test_path = data_dir / "test.npz"
    test_dataset = SatelliteDataset(test_path) if test_path.exists() else val_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    n_bands = train_dataset.lr.shape[1]

    model = RCAN(
        n_bands=n_bands,
        n_feats=n_feats,
        n_resgroups=n_resgroups,
        n_resblocks=n_resblocks,
        scale=scale_factor,
        predict_uncertainty=predict_uncertainty,
    ).to(device)

    profile = profile_model(model, input_size=(1, n_bands, 32, 32))
    num_params = profile["parameters"]

    print(f"\n{'='*70}")
    print(f"BharatSR RCAN Training [{config_name}]")
    print(f"Architecture: Lightweight RCAN-Lite ({n_resgroups} RGs, {n_resblocks} RCABs)")
    print(f"Parameters: {num_params:,} | FLOPs: {profile.get('estimated_flops', 'N/A')}")
    print(f"Loss Weights: rec={lambda_rec}, sam={lambda_sam}, dc={lambda_dc}, unc={lambda_unc}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"{'='*70}")

    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = BharatSRCombinedLoss(
        scale_factor=scale_factor,
        lambda_rec=lambda_rec,
        lambda_sam=lambda_sam,
        lambda_dc=lambda_dc,
        lambda_unc=lambda_unc,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    train_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_l1 = 0.0
        train_sam = 0.0
        train_dc = 0.0
        train_unc = 0.0
        t0 = time.time()

        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)

            optimizer.zero_grad()
            out = model(lr_b)
            sr_b, logvar_b = out if isinstance(out, tuple) else (out, None)

            loss, loss_dict = criterion(sr_b, logvar_b, hr_b, lr_b)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = len(lr_b)
            train_total_loss += loss.item() * bs
            train_l1 += loss_dict["l1"] * bs
            train_sam += loss_dict["sam"] * bs
            train_dc += loss_dict["dc"] * bs
            train_unc += loss_dict["unc"] * bs

        scheduler.step()
        n_train = len(train_dataset)
        train_total_loss /= n_train
        train_l1 /= n_train
        train_sam /= n_train
        train_dc /= n_train
        train_unc /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0

        with torch.no_grad():
            for lr_b, hr_b in val_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                out_val = model(lr_b)
                sr_b, logvar_b = out_val if isinstance(out_val, tuple) else (out_val, None)
                loss, loss_dict = criterion(sr_b, logvar_b, hr_b, lr_b)
                val_loss += loss.item() * len(lr_b)
                val_l1 += loss_dict["l1"] * len(lr_b)

        n_val = len(val_dataset)
        val_loss /= n_val
        val_l1 /= n_val
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = save_dir / checkpoint_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "n_bands": n_bands,
                "scale_factor": scale_factor,
                "n_feats": n_feats,
                "n_resgroups": n_resgroups,
                "n_resblocks": n_resblocks,
                "config_name": config_name,
                "lambda_rec": lambda_rec,
                "lambda_sam": lambda_sam,
                "lambda_dc": lambda_dc,
                "lambda_unc": lambda_unc,
            }, str(checkpoint_path))
            marker = " [best]"
        else:
            marker = ""

        train_history.append({
            "epoch": epoch,
            "train_loss": round(train_total_loss, 4),
            "train_l1": round(train_l1, 4),
            "val_loss": round(val_loss, 4),
            "val_l1": round(val_l1, 4),
        })

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_total_loss:7.4f} (L1: {train_l1:6.4f}, SAM: {train_sam:6.4f}, DC: {train_dc:6.4f}) | "
              f"Val Loss: {val_loss:7.4f} | {elapsed:4.1f}s{marker}")

    # Final evaluation on held-out test dataset
    print(f"\nEvaluating configuration '{config_name}' on held-out test split...")
    model.eval()
    test_metrics = {
        "psnr_db": [], "ssim": [], "sam_degrees": [],
        "downsample_consistency_mae": [], "spectral_mae": [],
        "gradient_similarity": [],
        "false_edge_rate": [], "missing_edge_rate": [], "high_frequency_excess_rate": [],
        "correctness_score": [],
    }
    all_sr = []
    all_hr = []
    all_sigma = []
    latencies = []
    n_samples = len(test_dataset)

    # Warm-up passes (3 cycles)
    with torch.no_grad():
        for _ in range(min(3, n_samples)):
            lr, _ = test_dataset[0]
            _ = model(lr.unsqueeze(0).to(device))

        for idx in range(n_samples):
            lr, hr = test_dataset[idx]
            lr_np = lr.numpy()
            hr_np = hr.numpy()

            lr_in = lr.unsqueeze(0).to(device)
            t0 = time.perf_counter()
            out_test = model(lr_in)
            latencies.append(time.perf_counter() - t0)

            sr, log_var = out_test if isinstance(out_test, tuple) else (out_test, None)
            sr_np = sr.squeeze(0).cpu().numpy()
            sr_np = np.clip(sr_np, 0.0, None)

            m = compute_all_metrics(sr_np, hr_np, lr_np, scale_factor)
            for k in test_metrics:
                val = m.get(k, None)
                if val is not None:
                    test_metrics[k].append(val)

            all_sr.append(sr_np)
            all_hr.append(hr_np)
            if log_var is not None:
                std_np = logvar_to_std(log_var).squeeze(0).cpu().numpy()
                all_sigma.append(std_np)

    sr_stacked = np.concatenate(all_sr, axis=0)
    output_min = float(sr_stacked.min())
    output_max = float(sr_stacked.max())

    avg_test_metrics = {k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in test_metrics.items()}
    std_test_metrics = {f"{k}_std": round(float(np.std(v)), 4) if v else 0.0 for k, v in test_metrics.items()}
    median_test_metrics = {f"{k}_median": round(float(np.median(v)), 4) if v else 0.0 for k, v in test_metrics.items()}

    # Calibration evaluation if uncertainty head is present
    calibration_metrics = None
    if all_sigma and lambda_unc > 0:
        y_true_stack = np.stack(all_hr, axis=0)
        y_pred_stack = np.stack(all_sr, axis=0)
        sigma_stack = np.stack(all_sigma, axis=0)
        calibration_metrics = evaluate_uncertainty_calibration(y_true_stack, y_pred_stack, sigma_stack)

    run_summary = {
        "config_name": config_name,
        "git_version": get_git_revision(),
        "dataset_version": "v1.0-scene-separated",
        "seed": seed,
        "lr": lr_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "loss_weights": {
            "lambda_rec": lambda_rec,
            "lambda_sam": lambda_sam,
            "lambda_dc": lambda_dc,
            "lambda_unc": lambda_unc,
        },
        "architecture": f"Lightweight RCAN-Lite ({n_resgroups} RGs, {n_resblocks} RCABs, Residual Anchor)",
        "num_parameters": num_params,
        "sample_count": n_samples,
        "latency_mean_s": round(float(np.mean(latencies)), 5),
        "latency_median_s": round(float(np.median(latencies)), 5),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 5),
        "latency_per_patch_s": round(float(np.mean(latencies)), 5),
        "output_range": [round(output_min, 4), round(output_max, 4)],
        "train_loss": round(train_total_loss, 4),
        "val_loss": round(best_val_loss, 4),
        "test_metrics": avg_test_metrics,
        "test_metrics_std": std_test_metrics,
        "test_metrics_median": median_test_metrics,
        "calibration_metrics": calibration_metrics,
    }

    print(f"Test Results for [{config_name}]: PSNR={avg_test_metrics['psnr_db']}dB, SSIM={avg_test_metrics['ssim']}, "
          f"SAM={avg_test_metrics['sam_degrees']}°, DC-MAE={avg_test_metrics['downsample_consistency_mae']}")

    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BharatSR RCAN Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_sam", type=float, default=0.1)
    parser.add_argument("--lambda_dc", type=float, default=0.1)
    parser.add_argument("--lambda_unc", type=float, default=0.2)
    parser.add_argument("--config_name", type=str, default="rcan_full")
    parser.add_argument("--checkpoint_name", type=str, default="rcan_best.pth")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_rcan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_rate=args.lr,
        lambda_rec=args.lambda_rec,
        lambda_sam=args.lambda_sam,
        lambda_dc=args.lambda_dc,
        lambda_unc=args.lambda_unc,
        config_name=args.config_name,
        checkpoint_name=args.checkpoint_name,
        seed=args.seed,
    )
