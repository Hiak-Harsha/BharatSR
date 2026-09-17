"""
BharatSR — RCAN Training Script (Phase 5)

Trains the Residual Channel Attention Network (RCAN) with:
- Dual-head: 4-band SR output + spatial uncertainty map
- Combined Physics Loss: NLL uncertainty + Spectral consistency + L1
- Evaluates PSNR, SSIM, SAM, downsample consistency, and uncertainty distribution.
- Saves checkpoint to backend/weights/rcan_best.pth
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.rcan import RCAN
from backend.app.models_ml.uncertainty import logvar_to_std, generate_uncertainty_heatmap, summarize_uncertainty
from training.losses import BharatSRCombinedLoss, compute_all_metrics


class SatelliteDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(str(npz_path))
        self.lr = data["lr"].astype(np.float32)
        self.hr = data["hr"].astype(np.float32)
        print(f"Loaded {len(self.lr)} patches from {npz_path}")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        return torch.from_numpy(self.lr[idx]), torch.from_numpy(self.hr[idx])


def train_rcan(
    epochs: int = 10,
    batch_size: int = 4,
    lr_rate: float = 5e-4,
    scale_factor: int = 4,
    n_feats: int = 36,
    n_resgroups: int = 3,
    n_resblocks: int = 3,
):
    save_dir = PROJECT_ROOT / "backend" / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data" / "processed"

    train_dataset = SatelliteDataset(data_dir / "train.npz")
    val_dataset = SatelliteDataset(data_dir / "val.npz")

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
        predict_uncertainty=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nInitialized BharatSR RCAN Model: {num_params:,} parameters")
    print(f"Device: {device} | Bands: {n_bands} | Scale: {scale_factor}x")

    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = BharatSRCombinedLoss(scale_factor=scale_factor, spectral_weight=0.1, l1_weight=1.0)

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\n{'='*65}")
    print(f"Training RCAN — {epochs} epochs, batch_size={batch_size}")
    print(f"{'='*65}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_nll = 0.0
        train_spec = 0.0
        train_l1 = 0.0
        t0 = time.time()

        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)

            optimizer.zero_grad()
            sr_b, logvar_b = model(lr_b)

            loss, loss_dict = criterion(sr_b, logvar_b, hr_b, lr_b)
            loss.backward()

            # Gradient clipping to stabilize NLL training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total_loss += loss.item() * len(lr_b)
            train_nll += loss_dict["nll"] * len(lr_b)
            train_spec += loss_dict["spectral"] * len(lr_b)
            train_l1 += loss_dict["l1"] * len(lr_b)

        scheduler.step()
        n_train = len(train_dataset)
        train_total_loss /= n_train
        train_l1 /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0

        with torch.no_grad():
            for lr_b, hr_b in val_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                sr_b, logvar_b = model(lr_b)
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
            checkpoint_path = save_dir / "rcan_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "n_bands": n_bands,
                "scale_factor": scale_factor,
                "n_feats": n_feats,
                "n_resgroups": n_resgroups,
                "n_resblocks": n_resblocks,
            }, str(checkpoint_path))
            marker = " [best]"
        else:
            marker = ""

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_total_loss:7.4f} (L1: {train_l1:6.4f}) | "
              f"Val Loss: {val_loss:7.4f} (L1: {val_l1:6.4f}) | {elapsed:4.1f}s{marker}")

    print(f"\n{'='*65}")
    print(f"RCAN Training Complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved: {save_dir / 'rcan_best.pth'}")
    print(f"{'='*65}")

    # Comprehensive evaluation
    evaluate_rcan(model, val_dataset, scale_factor, device)
    save_rcan_visualizations(model, val_dataset, scale_factor, device)

    return model


def evaluate_rcan(model, val_dataset, scale_factor, device):
    """Evaluate RCAN across PSNR, SSIM, SAM, downsample consistency, and uncertainty."""
    model.eval()
    all_metrics = {
        "psnr_db": [], "ssim": [], "sam_degrees": [],
        "downsample_consistency_mae": []
    }
    uncertainty_stats_list = []

    with torch.no_grad():
        for idx in range(min(len(val_dataset), 50)):
            lr, hr = val_dataset[idx]
            lr_np = lr.numpy()
            hr_np = hr.numpy()

            sr, log_var = model(lr.unsqueeze(0).to(device))
            sr_np = sr.squeeze(0).cpu().numpy()
            logvar_np = log_var.squeeze(0).cpu().numpy()

            # Physical metrics
            m = compute_all_metrics(sr_np, hr_np, lr_np, scale_factor)
            for k, v in m.items():
                if k in all_metrics:
                    all_metrics[k].append(v)

            # Uncertainty calibration
            std_map = logvar_to_std(logvar_np)
            u_stats = summarize_uncertainty(std_map)
            uncertainty_stats_list.append(u_stats["mean_sigma"])

    print(f"\n{'='*65}")
    print(f"RCAN EVALUATION RESULTS (mean +/- std)")
    print(f"{'='*65}")
    for k in ["psnr_db", "ssim", "sam_degrees", "downsample_consistency_mae"]:
        mean_val = np.mean(all_metrics[k])
        std_val = np.std(all_metrics[k])
        print(f"{k:<30} {mean_val:>8.4f} +/- {std_val:<6.4f}")

    if uncertainty_stats_list:
        u_mean = np.mean(uncertainty_stats_list)
        u_std = np.std(uncertainty_stats_list)
        print(f"{'mean_spatial_uncertainty_sigma':<30} {u_mean:>8.4f} +/- {u_std:<6.4f}")
    print(f"{'='*65}")


def save_rcan_visualizations(model, val_dataset, scale_factor, device, n_images=4):
    """Save 4-panel visualizations: LR Input, RCAN SR, Uncertainty Heatmap, Ground Truth."""
    save_dir = PROJECT_ROOT / "data" / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_images = min(n_images, len(val_dataset))
    model.eval()

    with torch.no_grad():
        for i in range(n_images):
            lr, hr = val_dataset[i]
            sr, log_var = model(lr.unsqueeze(0).to(device))
            sr_np = sr.squeeze(0).cpu().numpy()
            logvar_np = log_var.squeeze(0).cpu().numpy()
            lr_np = lr.numpy()
            hr_np = hr.numpy()

            # RGB for visualization (bands 0, 1, 2)
            lr_rgb = np.clip(np.transpose(lr_np[:3], (1, 2, 0)), 0, 1)
            sr_rgb = np.clip(np.transpose(sr_np[:3], (1, 2, 0)), 0, 1)
            hr_rgb = np.clip(np.transpose(hr_np[:3], (1, 2, 0)), 0, 1)

            # Uncertainty heatmap
            std_map = logvar_to_std(logvar_np)[0]
            p_low, p_high = np.percentile(std_map, [2, 98])
            norm_std = np.clip((std_map - p_low) / max(p_high - p_low, 1e-6), 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(lr_rgb)
            axes[0].set_title(f"1. LR Input\n{lr_np.shape[1]}x{lr_np.shape[2]}")
            axes[0].axis("off")

            m = compute_all_metrics(sr_np, hr_np, lr_np, scale_factor)
            axes[1].imshow(sr_rgb)
            axes[1].set_title(f"2. BharatSR RCAN 4x\nPSNR: {m['psnr_db']:.2f}dB, SSIM: {m['ssim']:.3f}\nSAM: {m['sam_degrees']:.2f} deg")
            axes[1].axis("off")

            im_u = axes[2].imshow(norm_std, cmap="magma")
            axes[2].set_title("3. Spatial Uncertainty Map\n(Bright = High Variance/Edges)")
            axes[2].axis("off")
            plt.colorbar(im_u, ax=axes[2], fraction=0.046, pad=0.04)

            axes[3].imshow(hr_rgb)
            axes[3].set_title(f"4. HR Reference Target\n{hr_np.shape[1]}x{hr_np.shape[2]}")
            axes[3].axis("off")

            plt.suptitle(f"BharatSR — RCAN Attention + Uncertainty Quantification (Sample {i})", fontsize=14, fontweight="bold")
            plt.tight_layout()

            out_path = save_dir / f"rcan_comparison_{i}.png"
            plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved RCAN visualization: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BharatSR RCAN Model")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    args = parser.parse_args()

    train_rcan(epochs=args.epochs, batch_size=args.batch_size, lr_rate=args.lr)
