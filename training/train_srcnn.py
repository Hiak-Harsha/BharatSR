"""
BharatSR — SRCNN Training Script (Phase 2)

Trains the SRCNN baseline on preprocessed LR/HR patch pairs.
Designed to run on CPU in ~10-30 minutes with small patches.

Usage:
    python training/train_srcnn.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.srcnn import SRCNN
from training.losses import compute_all_metrics


class SatellitePatchDataset(Dataset):
    """
    Dataset of pre-processed LR/HR satellite patch pairs.
    LR patches are bicubic-upsampled to HR size before yielding.
    """

    def __init__(self, npz_path, scale_factor=4):
        data = np.load(npz_path)
        self.lr = data["lr"]  # (N, C, H_lr, W_lr)
        self.hr = data["hr"]  # (N, C, H_hr, W_hr)
        self.scale_factor = scale_factor
        print(f"Loaded {len(self.lr)} patches from {npz_path}")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx])  # (C, H_lr, W_lr)
        hr = torch.from_numpy(self.hr[idx])  # (C, H_hr, W_hr)
        return lr, hr


def train_srcnn(
    epochs=30,
    batch_size=8,
    lr_rate=1e-3,
    scale_factor=4,
    save_dir=None,
):
    """Train SRCNN baseline model."""

    if save_dir is None:
        save_dir = PROJECT_ROOT / "backend" / "weights"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "processed"

    # Load data
    train_dataset = SatellitePatchDataset(data_dir / "train.npz", scale_factor)
    val_dataset = SatellitePatchDataset(data_dir / "val.npz", scale_factor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Model
    device = torch.device("cpu")  # CPU-only for now
    n_bands = train_dataset.lr.shape[1]
    model = SRCNN(n_bands=n_bands).to(device)
    print(f"\nSRCNN model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    print(f"Bands: {n_bands}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.L1Loss()

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training SRCNN — {epochs} epochs, batch_size={batch_size}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        t0 = time.time()

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # Bicubic upsample LR to HR size
            lr_up = SRCNN.upsample_input(lr_batch, scale_factor)

            # Forward
            sr = model(lr_up)
            loss = criterion(sr, hr_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * lr_batch.size(0)

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                lr_up = SRCNN.upsample_input(lr_batch, scale_factor)
                sr = model(lr_up)
                loss = criterion(sr, hr_batch)
                val_loss += loss.item() * lr_batch.size(0)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = save_dir / "srcnn_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'n_bands': n_bands,
                'scale_factor': scale_factor,
            }, str(checkpoint_path))
            marker = " [best]"
        else:
            marker = ""

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train L1: {train_loss:.6f} | Val L1: {val_loss:.6f} | "
              f"LR: {lr_current:.1e} | {elapsed:.1f}s{marker}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoint saved: {save_dir / 'srcnn_best.pth'}")
    print(f"{'='*60}")

    # Final evaluation with full metrics
    evaluate_model(model, val_dataset, scale_factor, device)

    return model


def evaluate_model(model, val_dataset, scale_factor, device):
    """Evaluate trained model and print comprehensive metrics."""
    model.eval()

    all_metrics = {"psnr_db": [], "ssim": [], "sam_degrees": [],
                   "downsample_consistency_mae": []}

    print(f"\nEvaluating on {len(val_dataset)} validation patches...")

    with torch.no_grad():
        for idx in range(min(len(val_dataset), 50)):  # Cap evaluation
            lr, hr = val_dataset[idx]
            lr_np = lr.numpy()
            hr_np = hr.numpy()

            # Inference
            lr_up = SRCNN.upsample_input(lr.unsqueeze(0), scale_factor)
            sr = model(lr_up).squeeze(0).numpy()

            # Compute metrics
            metrics = compute_all_metrics(sr, hr_np, lr_np, scale_factor)
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k].append(v)

    # Also compute bicubic baseline for comparison
    bicubic_metrics = {"psnr_db": [], "ssim": [], "sam_degrees": []}
    for idx in range(min(len(val_dataset), 50)):
        lr, hr = val_dataset[idx]
        lr_up = SRCNN.upsample_input(lr.unsqueeze(0), scale_factor).squeeze(0).numpy()
        hr_np = hr.numpy()
        m = compute_all_metrics(lr_up, hr_np)
        for k, v in m.items():
            if k in bicubic_metrics:
                bicubic_metrics[k].append(v)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (mean +/- std)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'SRCNN':>15} {'Bicubic':>15}")
    print(f"{'-'*60}")
    for k in ["psnr_db", "ssim", "sam_degrees"]:
        srcnn_mean = np.mean(all_metrics[k])
        srcnn_std = np.std(all_metrics[k])
        bic_mean = np.mean(bicubic_metrics[k])
        bic_std = np.std(bicubic_metrics[k])
        print(f"{k:<30} {srcnn_mean:>8.3f}+/-{srcnn_std:<5.3f} {bic_mean:>8.3f}+/-{bic_std:<5.3f}")

    if all_metrics["downsample_consistency_mae"]:
        dc_mean = np.mean(all_metrics["downsample_consistency_mae"])
        dc_std = np.std(all_metrics["downsample_consistency_mae"])
        print(f"{'downsample_consistency_mae':<30} {dc_mean:>8.5f}+/-{dc_std:<5.5f}")

    print(f"{'='*60}")


def save_comparison_images(model_path=None, n_images=4, scale_factor=4):
    """Save before/after comparison images for visual inspection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_dir = PROJECT_ROOT / "data" / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if model_path is None:
        model_path = PROJECT_ROOT / "backend" / "weights" / "srcnn_best.pth"

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    n_bands = checkpoint.get('n_bands', 4)
    model = SRCNN(n_bands=n_bands)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load validation data
    val_data = np.load(str(PROJECT_ROOT / "data" / "processed" / "val.npz"))
    lr_patches = val_data["lr"]
    hr_patches = val_data["hr"]

    n_images = min(n_images, len(lr_patches))

    for i in range(n_images):
        lr = torch.from_numpy(lr_patches[i]).unsqueeze(0)
        hr = hr_patches[i]

        with torch.no_grad():
            lr_up = SRCNN.upsample_input(lr, scale_factor)
            sr = model(lr_up).squeeze(0).numpy()

        # Convert to RGB for display (clip for display ONLY)
        lr_rgb = np.clip(lr_patches[i][:3].transpose(1, 2, 0), 0, 1)
        sr_rgb = np.clip(sr[:3].transpose(1, 2, 0), 0, 1)
        hr_rgb = np.clip(hr[:3].transpose(1, 2, 0), 0, 1)
        bic_rgb = np.clip(lr_up.squeeze(0).numpy()[:3].transpose(1, 2, 0), 0, 1)

        metrics = compute_all_metrics(sr, hr, lr_patches[i], scale_factor)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(lr_rgb)
        axes[0].set_title(f"LR Input\n{lr_patches[i].shape[1]}×{lr_patches[i].shape[2]}")
        axes[0].axis("off")

        axes[1].imshow(bic_rgb)
        axes[1].set_title(f"Bicubic Upsample\n{hr.shape[1]}×{hr.shape[2]}")
        axes[1].axis("off")

        axes[2].imshow(sr_rgb)
        axes[2].set_title(f"SRCNN Output\nPSNR: {metrics['psnr_db']:.1f}dB, "
                         f"SSIM: {metrics['ssim']:.3f}\n"
                         f"SAM: {metrics['sam_degrees']:.2f}°")
        axes[2].axis("off")

        axes[3].imshow(hr_rgb)
        axes[3].set_title(f"HR Ground Truth\n{hr.shape[1]}×{hr.shape[2]}")
        axes[3].axis("off")

        plt.suptitle("BharatSR — SRCNN Baseline Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()

        path = save_dir / f"srcnn_comparison_{i}.png"
        plt.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BharatSR SRCNN Baseline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args = parser.parse_args()

    model = train_srcnn(epochs=args.epochs, batch_size=args.batch_size, lr_rate=args.lr)
    save_comparison_images()
