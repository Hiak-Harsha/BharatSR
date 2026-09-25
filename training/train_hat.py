"""
BharatSR — HAT-SR (Hybrid Attention Transformer) Training Script
Trains the Hybrid Attention Transformer combining channel attention and shifted-window self-attention
tailored for 4-band Sentinel-2 reflectance super-resolution with heteroscedastic uncertainty.

Usage:
    python training/train_hat.py --epochs 5 --batch_size 4
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.hat_sr import HAT_SR
from backend.app.models_ml.uncertainty import logvar_to_std
from training.losses import BharatSRCombinedLoss, compute_all_metrics


class SatelliteDataset(Dataset):
    def __init__(self, npz_path):
        npz_path = Path(npz_path)
        data = np.load(str(npz_path))
        self.lr = data["lr"].astype(np.float32)
        self.hr = data["hr"].astype(np.float32)
        print(f"Loaded {len(self.lr)} patches from {npz_path.name}")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr_p = self.lr[idx]
        hr_p = self.hr[idx]
        c, h, w = lr_p.shape
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4
        if pad_h > 0 or pad_w > 0:
            lr_p = np.pad(lr_p, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            hr_p = np.pad(hr_p, ((0, 0), (0, pad_h * 4), (0, pad_w * 4)), mode="reflect")
        return torch.from_numpy(lr_p), torch.from_numpy(hr_p)


def train_hat(
    epochs: int = 5,
    batch_size: int = 4,
    lr_rate: float = 3e-4,
    scale_factor: int = 4,
    embed_dim: int = 48,
    depths: list = None,
    num_heads: list = None,
    window_size: int = 4,
    predict_uncertainty: bool = True,
    checkpoint_name: str = "hat_best.pth",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train HAT-SR model on Sentinel-2 dataset."""
    if depths is None:
        depths = [4, 4, 4]
    if num_heads is None:
        num_heads = [4, 4, 4]

    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir = PROJECT_ROOT / "backend" / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data" / "processed"

    if not (data_dir / "train.npz").exists():
        raise FileNotFoundError(f"Training data not found in {data_dir}. Run prepare_data.py first.")

    train_dataset = SatelliteDataset(data_dir / "train.npz")
    val_dataset = SatelliteDataset(data_dir / "val.npz")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training HAT-SR on device: {device}")

    n_bands = 4
    model = HAT_SR(
        n_bands=n_bands,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        scale=scale_factor,
        predict_uncertainty=predict_uncertainty,
    ).to(device)

    criterion = BharatSRCombinedLoss(
        scale_factor=scale_factor,
        lambda_rec=1.0,
        lambda_sam=0.1,
        lambda_dc=0.1,
        lambda_unc=0.2 if predict_uncertainty else 0.0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    print(f"Starting HAT-SR training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for lr_b, hr_b in train_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            optimizer.zero_grad()

            out = model(lr_b)
            sr_b, lv_b = out if isinstance(out, tuple) else (out, None)
            loss, _ = criterion(sr_b, lv_b, hr_b, lr_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        n_val = 0

        with torch.no_grad():
            for lr_b, hr_b in val_loader:
                lr_b, hr_b = lr_b.to(device), hr_b.to(device)
                out = model(lr_b)
                sr_b, lv_b = out if isinstance(out, tuple) else (out, None)
                loss, _ = criterion(sr_b, lv_b, hr_b, lr_b)
                val_loss += loss.item()

                sr_np = sr_b.cpu().numpy()
                hr_np = hr_b.cpu().numpy()
                for i in range(len(sr_np)):
                    metrics = compute_all_metrics(sr_np[i], hr_np[i])
                    val_psnr += metrics["psnr_db"]
                    val_ssim += metrics["ssim"]
                    n_val += 1

        val_loss /= max(1, len(val_loader))
        val_psnr /= max(1, n_val)
        val_ssim /= max(1, n_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / checkpoint_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "n_bands": n_bands,
                "scale_factor": scale_factor,
                "embed_dim": embed_dim,
                "depths": depths,
                "num_heads": num_heads,
                "window_size": window_size,
            }, str(ckpt_path))
            print(f"  [SAVED] New best validation checkpoint -> {ckpt_path.name}")

    return {
        "model": "HAT-SR",
        "best_val_loss": best_val_loss,
        "checkpoint": str(save_dir / checkpoint_name),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BharatSR HAT Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    train_hat(epochs=args.epochs, batch_size=args.batch_size)
