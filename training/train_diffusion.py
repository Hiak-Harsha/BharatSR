"""
BharatSR — DiffusionSR Training Script
Trains the lightweight 4-step DDIM conditional super-resolution model
on 4-band Sentinel-2 reflectance imagery.
Produces backend/weights/diffusion_best.pth and training history.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.diffusion_sr import DiffusionSR
from backend.app.models_ml.uncertainty import logvar_to_std
from training.losses import BharatSRCombinedLoss, compute_all_metrics


class SatelliteDataset(Dataset):
    def __init__(self, npz_path, max_samples: int = None):
        npz_path = Path(npz_path)
        data = np.load(str(npz_path))
        lr = data["lr"].astype(np.float32)
        hr = data["hr"].astype(np.float32)
        if max_samples:
            lr = lr[:max_samples]
            hr = hr[:max_samples]
        self.lr = lr
        self.hr = hr
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


def train_diffusion(
    epochs: int = 3,
    batch_size: int = 4,
    lr_rate: float = 3e-4,
    scale_factor: int = 4,
    num_steps: int = 4,
    max_train_samples: int = 40,
    checkpoint_name: str = "diffusion_best.pth",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train DiffusionSR model and save checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir = PROJECT_ROOT / "backend" / "weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    history_dir = PROJECT_ROOT / "training" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "data" / "processed"

    if not (data_dir / "train.npz").exists():
        raise FileNotFoundError(f"Training data not found in {data_dir}.")

    train_dataset = SatelliteDataset(data_dir / "train.npz", max_samples=max_train_samples)
    val_dataset = SatelliteDataset(data_dir / "val.npz", max_samples=20)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DiffusionSR on device: {device}")

    model = DiffusionSR(
        n_bands=4,
        scale=scale_factor,
        num_steps=num_steps,
    ).to(device)

    criterion = BharatSRCombinedLoss(
        scale_factor=scale_factor,
        lambda_rec=1.0,
        lambda_sam=0.05,
        lambda_dc=0.1,
        lambda_unc=0.01,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {"train_history": [], "final_metrics": {}}

    print(f"Starting DiffusionSR training for {epochs} epochs...")
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
                l, _ = criterion(sr_b, lv_b, hr_b, lr_b)
                val_loss += l.item()

                for b_idx in range(len(lr_b)):
                    sr_np = sr_b[b_idx].cpu().numpy()
                    hr_np = hr_b[b_idx].cpu().numpy()
                    lr_np = lr_b[b_idx].cpu().numpy()
                    m = compute_all_metrics(sr_np, hr_np, lr_np, scale_factor=scale_factor)
                    val_psnr += m["psnr_db"]
                    val_ssim += m["ssim"]
                    n_val += 1

        val_loss /= max(1, len(val_loader))
        val_psnr /= max(1, n_val)
        val_ssim /= max(1, n_val)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}dB, Val SSIM: {val_ssim:.4f}")

        history["train_history"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_psnr": round(val_psnr, 2),
            "val_ssim": round(val_ssim, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / checkpoint_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": float(val_loss),
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim),
                "n_bands": 4,
                "scale_factor": scale_factor,
                "num_steps": num_steps,
                "config_name": "DiffusionSR_4Step_DDIM",
            }, str(ckpt_path))
            print(f"  -> Checkpoint saved to {ckpt_path}")

    history["final_metrics"] = {
        "psnr_db": round(val_psnr, 2),
        "ssim": round(val_ssim, 4),
        "val_loss": round(best_val_loss, 4),
    }

    # Save training history
    history_file = history_dir / "diffusion_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_file}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiffusionSR")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=40)
    args = parser.parse_args()

    train_diffusion(epochs=args.epochs, batch_size=args.batch_size, max_train_samples=args.max_samples)
