"""
BharatSR Patch Visualizer — Phase 1
Loads processed patches and saves side-by-side comparison PNGs for manual inspection.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
VIZ_DIR = PROJECT_ROOT / "data" / "visualizations"


def reflectance_to_rgb(img, bands=(0, 1, 2)):
    """
    Convert multi-band reflectance image to displayable RGB.
    Input: (C, H, W) in reflectance [0, ~1+]
    Output: (H, W, 3) in [0, 1] clipped for display only.

    NOTE: We clip to [0, 1] here ONLY for matplotlib display purposes.
    The actual data retains values > 1.0 for bright targets.
    """
    rgb = np.stack([img[b] for b in bands], axis=-1)
    # Clip for display only — data itself is NOT modified
    rgb = np.clip(rgb, 0, 1)
    return rgb


def visualize_patches(n_patches=6):
    """Load and visualize a sample of LR/HR patch pairs."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation set (more useful to inspect)
    val_path = DATA_DIR / "val.npz"
    if not val_path.exists():
        print(f"No validation data found at {val_path}")
        print("Run prepare_data.py first!")
        return

    data = np.load(str(val_path))
    lr_patches = data["lr"]
    hr_patches = data["hr"]

    n_patches = min(n_patches, len(lr_patches))

    print(f"Visualizing {n_patches} patch pairs from validation set...")
    print(f"LR shape: {lr_patches.shape}, HR shape: {hr_patches.shape}")

    for i in range(n_patches):
        lr = lr_patches[i]
        hr = hr_patches[i]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # LR (upsampled for visual comparison)
        lr_rgb = reflectance_to_rgb(lr)
        axes[0].imshow(lr_rgb)
        axes[0].set_title(f"LR Input\n{lr.shape[1]}×{lr.shape[2]}, "
                         f"range [{lr.min():.3f}, {lr.max():.3f}]")
        axes[0].axis("off")

        # HR
        hr_rgb = reflectance_to_rgb(hr)
        axes[1].imshow(hr_rgb)
        axes[1].set_title(f"HR Target\n{hr.shape[1]}×{hr.shape[2]}, "
                         f"range [{hr.min():.3f}, {hr.max():.3f}]")
        axes[1].axis("off")

        # Band histogram
        for b, color, name in [(0, 'r', 'R'), (1, 'g', 'G'),
                                (2, 'b', 'B'), (3, 'orange', 'NIR')]:
            if b < hr.shape[0]:
                axes[2].hist(hr[b].ravel(), bins=50, alpha=0.5,
                           color=color, label=f'{name} band')
        axes[2].set_title("HR Band Histograms")
        axes[2].set_xlabel("Reflectance")
        axes[2].set_ylabel("Count")
        axes[2].legend()
        axes[2].axvline(x=1.0, color='k', linestyle='--', alpha=0.3,
                       label='reflectance=1.0')

        plt.tight_layout()
        save_path = VIZ_DIR / f"patch_pair_{i}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    # Summary statistics
    print(f"\n{'='*50}")
    print(f"PATCH STATISTICS")
    print(f"{'='*50}")
    print(f"Total val patches: {len(lr_patches)}")
    print(f"LR patch size: {lr_patches.shape[2]}×{lr_patches.shape[3]}")
    print(f"HR patch size: {hr_patches.shape[2]}×{hr_patches.shape[3]}")
    print(f"Bands: {lr_patches.shape[1]}")
    print(f"Scale factor: {hr_patches.shape[2] / lr_patches.shape[2]:.0f}x")
    print(f"LR range: [{lr_patches.min():.4f}, {lr_patches.max():.4f}]")
    print(f"HR range: [{hr_patches.min():.4f}, {hr_patches.max():.4f}]")
    n_bright = np.sum(hr_patches > 1.0)
    print(f"Pixels > 1.0 (bright targets): {n_bright} "
          f"({100*n_bright/hr_patches.size:.2f}%)")
    print(f"\nVisualizations saved to: {VIZ_DIR}")


if __name__ == "__main__":
    visualize_patches()
