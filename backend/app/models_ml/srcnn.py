"""
BharatSR — SRCNN Baseline Model (Phase 2)
3-layer CNN for satellite image super-resolution.

Architecture:
  1. Patch extraction: 9×9 conv, 4 → 64 channels
  2. Non-linear mapping: 1×1 conv, 64 → 32 channels
  3. Reconstruction: 5×5 conv, 32 → 4 channels (output bands)

Input: 4-band LR image, bicubic-upsampled to target HR size
Output: 4-band SR image at HR resolution

NOTE: No batch normalization — small network, and BN can interfere
with reflectance value preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    Super-Resolution CNN baseline.
    Expects input already upsampled to target size via bicubic interpolation.
    """

    def __init__(self, n_bands=4):
        super().__init__()
        self.n_bands = n_bands

        # Layer 1: Patch extraction and representation
        self.conv1 = nn.Conv2d(n_bands, 64, kernel_size=9, padding=4)

        # Layer 2: Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)

        # Layer 3: Reconstruction
        self.conv3 = nn.Conv2d(32, n_bands, kernel_size=5, padding=2)

        # ReLU activation (no BN — preserves reflectance scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, n_bands, H, W) — LR image bicubic-upsampled to HR size
        Returns:
            (B, n_bands, H, W) — super-resolved output
        NOTE: Output is NOT clipped to [0, 1]. Bright targets may
              legitimately exceed 1.0 in physical reflectance.
        """
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)  # No activation on output — reflectance can be any positive value
        return out

    @staticmethod
    def upsample_input(lr, scale_factor=4):
        """
        Bicubic upsample LR input to HR size.
        This is the pre-processing step before feeding into SRCNN.
        """
        return F.interpolate(lr, scale_factor=scale_factor,
                           mode='bicubic', align_corners=False)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = SRCNN(n_bands=4)
    print(f"SRCNN parameters: {count_parameters(model):,}")

    # Test forward pass
    lr = torch.randn(1, 4, 64, 64)
    lr_up = SRCNN.upsample_input(lr, scale_factor=4)
    sr = model(lr_up)

    print(f"LR shape:         {lr.shape}")
    print(f"LR upsampled:     {lr_up.shape}")
    print(f"SR output shape:  {sr.shape}")
    assert sr.shape == (1, 4, 256, 256), f"Expected (1,4,256,256), got {sr.shape}"
    print("SRCNN forward pass OK ✓")
