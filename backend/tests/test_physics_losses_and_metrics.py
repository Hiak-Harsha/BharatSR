"""
Tests for BharatSR Physics-Constrained Losses & Metrics
Verifies canonical degradation operator, consistency loss, SAM loss stabilization,
and full remote-sensing metric calculation suite.
"""

import pytest
import numpy as np
import torch

from training.losses import (
    degrade_canonical_4x,
    SpectralConsistencyLoss,
    SAMLoss,
    BharatSRCombinedLoss,
    compute_psnr,
    compute_ssim,
    compute_sam,
    compute_spectral_mae,
    compute_downsample_consistency,
    compute_hallucination_and_correctness,
    compute_all_metrics,
)


def test_canonical_degradation_operator_exact_reproduction():
    """
    Verify that a high-resolution image degraded with the canonical 4x4 area average
    reproduces low-resolution within machine precision.
    """
    np.random.seed(42)
    hr_np = np.random.rand(4, 128, 128).astype(np.float32)
    hr_t = torch.from_numpy(hr_np).unsqueeze(0)  # (1, 4, 128, 128)

    lr_t = degrade_canonical_4x(hr_t, scale_factor=4)
    assert lr_t.shape == (1, 4, 32, 32)

    # Calculate expected manual 4x4 block mean
    lr_manual = np.zeros((4, 32, 32), dtype=np.float32)
    for c in range(4):
        for i in range(32):
            for j in range(32):
                block = hr_np[c, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
                lr_manual[c, i, j] = np.mean(block)

    assert np.allclose(lr_t.squeeze(0).numpy(), lr_manual, atol=1e-6), \
        "Canonical degradation operator deviated from exact 4x4 area average!"


def test_consistency_loss_zero_when_perfectly_consistent():
    """Downsample consistency loss L_DC must be identically 0 when SR matches LR under D."""
    hr_np = np.random.rand(4, 64, 64).astype(np.float32)
    hr_t = torch.from_numpy(hr_np).unsqueeze(0)
    lr_t = degrade_canonical_4x(hr_t, scale_factor=4)

    dc_loss_fn = SpectralConsistencyLoss(scale_factor=4)
    loss = dc_loss_fn(hr_t, lr_t)

    assert loss.item() < 1e-6, f"Expected near-zero L_DC, got {loss.item()}"


def test_sam_loss_numerical_stabilization():
    """
    Verify SAM loss handles edge cases without NaN gradients:
    - Identical vectors (cos = 1.0 -> SAM = 0)
    - Near-zero spectral vectors (noise or shadow)
    - Anti-aligned vectors
    """
    sam_fn = SAMLoss()

    # Case 1: Identical vectors
    v1 = torch.ones(2, 4, 16, 16, dtype=torch.float32) * 0.5
    loss_identical = sam_fn(v1, v1)
    assert not torch.isnan(loss_identical)
    assert loss_identical.item() < 0.005

    # Case 2: Near-zero vectors
    v_zero = torch.zeros(2, 4, 16, 16, dtype=torch.float32)
    v_norm = torch.rand(2, 4, 16, 16, dtype=torch.float32) * 0.5
    loss_zero = sam_fn(v_zero, v_norm)
    assert not torch.isnan(loss_zero)

    # Case 3: Anti-aligned vectors
    loss_anti = sam_fn(v1, -v1)
    assert not torch.isnan(loss_anti)
    assert loss_anti.item() > 2.5  # ~pi radians


def test_combined_physics_loss_logging():
    """Verify BharatSRCombinedLoss computes and logs all 4 loss components."""
    criterion = BharatSRCombinedLoss(
        scale_factor=4,
        lambda_rec=1.0,
        lambda_sam=0.1,
        lambda_dc=0.1,
        lambda_unc=0.2,
    )

    sr = torch.rand(2, 4, 64, 64, requires_grad=True)
    logvar = torch.zeros(2, 1, 64, 64, requires_grad=True)
    hr = torch.rand(2, 4, 64, 64)
    lr = degrade_canonical_4x(hr, scale_factor=4)

    loss, loss_dict = criterion(sr, logvar, hr, lr)

    assert "total" in loss_dict
    assert "l1" in loss_dict
    assert "sam" in loss_dict
    assert "dc" in loss_dict
    assert "unc" in loss_dict

    loss.backward()
    assert sr.grad is not None
    assert not torch.isnan(sr.grad).any()


def test_metric_computations_bounds():
    """Verify metrics conform to physical remote-sensing definitions and bounds."""
    np.random.seed(42)
    hr = np.random.rand(4, 64, 64).astype(np.float32) * 0.8
    lr = degrade_canonical_4x(torch.from_numpy(hr).unsqueeze(0), scale_factor=4).squeeze(0).numpy()

    # Small perturbation
    sr = np.clip(hr + np.random.randn(4, 64, 64).astype(np.float32) * 0.01, 0.0, None)

    metrics = compute_all_metrics(sr, hr, lr_original=lr, scale_factor=4)

    assert metrics["psnr_db"] > 25.0
    assert 0.0 <= metrics["ssim"] <= 1.0
    assert 0.0 <= metrics["sam_degrees"] <= 180.0
    assert metrics["downsample_consistency_mae"] >= 0.0
    assert metrics["spectral_mae"] >= 0.0
    assert 0.0 <= metrics["correctness_score"] <= 1.0
    assert 0.0 <= metrics["high_freq_hallucination_rate"] <= 1.0
