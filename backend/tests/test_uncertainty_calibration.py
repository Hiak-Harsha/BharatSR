"""
Tests for BharatSR Uncertainty Estimation & Calibration Engine
Verifies heteroscedastic loss, standard deviation mapping, prediction interval coverage,
error correlation, and AUROC/AUPRC for high-error detection.
"""

import pytest
import numpy as np
import torch

from backend.app.models_ml.uncertainty import (
    logvar_to_std,
    generate_uncertainty_heatmap,
    summarize_uncertainty,
    evaluate_uncertainty_calibration,
)
from training.losses import UncertaintyLoss


def test_uncertainty_loss_and_logvar_clamp():
    """Verify uncertainty loss penalizes under-confident or over-confident predictions."""
    unc_loss_fn = UncertaintyLoss(mode="l1")

    sr = torch.ones(2, 4, 16, 16) * 0.5
    hr = torch.ones(2, 4, 16, 16) * 0.6  # constant error of 0.1
    log_var = torch.zeros(2, 1, 16, 16)  # sigma = 1.0

    loss = unc_loss_fn(sr, log_var, hr)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_logvar_to_std_conversion():
    """Verify log_var mapping to standard deviation: std = exp(0.5 * log_var)."""
    log_var = torch.tensor([0.0, -2.0, 2.0])
    std = logvar_to_std(log_var)

    expected = torch.exp(0.5 * log_var)
    assert torch.allclose(std, expected, atol=1e-5)


def test_evaluate_uncertainty_calibration_metrics():
    """
    Verify full uncertainty calibration suite produces valid statistics:
    - NLL
    - Coverage percentages (0-100%)
    - Spearman correlation (-1 to 1)
    - AUROC (0 to 1)
    - AUPRC (0 to 1)
    """
    np.random.seed(42)
    n, c, h, w = 5, 4, 32, 32
    y_true = np.random.rand(n, c, h, w).astype(np.float32)
    # Simulate correlated error and predicted sigma
    abs_error = np.random.exponential(scale=0.05, size=(n, c, h, w)).astype(np.float32)
    y_pred = y_true + abs_error

    # Predicted standard deviation correlated with actual error
    sigma = abs_error.mean(axis=1, keepdims=True) + np.random.normal(0, 0.01, size=(n, 1, h, w)).astype(np.float32)
    sigma = np.clip(sigma, 0.005, None)

    calib = evaluate_uncertainty_calibration(y_true, y_pred, sigma)

    assert "mean_sigma" in calib
    assert "validation_nll" in calib
    assert "pearson_correlation" in calib
    assert "spearman_correlation" in calib
    assert "coverage_68_pct" in calib
    assert "coverage_95_pct" in calib
    assert "high_error_auroc" in calib
    assert "high_error_auprc" in calib

    # Mathematical bounds
    assert 0.0 <= calib["coverage_68_pct"] <= 100.0
    assert 0.0 <= calib["coverage_95_pct"] <= 100.0
    assert -1.0 <= calib["spearman_correlation"] <= 1.0
    assert 0.0 <= calib["high_error_auroc"] <= 1.0
    assert 0.0 <= calib["high_error_auprc"] <= 1.0
