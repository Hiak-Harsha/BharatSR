"""
Smoke tests for new architectures: SwinIR_SR, HAT_SR, DiffusionSR, EnsembleSR.
"""

import time
import torch
import numpy as np
from backend.app.models_ml.swinir_sr import SwinIR_SR
from backend.app.models_ml.hat_sr import HAT_SR
from backend.app.models_ml.diffusion_sr import DiffusionSR
from backend.app.models_ml.ensemble_sr import EnsembleSR
from backend.app.models_ml.rcan import RCAN


def test_swinir_output_shape():
    model = SwinIR_SR(
        n_bands=4,
        embed_dim=24,
        depths=[2, 2],
        num_heads=[4, 4],
        window_size=4,
        scale=4,
        predict_uncertainty=True,
    )
    x = torch.randn(1, 4, 32, 32)
    sr, log_var = model(x)
    assert sr.shape == (1, 4, 128, 128), f"Expected (1, 4, 128, 128), got {sr.shape}"
    assert log_var.shape == (1, 1, 128, 128), f"Expected (1, 1, 128, 128), got {log_var.shape}"


def test_hat_output_shape():
    model = HAT_SR(
        n_bands=4,
        embed_dim=24,
        depths=[2, 2],
        num_heads=[4, 4],
        window_size=4,
        scale=4,
        predict_uncertainty=True,
    )
    x = torch.randn(1, 4, 32, 32)
    sr, log_var = model(x)
    assert sr.shape == (1, 4, 128, 128), f"Expected (1, 4, 128, 128), got {sr.shape}"


def test_diffusion_sr_inference_speed():
    model = DiffusionSR(n_bands=4, scale=4, num_steps=4)
    x = torch.randn(1, 4, 32, 32)
    t0 = time.time()
    sr, unc = model(x)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Diffusion inference too slow: {elapsed}s"
    assert sr.shape == (1, 4, 128, 128)


def test_ensemble_weighted_average():
    rcan = RCAN(n_bands=4, n_feats=16, n_resgroups=1, n_resblocks=1, scale=4, predict_uncertainty=True)
    swin = SwinIR_SR(n_bands=4, embed_dim=24, depths=[2], num_heads=[4], scale=4, predict_uncertainty=True)

    ensemble = EnsembleSR([rcan, swin], weights=[0.6, 0.4])
    x = torch.randn(1, 4, 32, 32)
    sr, log_var = ensemble(x)
    assert sr.shape == (1, 4, 128, 128)
    assert log_var.shape == (1, 1, 128, 128)
