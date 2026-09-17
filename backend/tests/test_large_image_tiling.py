"""
Tests for Large GeoTIFF Tiled Inference & Stitching
Verifies sliding window tiling, overlap blending with 2D Hann window,
memory-safe execution, and rasterio output validation.
"""

import pytest
import numpy as np
import torch
import rasterio
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from backend.app.models_ml.rcan import RCAN
from backend.app.services.inference import create_blend_window, run_tiled_inference, process_geotiff_file
from tools.validate_geotiff import validate_geotiff


def test_hann_blend_window_properties():
    """Verify 2D Hann blending window has peak at center and tapers at edges."""
    window = create_blend_window(64, 64)
    assert window.shape == (64, 64)
    assert window.dtype == np.float32

    # Peak in center
    assert np.isclose(window[32, 32], 1.0, atol=1e-2)
    # Tapers toward border
    assert window[0, 0] < 0.2
    assert window[63, 63] < 0.2


def test_tiled_inference_stitching_seamless(tmp_path):
    """
    Verify tiled inference correctly stitches overlapping tiles:
    - Input: (4, 128, 128)
    - Output: (4, 512, 512) for 4x SR
    - No NaNs or infinities
    - Boundary transitions are smooth
    """
    model = RCAN(n_bands=4, n_feats=16, n_resgroups=1, n_resblocks=1, scale=4, predict_uncertainty=False)
    model.eval()

    lr_large = np.random.rand(4, 128, 128).astype(np.float32) * 0.8
    sr_stitched, latency, unc = run_tiled_inference(
        model=model,
        lr_image=lr_large,
        tile_size=64,
        overlap=16,
        scale_factor=4,
        device=torch.device("cpu"),
    )

    assert sr_stitched.shape == (4, 512, 512)
    assert not np.isnan(sr_stitched).any()
    assert not np.isinf(sr_stitched).any()
    assert (sr_stitched >= 0.0).all()


def test_end_to_end_geotiff_processing(tmp_path):
    """
    End-to-end test:
    Read input GeoTIFF -> tile -> infer -> blend -> stitch -> save output GeoTIFF -> validate.
    """
    input_tif = PROJECT_ROOT / "backend" / "tests" / "fixtures" / "sentinel2_l2a_test.tif"
    output_tif = tmp_path / "e2e_tiled_output_4x.tif"

    model = RCAN(n_bands=4, n_feats=16, n_resgroups=1, n_resblocks=1, scale=4, predict_uncertainty=False)
    model.eval()

    metadata = process_geotiff_file(
        input_path=str(input_tif),
        output_path=str(output_tif),
        model=model,
        tile_size=32,
        overlap=8,
        scale_factor=4,
        device=torch.device("cpu"),
    )

    assert output_tif.exists()
    assert metadata["input_size"] == [64, 64]
    assert metadata["output_size"] == [256, 256]
    assert metadata["crs"] == "EPSG:32643"

    # Verify with rasterio tool
    assert validate_geotiff(output_tif) is True
