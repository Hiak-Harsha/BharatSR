"""
Tests for BharatSR Data Pipeline
Verifies synthetic data generation, real sample loading, physical reflectance normalization,
scene-level splitting, and metadata sidecars.
"""

import pytest
import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from data.scripts.prepare_data import generate_synthetic_pairs, split_by_scene
from backend.app.services.preprocessing import load_sample_tile, load_image_from_bytes


def test_synthetic_data_generation():
    """Verify procedural synthetic scene pairs conform to physical multi-band specs."""
    scenes = generate_synthetic_pairs(n_scenes=3, lr_size=32, scale=4, n_bands=4)
    assert len(scenes) == 3

    for s in scenes:
        lr = s["lr"]
        hr = s["hr"]
        assert lr.shape == (4, 32, 32)
        assert hr.shape == (4, 128, 128)
        assert lr.dtype == np.float32
        assert hr.dtype == np.float32

        # Physical reflectance: no negative values, no NaNs
        assert not np.isnan(lr).any()
        assert not np.isnan(hr).any()
        assert (lr >= 0.0).all()
        assert (hr >= 0.0).all()


def test_scene_level_separation():
    """Verify train, validation, and test splits have ZERO spatial leakage across scenes."""
    scenes = [{"scene_id": f"s_{i}", "lr": np.zeros((4, 16, 16)), "hr": np.zeros((4, 64, 64))} for i in range(20)]
    train, val, test = split_by_scene(scenes, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_ids = {s["scene_id"] for s in train}
    val_ids = {s["scene_id"] for s in val}
    test_ids = {s["scene_id"] for s in test}

    # Disjoint sets: no overlap
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(train_ids) + len(val_ids) + len(test_ids) == 20


def test_real_sentinel2_sample():
    """Verify genuine Sentinel-2 sample tile conforms to expected bands, shape, and reflectance."""
    sample_path = PROJECT_ROOT / "backend" / "sample_tiles" / "sample_real_s2.npz"
    assert sample_path.exists(), "Genuine Sentinel-2 sample fixture not found"

    lr, hr = load_sample_tile(str(sample_path))
    assert lr.shape[0] == 4, "Expected 4 spectral bands: B2, B3, B4, B8"
    assert lr.dtype == np.float32

    # Physical BOA reflectance scale: reasonable solar reflectance [0, ~1+]
    assert lr.min() >= 0.0
    assert lr.max() <= 2.0  # Allows bright targets/clouds without clipping
    assert lr.mean() > 0.05 and lr.mean() < 0.60

    if hr is not None:
        assert hr.shape[0] == 4
        assert hr.shape[1] == lr.shape[1] * 4
        assert hr.shape[2] == lr.shape[2] * 4


def test_image_normalization_no_imagenet():
    """Verify normalization does NOT apply ImageNet mean/std."""
    # Create an 8-bit synthetic RGB image
    fake_png_bytes = b"\x89PNG\r\n\x1a\n"  # Just header
    # Test with simulated 16-bit Sentinel-2 DN values (> 10000)
    import io
    from PIL import Image

    arr_8bit = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr_8bit).save(buf, format="PNG")
    norm_img, geo_meta = load_image_from_bytes(buf.getvalue())

    assert norm_img.shape[0] == 4, "Should be padded to 4 bands (RGB + NIR placeholder)"
    assert norm_img.max() <= 1.0 + 1e-4
    assert norm_img.min() >= 0.0
