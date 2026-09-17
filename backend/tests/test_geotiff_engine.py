"""
Tests for BharatSR Geospatial & GeoTIFF Engine
Verifies CRS preservation, 4x pixel size scaling (p_out = p_in / 4), affine transform calculation,
rasterio compliance, and rejection of non-georeferenced inputs without silent fabrication.
"""

import io
import pytest
import numpy as np
import rasterio
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from backend.app.services.preprocessing import load_image_from_bytes, export_geotiff_bytes
from tools.validate_geotiff import validate_geotiff


def test_real_sentinel2_geotiff_fixture_metadata():
    """Verify genuine Sentinel-2 GeoTIFF fixture preserves all spatial parameters."""
    tif_path = PROJECT_ROOT / "backend" / "tests" / "fixtures" / "sentinel2_l2a_test.tif"
    assert tif_path.exists(), "Test GeoTIFF fixture not found"

    with open(tif_path, "rb") as f:
        file_bytes = f.read()

    img, geo_meta = load_image_from_bytes(file_bytes)

    assert geo_meta["has_geo"] is True
    assert geo_meta["crs"] == "EPSG:32643"
    assert geo_meta["width"] == 64
    assert geo_meta["height"] == 64
    assert geo_meta["pixel_size"] == [10.0, 10.0]
    assert img.shape == (4, 64, 64)
    assert img.dtype == np.float32


def test_4x_sr_affine_transform_scaling():
    """
    For 4x SR:
    input pixel size = p -> output pixel size = p / 4.
    The origin (c, f) must be preserved, while a (dx) and e (dy) are divided by 4.
    """
    tif_path = PROJECT_ROOT / "backend" / "tests" / "fixtures" / "sentinel2_l2a_test.tif"
    with open(tif_path, "rb") as f:
        file_bytes = f.read()
    img, geo_meta = load_image_from_bytes(file_bytes)

    # Simulate 4x SR output: 64x64 -> 256x256
    c, h, w = img.shape
    sr_mock = np.repeat(np.repeat(img, 4, axis=1), 4, axis=2)
    assert sr_mock.shape == (4, 256, 256)

    geotiff_bytes = export_geotiff_bytes(sr_mock, geo_metadata=geo_meta, scale_factor=4)

    # Open with rasterio and verify every geospatial attribute
    with rasterio.open(io.BytesIO(geotiff_bytes)) as ds:
        assert str(ds.crs) == "EPSG:32643", f"Wrong CRS: {ds.crs}"
        assert ds.width == 256
        assert ds.height == 256
        assert ds.count == 4
        assert ds.dtypes[0] == "float32"

        # Pixel size must be exactly 2.5m (10m / 4)
        assert abs(ds.res[0] - 2.5) < 1e-5
        assert abs(ds.res[1] - 2.5) < 1e-5

        # Origin coordinates must match input exactly
        in_transform = geo_meta["transform"]
        assert abs(ds.transform.c - in_transform[2]) < 1e-5
        assert abs(ds.transform.f - in_transform[5]) < 1e-5

        # Tags and descriptions
        tags = ds.tags()
        assert "2.50m-equivalent output grid" in tags.get("gsd", "")
        assert "Sentinel-2 MSI" in tags.get("sensor", "")
        assert ds.descriptions[0] == "B2 - Blue (490nm)"
        assert ds.descriptions[3] == "B8 - NIR (842nm)"


def test_reject_unreferenced_geotiff_export():
    """
    CRITICAL REQUIREMENT:
    If input has no geospatial metadata, export_geotiff_bytes must raise ValueError.
    It must NEVER silently assign EPSG:4326 or Delhi coordinates.
    """
    fake_sr = np.random.rand(4, 128, 128).astype(np.float32)

    # None geo_metadata
    with pytest.raises(ValueError, match="No geospatial reference available"):
        export_geotiff_bytes(fake_sr, geo_metadata=None, scale_factor=4)

    # Non-georeferenced synthetic dictionary
    unreferenced_meta = {
        "has_geo": False,
        "message": "No geospatial reference available",
        "crs": None,
        "transform": None,
    }
    with pytest.raises(ValueError, match="No geospatial reference available"):
        export_geotiff_bytes(fake_sr, geo_metadata=unreferenced_meta, scale_factor=4)


def test_validate_geotiff_tool_passes_valid_file(tmp_path):
    """Verify tools/validate_geotiff.py passes a valid 4-band Float32 SR GeoTIFF."""
    tif_path = PROJECT_ROOT / "backend" / "tests" / "fixtures" / "sentinel2_l2a_test.tif"
    with open(tif_path, "rb") as f:
        file_bytes = f.read()
    img, geo_meta = load_image_from_bytes(file_bytes)
    sr_mock = np.repeat(np.repeat(img, 4, axis=1), 4, axis=2)

    geotiff_bytes = export_geotiff_bytes(sr_mock, geo_metadata=geo_meta, scale_factor=4)
    out_file = tmp_path / "valid_sr_output.tif"
    with open(out_file, "wb") as f:
        f.write(geotiff_bytes)

    assert validate_geotiff(out_file) is True
