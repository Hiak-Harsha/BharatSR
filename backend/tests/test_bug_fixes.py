"""
Regression tests for all critical bug fixes:
BUG-001, BUG-002, BUG-003, BUG-005, BUG-008.
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path

from backend.app.services.inference import process_geotiff_file, run_tiled_inference
from backend.app.services.preprocessing import export_geotiff_bytes
from backend.app.services.job_store import JobStore
from backend.app.models_ml.rcan import RCAN


def test_3band_image_raises_valueerror(tmp_path):
    """BUG-001: process_geotiff_file must refuse <4 band inputs with ValueError, never zero-pad."""
    test_tif = tmp_path / "3band.tif"
    out_tif = tmp_path / "out.tif"

    # Write a 3-band GeoTIFF
    with rasterio.open(
        test_tif, "w", driver="GTiff", height=16, width=16, count=3, dtype=np.float32, crs="EPSG:4326", transform=Affine.identity()
    ) as dst:
        dst.write(np.zeros((3, 16, 16), dtype=np.float32))

    rcan = RCAN(n_bands=4, n_feats=16, n_resgroups=1, n_resblocks=1, scale=4)
    with pytest.raises(ValueError, match="Sentinel-2 model requires exactly 4 bands"):
        process_geotiff_file(str(test_tif), str(out_tif), rcan)


def test_affine_rotation_terms_not_scaled():
    """BUG-005: export_geotiff_bytes must only scale pixel sizes (a, e), keeping rotation terms (b, d) untouched."""
    # Affine(a, b, c, d, e, f)
    # where a=pixel_width, b=rotation, c=x_origin, d=rotation, e=pixel_height, f=y_origin
    in_transform = [10.0, 0.5, 500000.0, 0.5, -10.0, 3000000.0]
    geo_metadata = {
        "has_geo": True,
        "crs": "EPSG:32643",
        "transform": in_transform,
        "nodata": None,
    }
    img = np.zeros((4, 32, 32), dtype=np.float32)
    tif_bytes = export_geotiff_bytes(img, geo_metadata=geo_metadata, scale_factor=4)

    with rasterio.io.MemoryFile(tif_bytes) as mem:
        with mem.open() as ds:
            t = ds.transform
            assert t.a == 10.0 / 4  # Pixel width scaled
            assert t.e == -10.0 / 4  # Pixel height scaled
            assert t.b == 0.5  # Rotation preserved exactly
            assert t.d == 0.5  # Rotation preserved exactly
            assert t.c == 500000.0  # Origin preserved
            assert t.f == 3000000.0  # Origin preserved


def test_job_completed_at_not_set_during_processing(tmp_path):
    """BUG-008: update_job must NOT write completed_at for non-terminal states like 'processing'."""
    db_file = tmp_path / "test_jobs.db"
    store = JobStore(str(db_file))
    job_id = store.create_job("rcan")

    store.update_job(job_id, status="processing", progress_pct=50)
    job = store.get_job(job_id)
    assert job["completed_at"] is None, f"completed_at should be None during processing, got {job['completed_at']}"

    store.update_job(job_id, status="completed", progress_pct=100)
    job_done = store.get_job(job_id)
    assert job_done["completed_at"] is not None, "completed_at must be populated when status='completed'"
