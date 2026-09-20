"""
Automated Pytest Suite for BharatSR FastAPI Endpoints
Covers all 11 endpoints with FastAPI TestClient and lifespan support.
"""

import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import rasterio
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI application with its full lifespan context."""
    with TestClient(app) as c:
        yield c


def test_endpoint_health(client):
    """GET /api/health"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "device" in data
    assert data["version"] == "0.2.0"


def test_endpoint_models(client):
    """GET /api/models"""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) > 0
    model_ids = [m["id"] for m in data["models"]]
    assert "rcan" in model_ids or "srcnn" in model_ids


def test_endpoint_samples(client):
    """GET /api/samples - verifies honest provenance metadata (no fabricated city labels for synthetic)"""
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

    for s in data["samples"]:
        assert "id" in s
        assert "title" in s
        assert "has_geo" in s
        assert "views" in s
        assert "rgb" in s["views"]
        if not s["has_geo"]:
            assert "Synthetic" in s["title"] or "Synthetic" in s["region"] or "No geospatial reference" in s["coordinates"]


def test_endpoint_superresolve(client):
    """POST /api/superresolve"""
    resp = client.post("/api/superresolve", data={"sample_id": "sample_0", "model_id": "rcan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "output" in data
    assert data["output"]["shape"] == [4, 256, 256]
    assert "metrics" in data
    assert "psnr" in data["metrics"]


def test_endpoint_compare(client):
    """POST /api/compare"""
    resp = client.post("/api/compare", data={"sample_id": "sample_0"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "comparison_table" in data
    assert len(data["comparison_table"]) >= 4


def test_endpoint_pixel_profile(client):
    """POST /api/pixel-profile"""
    resp = client.post("/api/pixel-profile", data={"sample_id": "sample_0", "x": 64, "y": 64, "model_id": "rcan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "bands_data" in data
    assert len(data["bands_data"]) == 4
    assert "ndvi" in data
    assert "interpretation_disclaimer" in data


def test_endpoint_async_job_lifecycle(client):
    """POST /api/superresolve/async and GET /api/jobs/{id}"""
    submit_resp = client.post("/api/superresolve/async", data={"sample_id": "sample_real_s2", "model_id": "rcan"})
    assert submit_resp.status_code == 200
    job_info = submit_resp.json()
    assert job_info["status"] == "accepted"
    job_id = job_info["job_id"]

    # Check status
    status_resp = client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    s_data = status_resp.json()
    assert s_data["job_id"] == job_id
    assert s_data["status"] in ("pending", "processing", "completed")


def test_endpoint_jobs_list(client):
    """GET /api/jobs"""
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_endpoint_job_cancellation(client):
    """POST /api/jobs/{job_id}/cancel"""
    # Create a job first
    submit_resp = client.post("/api/superresolve/async", data={"sample_id": "sample_0", "model_id": "rcan"})
    job_id = submit_resp.json()["job_id"]

    # Cancel it
    cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_resp.status_code in (200, 400)


def test_endpoint_export_geotiff_rejection_for_synthetic(client):
    """GET /api/export/geotiff on unreferenced synthetic sample must return 400"""
    resp = client.get("/api/export/geotiff?sample_id=sample_0&model_id=rcan")
    assert resp.status_code == 400
    assert "No geospatial reference available" in resp.json()["detail"]


def test_endpoint_export_geotiff_success_for_real_sentinel2(client):
    """GET /api/export/geotiff on genuine Sentinel-2 sample must return valid Float32 GeoTIFF"""
    resp = client.get("/api/export/geotiff?sample_id=sample_real_s2&model_id=rcan")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/tiff"

    with rasterio.open(io.BytesIO(resp.content)) as ds:
        assert str(ds.crs) == "EPSG:32643"
        assert ds.count == 4
        assert ds.dtypes[0] == "float32"
        assert abs(ds.res[0] - 2.5) < 1e-4
        assert abs(ds.res[1] - 2.5) < 1e-4


def test_endpoint_export_report(client):
    """GET /api/export/report"""
    resp = client.get("/api/export/report?sample_id=sample_real_s2&model_id=rcan")
    assert resp.status_code == 200
    data = resp.json()
    assert "title" in data
    assert "problem_statement" in data
    assert data["problem_statement"] == "SIH26142 - Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery"
    assert "metrics" in data
    assert "spectral_integrity_compliance" in data


def test_endpoint_downstream_masks(client):
    """POST /api/downstream-masks returns binary mask overlays and segmentation metrics"""
    resp = client.post(
        "/api/downstream-masks",
        data={"sample_id": "sample_real_s2", "model_id": "rcan"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "canopy_segmentation" in data["tasks"]
    assert "built_up_infrastructure" in data["tasks"]
    canopy = data["tasks"]["canopy_segmentation"]
    assert "rcan" in canopy
    assert "bicubic" in canopy
    assert "masks" in canopy
    assert "rcan" in canopy["masks"]
    assert canopy["masks"]["rcan"].startswith("data:image/png;base64,")


def test_endpoint_run_caching_and_run_id_geotiff(client):
    """POST /api/superresolve returns a run_id which can be used to export GeoTIFF and inspect pixels"""
    sr_resp = client.post(
        "/api/superresolve",
        data={"sample_id": "sample_real_s2", "model_id": "rcan"}
    )
    assert sr_resp.status_code == 200
    sr_data = sr_resp.json()
    assert "run_id" in sr_data
    run_id = sr_data["run_id"]

    # Download GeoTIFF by run_id
    tif_resp = client.get(f"/api/export/geotiff?run_id={run_id}")
    assert tif_resp.status_code == 200
    assert tif_resp.headers["content-type"] == "image/tiff"

    # Inspect pixel by run_id
    px_resp = client.post(
        "/api/pixel-profile",
        data={"run_id": run_id, "x": 100, "y": 100, "model_id": "rcan"}
    )
    assert px_resp.status_code == 200
    px_data = px_resp.json()
    assert px_data["run_id"] == run_id
    assert len(px_data["bands_data"]) == 4


def test_upload_sample_real_s2_infer_export_reopen(client):
    """
    End-to-end integration test:
    1. Upload authentic Sentinel-2 GeoTIFF (sample_real_s2.tif)
    2. Infer 4x super-resolution with RCAN
    3. Export GeoTIFF by run_id
    4. Reopen exported GeoTIFF with rasterio and verify CRS EPSG:32643, 2.5m resolution, 4 bands
    """
    tif_path = PROJECT_ROOT / "backend" / "sample_tiles" / "sample_real_s2.tif"
    assert tif_path.exists(), "sample_real_s2.tif not found"

    with open(tif_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("sample_real_s2.tif", file_bytes, "image/tiff")}
    data = {"model_id": "rcan"}

    # 1. POST /api/superresolve via file upload
    resp = client.post("/api/superresolve", files=files, data=data)
    assert resp.status_code == 200, f"Upload inference failed: {resp.text}"
    resp_data = resp.json()
    assert resp_data["status"] == "success"
    assert "run_id" in resp_data
    run_id = resp_data["run_id"]
    assert resp_data["geospatial_metadata"]["has_geo"] is True
    assert resp_data["geospatial_metadata"]["crs"] == "EPSG:32643"

    # 2. GET /api/export/geotiff with run_id
    export_resp = client.get(f"/api/export/geotiff?run_id={run_id}")
    assert export_resp.status_code == 200, f"GeoTIFF export failed: {export_resp.text}"
    assert export_resp.headers["content-type"] == "image/tiff"

    # 3. Reopen exported bytes with rasterio
    with rasterio.open(io.BytesIO(export_resp.content)) as ds:
        assert str(ds.crs) == "EPSG:32643", f"Wrong CRS: {ds.crs}"
        assert ds.count == 4, f"Wrong band count: {ds.count}"
        assert ds.dtypes[0] == "float32", f"Wrong dtype: {ds.dtypes[0]}"
        assert ds.shape == (256, 256), f"Wrong shape: {ds.shape}"
        assert abs(ds.res[0] - 2.5) < 1e-4, f"Wrong X pixel size: {ds.res[0]}"
        assert abs(ds.res[1] - 2.5) < 1e-4, f"Wrong Y pixel size: {ds.res[1]}"

        # Read bands and verify valid physical reflectance
        data_arr = ds.read()
        assert not np.isnan(data_arr).any(), "Found NaNs in exported GeoTIFF"
        assert not np.isinf(data_arr).any(), "Found Infs in exported GeoTIFF"
        assert data_arr.min() >= 0.0, "Found negative reflectance values"
        assert data_arr.max() > 0.0, "Exported GeoTIFF is completely empty"

