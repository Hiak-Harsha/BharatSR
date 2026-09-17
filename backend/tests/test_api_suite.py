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
