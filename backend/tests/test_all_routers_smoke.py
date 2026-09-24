"""
BharatSR — All Routers Smoke Test
Validates that every single modular router has a working, well-formed happy path.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI application with its full lifespan context."""
    with TestClient(app) as c:
        yield c


def test_health_router(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_models_router(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 2
    model_ids = [m["id"] for m in data["models"]]
    assert "rcan" in model_ids and "srcnn" in model_ids


def test_samples_router(client):
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

    preview_resp = client.get("/api/samples/sample_0/preview?crop=16")
    assert preview_resp.status_code == 200
    assert "views" in preview_resp.json()


def test_inference_router(client):
    # Single SR
    sr_resp = client.post("/api/superresolve", data={"sample_id": "sample_0", "model_id": "bicubic"})
    assert sr_resp.status_code == 200
    sr_data = sr_resp.json()
    assert sr_data["status"] == "success"
    run_id = sr_data["run_id"]

    # Pixel profile with run_id
    prof_resp = client.post("/api/pixel-profile", data={"run_id": run_id, "x": 10, "y": 10})
    assert prof_resp.status_code == 200
    assert len(prof_resp.json()["bands_data"]) == 4

    # Downstream masks with run_id
    mask_resp = client.post("/api/downstream-masks", data={"run_id": run_id, "model_id": "bicubic"})
    assert mask_resp.status_code == 200
    assert "tasks" in mask_resp.json()


def test_jobs_router(client):
    # Submit async
    async_resp = client.post("/api/superresolve/async", data={"sample_id": "sample_0", "model_id": "bicubic"})
    assert async_resp.status_code == 200
    job_id = async_resp.json()["job_id"]

    # Query status
    status_resp = client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id

    # List jobs
    list_resp = client.get("/api/jobs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()["jobs"]) > 0


def test_batch_router(client):
    batch_resp = client.post("/api/batch", data={"sample_ids": "sample_0,sample_1", "model_id": "bicubic"})
    assert batch_resp.status_code == 200
    batch_id = batch_resp.json()["batch_id"]

    status_resp = client.get(f"/api/batch/{batch_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["batch_id"] == batch_id


def test_export_router(client):
    report_resp = client.get("/api/export/report?sample_id=sample_0&model_id=bicubic")
    assert report_resp.status_code == 200
    assert "metrics" in report_resp.json()


def test_analysis_router(client):
    # Indices
    idx_resp = client.post("/api/indices", data={"sample_id": "sample_0", "model_id": "bicubic"})
    assert idx_resp.status_code == 200
    assert "ndvi" in idx_resp.json()["indices"]

    # Crop health
    crop_resp = client.post("/api/crop-health", data={"sample_id": "sample_0", "model_id": "bicubic"})
    assert crop_resp.status_code == 200
    assert "health_score" in crop_resp.json()

    # Field boundary
    boundary_resp = client.post("/api/field-boundary", data={"sample_id": "sample_0", "model_id": "bicubic"})
    assert boundary_resp.status_code == 200
    assert "sr_edge_density" in boundary_resp.json()


def test_admin_router_security(client):
    # Security: paths outside weights/ directory are rejected
    bad_reload = client.post("/api/models/reload", data={"model_id": "rcan", "checkpoint_path": "../../etc/passwd"})
    assert bad_reload.status_code == 400
