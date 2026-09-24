"""
Automated Pytest Suite for BharatSR v2 Endpoints:
- /api/indices
- /api/crop-health
- /api/field-boundary
- /api/change-detect
- /api/batch and /api/batch/{batch_id}
- /api/models/reload
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_spectral_indices_endpoint(client):
    """POST /api/indices returns 7 spectral indices with stats and visualizations."""
    resp = client.post("/api/indices", data={"sample_id": "sample_0", "model_id": "rcan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    indices = data["indices"]
    for idx_name in ["ndvi", "ndwi", "evi", "savi", "rvi", "ndbi_approx", "gci"]:
        assert idx_name in indices
        assert "sr" in indices[idx_name]
        assert "mean" in indices[idx_name]["sr"]
        assert "sr_visualization" in indices[idx_name]
        assert indices[idx_name]["sr_visualization"].startswith("data:image/png;base64,")


def test_crop_health_returns_classification_map(client):
    """POST /api/crop-health returns 6-class classification map, area stats, recommendations."""
    resp = client.post("/api/crop-health", data={"sample_id": "sample_0", "model_id": "rcan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "classification_map" in data
    assert data["classification_map"].startswith("data:image/png;base64,")
    assert "area_statistics" in data
    assert "health_score" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert "disclaimer" in data


def test_field_boundary_returns_overlays(client):
    """POST /api/field-boundary returns edge overlay heatmaps and boundary improvement metrics."""
    resp = client.post("/api/field-boundary", data={"sample_id": "sample_0", "model_id": "rcan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "sr_edge_overlay" in data
    assert data["sr_edge_overlay"].startswith("data:image/png;base64,")
    assert "lr_edge_overlay" in data
    assert "boundary_improvement_ratio" in data
    assert data["boundary_improvement_ratio"] >= 0.0


def test_change_detect_endpoint(client):
    """POST /api/change-detect compares two runs and outputs change maps."""
    # Create two runs first
    r1 = client.post("/api/superresolve", data={"sample_id": "sample_0", "model_id": "rcan"})
    assert r1.status_code == 200
    run_id_1 = r1.json()["run_id"]

    r2 = client.post("/api/superresolve", data={"sample_id": "sample_0", "model_id": "srcnn"})
    assert r2.status_code == 200
    run_id_2 = r2.json()["run_id"]

    resp = client.post(
        "/api/change-detect",
        data={"run_id_t1": run_id_1, "run_id_t2": run_id_2, "method": "ndvi_diff"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "ndvi_difference_map" in data
    assert "spectral_difference_map" in data
    assert "change_magnitude_map" in data
    assert "statistics" in data
    assert "interpretation" in data


def test_batch_submit_and_poll(client):
    """POST /api/batch submits multi-tile jobs and GET /api/batch/{id} returns manifest status."""
    resp = client.post("/api/batch", data={"sample_ids": "sample_0", "model_id": "rcan"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    batch_id = data["batch_id"]
    assert len(data["job_ids"]) == 1

    status_resp = client.get(f"/api/batch/{batch_id}")
    assert status_resp.status_code == 200
    b_data = status_resp.json()
    assert b_data["batch_id"] == batch_id
    assert b_data["total"] == 1
    assert "overall_status" in b_data


def test_model_reload_security_check(client):
    """POST /api/models/reload rejects paths outside weights/ directory."""
    resp = client.post(
        "/api/models/reload",
        data={"model_id": "rcan", "checkpoint_path": "../../evil.pth"},
    )
    assert resp.status_code == 400
    assert "must be within the weights" in resp.json()["detail"]
