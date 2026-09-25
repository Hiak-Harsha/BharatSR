"""
Unit & Integration tests for Dataset & Training Transparency router (Part E).
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_dataset_summary_endpoint():
    """Verify /api/dataset/summary returns required structure and metrics."""
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_scenes" in data
    assert data["total_scenes"] > 0
    assert "splits" in data
    assert "train" in data["splits"]
    assert "val" in data["splits"]
    assert "test" in data["splits"]
    assert "synthetic_count" in data
    assert "real_count" in data
    assert "regions" in data
    assert isinstance(data["regions"], list)
    assert len(data["regions"]) > 0
    assert "sensors" in data
    assert "date_range" in data
    assert "mean_cloud_fraction" in data
    assert "mean_registration_rmse" in data


def test_dataset_scenes_endpoint():
    """Verify /api/dataset/scenes pagination and filtering."""
    response = client.get("/api/dataset/scenes?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "scenes" in data
    assert len(data["scenes"]) <= 10
    if data["scenes"]:
        first = data["scenes"][0]
        assert "scene_id" in first
        assert "region" in first
        assert "split" in first
        assert "sensor" in first


def test_qa_report_endpoint():
    """Verify /api/dataset/qa-report/{split} serves valid image."""
    for split in ["train", "val", "test"]:
        response = client.get(f"/api/dataset/qa-report/{split}")
        assert response.status_code == 200
        assert "image/png" in response.headers.get("content-type", "")

    # Invalid split should 400
    bad_resp = client.get("/api/dataset/qa-report/invalid_split")
    assert bad_resp.status_code == 400


def test_preprocessing_sample_endpoint():
    """Verify /api/dataset/preprocessing-sample returns 4 stage walkthrough."""
    response = client.get("/api/dataset/preprocessing-sample")
    assert response.status_code == 200
    data = response.json()
    assert "scene_id" in data
    assert "stages" in data
    assert len(data["stages"]) == 4
    for stage in data["stages"]:
        assert "title" in stage
        assert "description" in stage
        assert stage["image"].startswith("data:image/png;base64,")
    assert "training_pair" in data
    assert "lr" in data["training_pair"]
    assert "hr" in data["training_pair"]


def test_training_history_endpoint():
    """Verify /api/training/history/{model_name} returns history or final metrics."""
    for model in ["srcnn", "rcan", "hat", "swinir"]:
        response = client.get(f"/api/training/history/{model}")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == model
        assert "is_final_only" in data
        assert "epochs" in data
        assert "final_metrics" in data


def test_model_card_endpoint():
    """Verify /api/training/model-card/{model_name} returns architecture specifications."""
    for model in ["rcan", "srcnn", "hat", "swinir"]:
        response = client.get(f"/api/training/model-card/{model}")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == model
        assert "architecture" in data
        assert "parameters_count" in data
        assert data["parameters_count"] > 0
        assert "key_design" in data
        assert "training_config" in data
        assert "final_metrics" in data
        assert "recommended_use" in data
