"""
BharatSR — Backend API Integration Tests
Tests /api/health, /api/models, /api/samples, and /api/superresolve via FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from backend.app.main import app


def test_api_integration():
    with TestClient(app) as client:
        # 1. Health
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        health = r_health.json()
        assert health["status"] == "ok"

        # 2. Models
        r_models = client.get("/api/models")
        assert r_models.status_code == 200
        models = r_models.json()
        assert len(models["models"]) > 0

        # 3. Samples
        r_samples = client.get("/api/samples")
        assert r_samples.status_code == 200
        samples = r_samples.json()["samples"]
        assert len(samples) > 0
        sample_id = samples[0]["id"]

        # 4. Superresolve
        r_sr = client.post("/api/superresolve", data={"sample_id": sample_id, "model_id": "rcan"})
        assert r_sr.status_code == 200
        sr_data = r_sr.json()
        assert sr_data["status"] == "success"
        assert "metrics" in sr_data


if __name__ == "__main__":
    test_api_integration()
    print("API Integration test passed.")
