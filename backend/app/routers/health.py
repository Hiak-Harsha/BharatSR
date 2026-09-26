"""
BharatSR — Health Check Router
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from backend.app.schemas import HealthResponse
from backend.app.config import Settings
from backend.app.deps import get_settings, get_model_registry, get_job_store
from backend.app.services.inference import ModelRegistry
from backend.app.services.job_store import JobStore
from backend.app.routers.inference import ALLOWED_MODELS
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.health")
router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
    store: JobStore = Depends(get_job_store),
):
    """Health check endpoint confirming API status, per-model status, degraded state, and device."""
    active_count = 0
    if store:
        try:
            recent = store.list_jobs(limit=10)
            active_count = sum(1 for j in recent if j.get("status") in ("pending", "processing"))
        except Exception as e:
            logger.warning(f"Error querying active jobs in health check: {e}")

    loaded_list = registry.list_models()
    loaded_ids = {m["id"]: m for m in loaded_list}
    models_count = len(loaded_list)
    degraded = (models_count == 0)

    # Per-model status reporting
    models_status: List[Dict[str, Any]] = []
    for model_id in ALLOWED_MODELS:
        if model_id == "bicubic":
            models_status.append({
                "id": "bicubic",
                "name": "Bicubic Baseline",
                "status": "ready (algorithmic)",
                "has_uncertainty": False,
            })
        elif model_id in loaded_ids:
            meta = loaded_ids[model_id]
            models_status.append({
                "id": model_id,
                "name": meta.get("name", model_id),
                "status": "loaded",
                "has_uncertainty": meta.get("has_uncertainty", False),
            })
        else:
            models_status.append({
                "id": model_id,
                "status": "unavailable: checkpoint missing",
                "has_uncertainty": False,
            })

    return HealthResponse(
        status="degraded" if degraded else "ok",
        models_loaded=models_count,
        device=str(registry.device),
        version=settings.version,
        active_jobs=active_count,
        degraded=degraded,
        models=models_status,
    )

