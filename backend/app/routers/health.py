"""
BharatSR — Health Check Router
"""

from fastapi import APIRouter, Depends
from backend.app.schemas import HealthResponse
from backend.app.config import Settings
from backend.app.deps import get_settings, get_model_registry, get_job_store
from backend.app.services.inference import ModelRegistry
from backend.app.services.job_store import JobStore

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
    store: JobStore = Depends(get_job_store),
):
    """Health check endpoint confirming API status, loaded models, and device."""
    active_count = 0
    if store:
        try:
            recent = store.list_jobs(limit=10)
            active_count = sum(1 for j in recent if j.get("status") in ("pending", "processing"))
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        models_loaded=len(registry.list_models()),
        device=str(registry.device),
        version=settings.version,
        active_jobs=active_count,
    )
