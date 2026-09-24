"""
BharatSR — Models Discovery Router
"""

from fastapi import APIRouter, Depends
from backend.app.schemas import ModelsListResponse, ModelInfo
from backend.app.deps import get_model_registry
from backend.app.services.inference import ModelRegistry

router = APIRouter(tags=["models"])


@router.get("/api/models", response_model=ModelsListResponse)
async def list_models(registry: ModelRegistry = Depends(get_model_registry)):
    """List available super-resolution models with parameter count and capabilities."""
    models = registry.list_models()
    return ModelsListResponse(models=[ModelInfo(**m) for m in models])
