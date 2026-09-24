"""
BharatSR — Admin Router
Provides administrative actions like model weight hot-reloading.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.app.config import Settings
from backend.app.deps import get_settings, get_model_registry, verify_api_key
from backend.app.schemas import ModelReloadResponse
from backend.app.services.inference import ModelRegistry
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.admin")
router = APIRouter(tags=["admin"])


@router.post(
    "/api/models/reload",
    response_model=ModelReloadResponse,
    dependencies=[Depends(verify_api_key)],
)
def reload_model(
    model_id: str = Form(...),
    checkpoint_path: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Hot-reload a model from an updated checkpoint without server restart.
    Security: checkpoint_path must be strictly within the configured weights directory.
    """
    weights_dir = Path(settings.weights_dir).resolve()

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).resolve()
        try:
            ckpt_path.relative_to(weights_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="checkpoint_path must be within the weights/ directory")
    else:
        ckpt_path = weights_dir / f"{model_id}_best.pth"

    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt_path}")

    success = registry.hot_reload(model_id, str(ckpt_path))
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to reload model '{model_id}'")

    meta = registry.get_metadata(model_id)
    logger.info(f"Model '{model_id}' successfully hot-reloaded from {ckpt_path}")

    return JSONResponse(
        content={
            "status": "success",
            "model_id": model_id,
            "checkpoint": str(ckpt_path),
            "metadata": meta,
        }
    )
