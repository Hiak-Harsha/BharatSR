"""
BharatSR — Inference Service
Model loading, inference pipeline, and result generation.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend.app.models_ml.srcnn import SRCNN
from backend.app.models_ml.rcan import RCAN
from backend.app.models_ml.uncertainty import logvar_to_std


class ModelRegistry:
    """
    Manages loaded models. Models are loaded once at startup and kept in memory.
    Supports swapping between SRCNN (Phase 2) and RCAN (Phase 5).
    """

    def __init__(self):
        self._models: Dict[str, torch.nn.Module] = {}
        self._metadata: Dict[str, dict] = {}
        self._device = torch.device("cpu")

    def load_model(self, model_id: str, checkpoint_path: str) -> bool:
        """Load a model from a checkpoint file."""
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(str(path), map_location=self._device, weights_only=False)
            n_bands = checkpoint.get("n_bands", 4)
            scale_factor = checkpoint.get("scale_factor", 4)

            if model_id == "srcnn":
                model = SRCNN(n_bands=n_bands)
                name = "SRCNN Baseline"
                arch = "3-layer CNN (L1 loss)"
            elif model_id == "rcan":
                n_feats = checkpoint.get("n_feats", 36)
                n_resgroups = checkpoint.get("n_resgroups", 3)
                n_resblocks = checkpoint.get("n_resblocks", 3)
                model = RCAN(
                    n_bands=n_bands,
                    n_feats=n_feats,
                    n_resgroups=n_resgroups,
                    n_resblocks=n_resblocks,
                    scale=scale_factor,
                    predict_uncertainty=True,
                )
                name = "RCAN Attention + Uncertainty"
                arch = f"RIR ({n_resgroups} RGs, {n_resblocks} RCABs, Channel Attention)"
            else:
                print(f"Unknown model ID: {model_id}")
                return False

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self._device)
            model.eval()

            self._models[model_id] = model
            self._metadata[model_id] = {
                "id": model_id,
                "name": name,
                "architecture": arch,
                "n_bands": n_bands,
                "scale_factor": scale_factor,
                "epoch": checkpoint.get("epoch", "unknown"),
                "val_loss": checkpoint.get("val_loss", None),
                "has_uncertainty": (model_id == "rcan"),
                "status": "loaded",
            }
            print(f"Loaded model '{model_id}' ({name}) from {path}")
            return True

        except Exception as e:
            print(f"Error loading model '{model_id}': {e}")
            return False

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[dict]:
        return self._metadata.get(model_id)

    def list_models(self) -> list:
        return list(self._metadata.values())

    @property
    def device(self):
        return self._device


def run_inference(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    scale_factor: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Run super-resolution inference on a single LR image.

    Args:
        model: loaded PyTorch model
        lr_image: (C, H, W) numpy array in reflectance scale
        scale_factor: upscaling factor
        device: torch device

    Returns:
        sr_image: (C, H*scale, W*scale) numpy array
        inference_time: seconds taken
        uncertainty_map: (H*scale, W*scale) standard deviation map or None
    """
    model.eval()

    # Prepare input tensor (1, C, H, W)
    lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).to(device)

    uncertainty_map = None
    t0 = time.time()

    with torch.no_grad():
        if isinstance(model, SRCNN):
            lr_up = F.interpolate(lr_tensor, scale_factor=scale_factor, mode="bicubic", align_corners=False)
            sr_tensor = model(lr_up)
        elif isinstance(model, RCAN):
            res = model(lr_tensor)
            if isinstance(res, tuple):
                sr_tensor, log_var = res
                logvar_np = log_var.squeeze(0).squeeze(0).cpu().numpy()
                uncertainty_map = logvar_to_std(logvar_np)
            else:
                sr_tensor = res
        else:
            sr_tensor = model(lr_tensor)

    inference_time = time.time() - t0
    sr_image = sr_tensor.squeeze(0).cpu().numpy()

    return sr_image, inference_time, uncertainty_map


def run_bicubic_baseline(
    lr_image: np.ndarray,
    scale_factor: int = 4
) -> Tuple[np.ndarray, float]:
    """
    Run standard bicubic interpolation baseline.
    Returns (C, H*scale, W*scale) numpy array and latency.
    """
    t0 = time.time()
    lr_tensor = torch.from_numpy(lr_image).unsqueeze(0)
    with torch.no_grad():
        sr_tensor = F.interpolate(
            lr_tensor, scale_factor=scale_factor, mode="bicubic", align_corners=False
        )
    inference_time = time.time() - t0
    sr_image = sr_tensor.squeeze(0).numpy()
    return sr_image, inference_time


# Global registry instance
model_registry = ModelRegistry()

