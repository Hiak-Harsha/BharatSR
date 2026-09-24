"""
BharatSR — Core SR Pipeline Execution Service
Encapsulates data loading, multi-model execution, uncertainty aggregation,
and async worker processing.
"""

import io
import json
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import HTTPException

from backend.app.services.inference import (
    model_registry,
    run_inference,
    run_inference_ensembled,
    run_bicubic_baseline,
    run_tiled_inference,
)
from backend.app.services.preprocessing import (
    load_image_from_bytes,
    load_sample_tile,
    generate_multi_spectral_views,
)
from backend.app.services.postprocessing import compute_inference_metrics
from backend.app.models_ml.uncertainty import generate_uncertainty_heatmap, summarize_uncertainty
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.pipeline")


@dataclass
class InferenceResult:
    """Carries SR numerical array and serializable response payload."""

    array: np.ndarray
    response_dict: Dict[str, Any]
    model_id: str
    latency: float
    uncertainty_map: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_response_dict(self) -> Dict[str, Any]:
        """Returns the dictionary representation conforming to SuperResolveResponse schema."""
        return self.response_dict


def mask_to_png_bytes(mask: np.ndarray, color_rgb: Tuple[int, int, int]) -> bytes:
    """Renders binary mask as RGBA PNG bytes with color overlay."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    active = mask > 0
    rgba[active, 0] = color_rgb[0]
    rgba[active, 1] = color_rgb[1]
    rgba[active, 2] = color_rgb[2]
    rgba[active, 3] = 220  # Alpha opacity
    rgba[~active, 3] = 0

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def load_input_data(
    sample_id: Optional[str],
    file_bytes: Optional[bytes],
    sample_tiles_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Loads LR image, optional HR ground truth, and legitimate geospatial metadata.
    Never fabricates coordinates or CRS.
    """
    if sample_id:
        sample_path = sample_tiles_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
        lr_image, hr_image = load_sample_tile(str(sample_path))

        # Check for genuine metadata sidecar
        sidecar_path = sample_tiles_dir / f"{sample_id}.json"
        geo_metadata = None
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r") as f:
                    sidecar = json.load(f)
                    if sidecar.get("has_geo", False):
                        geo_metadata = sidecar
                    else:
                        geo_metadata = {
                            "has_geo": False,
                            "message": "No geospatial reference available (Synthetic procedural benchmark)",
                            "source_dataset": sidecar.get("source_dataset", "Synthetic"),
                        }
            except Exception:
                geo_metadata = {"has_geo": False, "message": "No geospatial reference available"}
        else:
            geo_metadata = {"has_geo": False, "message": "No geospatial reference available"}

        return lr_image, hr_image, geo_metadata

    elif file_bytes:
        try:
            lr_image, geo_metadata = load_image_from_bytes(file_bytes)
            return lr_image, None, geo_metadata
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'sample_id'")


def execute_model_sr(
    model_id: str,
    lr_image: np.ndarray,
    hr_image: Optional[np.ndarray],
    scale_factor: int = 4,
    quality: str = "fast",
) -> InferenceResult:
    """
    Executes a single model inference and returns an InferenceResult carrying
    both the raw Float32 array and formatted response dictionary.
    """
    uncertainty_dict = None
    use_ensemble = quality == "high"

    if model_id == "bicubic":
        sr_image, latency = run_bicubic_baseline(lr_image, scale_factor=scale_factor)
        uncertainty_map = None
    else:
        model = model_registry.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' not loaded or checkpoint missing.")
        _, h_lr, w_lr = lr_image.shape
        if h_lr > 64 or w_lr > 64:
            sr_image, latency, uncertainty_map = run_tiled_inference(
                model=model,
                lr_image=lr_image,
                scale_factor=scale_factor,
                tile_size=64,
                overlap=16,
                device=model_registry.device,
                use_ensemble=use_ensemble,
            )
        elif use_ensemble:
            sr_image, latency, uncertainty_map = run_inference_ensembled(
                model, lr_image, scale_factor, model_registry.device
            )
        else:
            sr_image, latency, uncertainty_map = run_inference(
                model, lr_image, scale_factor, model_registry.device
            )

    metrics = compute_inference_metrics(
        sr=sr_image,
        hr=hr_image,
        lr_original=lr_image,
        scale_factor=scale_factor,
    )

    views = generate_multi_spectral_views(sr_image)
    sr_b64 = views.get("rgb", "")

    error_dict = None
    if hr_image is not None:
        error_map = np.mean(np.abs(sr_image.astype(np.float32) - hr_image.astype(np.float32)), axis=0)
        err_bytes = generate_uncertainty_heatmap(error_map, colormap="plasma")
        err_b64 = f"data:image/png;base64,{base64.b64encode(err_bytes).decode('utf-8')}"
        views["error"] = err_b64
        error_dict = {
            "image": err_b64,
            "mean_error": round(float(np.mean(error_map)), 5),
            "max_error": round(float(np.max(error_map)), 5),
        }

    if uncertainty_map is not None:
        u_bytes = generate_uncertainty_heatmap(uncertainty_map, colormap="magma")
        u_b64 = base64.b64encode(u_bytes).decode("utf-8")
        uncertainty_dict = {
            "image": f"data:image/png;base64,{u_b64}",
            "summary": summarize_uncertainty(uncertainty_map),
        }

        if hr_image is not None and error_dict is not None:
            u_map = uncertainty_map[0] if uncertainty_map.ndim == 3 else uncertainty_map
            h_s, w_s = error_map.shape
            sy = max(1, h_s // 8)
            sx = max(1, w_s // 8)
            u_sub = u_map[::sy, ::sx].flatten().astype(float)
            e_sub = error_map[::sy, ::sx].flatten().astype(float)
            n_pts = min(len(u_sub), len(e_sub), 64)
            scatter_points = [
                {"unc": round(float(u_sub[i]), 4), "err": round(float(e_sub[i]), 4)}
                for i in range(n_pts)
            ]
            corr_val = float(np.corrcoef(u_sub[:n_pts], e_sub[:n_pts])[0, 1]) if n_pts > 2 else 0.0
            uncertainty_dict["scatter"] = {
                "correlation": round(corr_val if not np.isnan(corr_val) else 0.0, 4),
                "points": scatter_points,
            }

    response_dict = {
        "model_id": model_id,
        "quality": "high" if use_ensemble else "fast",
        "inference_time_s": round(latency, 4),
        "shape": list(sr_image.shape),
        "image": sr_b64,
        "views": views,
        "metrics": metrics,
        "uncertainty": uncertainty_dict,
        "error_map": error_dict,
    }

    return InferenceResult(
        array=sr_image,
        response_dict=response_dict,
        model_id=model_id,
        latency=latency,
        uncertainty_map=uncertainty_map,
        metrics=metrics,
    )


def async_worker(
    job_id: str,
    sample_id: Optional[str],
    file_bytes: Optional[bytes],
    model_id: str,
    quality: str,
    job_store: Any,
    runs_dir: Path,
    sample_tiles_dir: Path,
) -> None:
    """Threadpool worker for asynchronous background execution of super-resolution jobs."""
    try:
        job_store.update_job(job_id, status="processing", progress_pct=10)
        lr_image, hr_image, geo_metadata = load_input_data(sample_id, file_bytes, sample_tiles_dir)

        if job_store.is_cancelled(job_id):
            return

        job_store.update_job(job_id, status="processing", progress_pct=40)
        result = execute_model_sr(model_id, lr_image, hr_image, scale_factor=4, quality=quality)

        if job_store.is_cancelled(job_id):
            return

        job_store.update_job(job_id, status="processing", progress_pct=80)

        # Cache run arrays for downstream export and profile analysis
        run_file = runs_dir / f"{job_id}.npz"
        np.savez_compressed(
            str(run_file),
            sr=result.array,
            lr=lr_image,
            hr=hr_image if hr_image is not None else np.zeros((0,), dtype=np.float32),
            has_hr=hr_image is not None,
            geo_json=json.dumps(geo_metadata) if geo_metadata else "",
            model_id=model_id,
            scale_factor=4,
        )

        job_store.update_job(
            job_id,
            status="completed",
            result_path=str(run_file),
            metrics=result.metrics,
            inference_time=result.latency,
            progress_pct=100,
        )
        logger.info(f"Async job {job_id} successfully completed in {result.latency:.3f}s")

    except Exception as e:
        logger.exception(f"Async job {job_id} failed: {e}")
        job_store.update_job(
            job_id,
            status="failed",
            error=str(e),
            progress_pct=100,
        )
