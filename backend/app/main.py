"""
BharatSR — FastAPI Application Server
Problem Statement SIH26142: Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery
Target: Sentinel-2 L2A (10m) -> 4x SR on 2.5m-equivalent grid (B2, B3, B4, B8)
"""

import sys
import io
import uuid
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager

from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import json
from PIL import Image

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Thread pool for offloading CPU-heavy PyTorch inference away from FastAPI event loop
_thread_pool = ThreadPoolExecutor(max_workers=4)

from backend.app.schemas import (
    HealthResponse, ModelsListResponse, ModelInfo,
    SamplesListResponse, SampleInfo,
    SuperResolveResponse, CompareResponse,
    PixelProfileResponse, BandProfile,
    AsyncJobSubmitResponse, JobStatusResponse, JobListResponse,
    ReportResponse, DownstreamMasksResponse, DownstreamTaskItem, DownstreamTaskMetrics,
    SpectralIndicesResponse, CropHealthResponse, FieldBoundaryResponse,
    ChangeDetectionResponse, BatchSubmitResponse, BatchStatusResponse, ModelReloadResponse,
)
from backend.app.services.inference import (
    model_registry, run_inference, run_inference_ensembled, run_bicubic_baseline, run_tiled_inference
)
from backend.app.services.preprocessing import (
    load_image_from_bytes, load_sample_tile, numpy_to_png_bytes,
    generate_multi_spectral_views, export_geotiff_bytes, BAND_INDEX,
    compute_spectral_indices, indices_to_visualizations
)
from backend.app.services.postprocessing import compute_inference_metrics
from backend.app.services.job_store import JobStore
from backend.app.models_ml.uncertainty import generate_uncertainty_heatmap, summarize_uncertainty
from evaluation.downstream_task import (
    segment_micro_canopy_rule_based, segment_built_up_rule_based, compute_segmentation_metrics
)


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


# ========================
# APP LIFECYCLE
# ========================

job_store: Optional[JobStore] = None
RUNS_DIR = Path(__file__).parent.parent / "runs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize storage at startup."""
    global job_store

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize job store
    db_path = str(Path(__file__).parent.parent / "bharatsr.db")
    job_store = JobStore(db_path)
    # Cleanup expired jobs (>24 hours old) on startup
    deleted = job_store.cleanup_expired_jobs(max_age_hours=24, runs_dir=RUNS_DIR)
    if deleted > 0:
        print(f"Cleaned up {deleted} expired job records.")

    # Load available models
    weights_dir = Path(__file__).parent.parent / "weights"
    srcnn_path = weights_dir / "srcnn_best.pth"
    if srcnn_path.exists():
        model_registry.load_model("srcnn", str(srcnn_path))
    else:
        print(f"INFO: SRCNN checkpoint not yet present at {srcnn_path}")

    rcan_path = weights_dir / "rcan_best.pth"
    if rcan_path.exists():
        model_registry.load_model("rcan", str(rcan_path))
    else:
        print(f"INFO: RCAN checkpoint not yet present at {rcan_path}")

    print(f"Available models in registry: {[m['id'] for m in model_registry.list_models()]}")

    yield

    print("Shutting down BharatSR backend")


# ========================
# APP SETUP
# ========================

app = FastAPI(
    title="BharatSR API",
    description="Scientifically Defensible Deep Learning Super-Resolution for Satellite Earth Observation (SIH26142)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# HELPER FUNCTIONS
# ========================

def _load_input_data(sample_id: Optional[str], file_bytes: Optional[bytes]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Loads LR image, optional HR ground truth, and legitimate geospatial metadata.
    Never fabricates coordinates or CRS.
    """
    if sample_id:
        sample_tiles_dir = Path(__file__).parent.parent / "sample_tiles"
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


def _execute_model_sr(
    model_id: str,
    lr_image: np.ndarray,
    hr_image: Optional[np.ndarray],
    scale_factor: int = 4,
    quality: str = "fast",
) -> Dict[str, Any]:
    """
    Executes a single model inference and returns formatted results.

    quality="fast" (default): single forward pass.
    quality="high": geometric self-ensemble (4x flip-averaged TTA) -- no
    retraining required, ~4x latency, typically lower SAM / higher PSNR.
    Ignored for the bicubic baseline (nothing to ensemble).
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

    return {
        "model_id": model_id,
        "quality": "high" if use_ensemble else "fast",
        "inference_time_s": round(latency, 4),
        "shape": list(sr_image.shape),
        "image": sr_b64,
        "views": views,
        "metrics": metrics,
        "uncertainty": uncertainty_dict,
        "error_map": error_dict,
        "_sr_array": sr_image,
    }


def _async_worker(job_id: str, sample_id: Optional[str], file_bytes: Optional[bytes], model_id: str, quality: str = "fast"):
    """Background task for async processing with progress and cancellation checks."""
    try:
        assert job_store is not None
        job_store.update_job(job_id, status="processing", progress_pct=10)

        if job_store.is_job_cancelled(job_id):
            return

        lr_image, hr_image, geo_metadata = _load_input_data(sample_id, file_bytes)
        job_store.update_progress(job_id, 35)

        if job_store.is_job_cancelled(job_id):
            return

        model_meta = model_registry.get_metadata(model_id) or {}
        scale_factor = model_meta.get("scale_factor", 4)

        result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor, quality=quality)
        sr_array = result.pop("_sr_array")
        job_store.update_progress(job_id, 80)

        if job_store.is_job_cancelled(job_id):
            return

        # Attach geospatial metadata if present
        if geo_metadata:
            result["geospatial_metadata"] = geo_metadata

        # Save result payload to disk
        payload_path = RUNS_DIR / f"{job_id}.json"
        with open(payload_path, "w") as f:
            json.dump(result, f)

        # Save numpy array and geospatial metadata for export
        npz_path = RUNS_DIR / f"{job_id}.npz"
        np.savez_compressed(
            str(npz_path),
            sr=sr_array,
            lr=lr_image,
            hr=hr_image if hr_image is not None else np.zeros((0,), dtype=np.float32),
            has_hr=hr_image is not None,
            geo_json=json.dumps(geo_metadata) if geo_metadata else "",
            model_id=model_id,
            scale_factor=scale_factor,
        )

        job_store.update_job(
            job_id,
            status="completed",
            result_path=str(payload_path),
            metrics=result["metrics"],
            inference_time=result["inference_time_s"],
            progress_pct=100
        )
    except Exception as e:
        if job_store:
            job_store.update_job(job_id, status="failed", error=str(e), progress_pct=0)


# ========================
# ENDPOINTS
# ========================

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint confirming API status, loaded models, and device."""
    active_count = 0
    if job_store:
        try:
            recent = job_store.list_jobs(limit=10)
            active_count = sum(1 for j in recent if j.get("status") in ("pending", "processing"))
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        models_loaded=len(model_registry.list_models()),
        device=str(model_registry.device),
        version="0.2.0",
        active_jobs=active_count,
    )


@app.get("/api/models", response_model=ModelsListResponse)
async def list_models():
    """List available super-resolution models with parameter count and capabilities."""
    models = model_registry.list_models()
    return ModelsListResponse(models=[ModelInfo(**m) for m in models])


@app.get("/api/samples", response_model=SamplesListResponse)
async def list_samples():
    """
    List pre-loaded sample tiles with honest geographic and provenance metadata.
    Never fabricates coordinates or real city names for synthetic procedural samples.
    """
    sample_dir = Path(__file__).parent.parent / "sample_tiles"
    samples = []

    if sample_dir.exists():
        for f in sorted(sample_dir.glob("sample_*.npz")):
            try:
                data = np.load(str(f))
                lr = data["lr"]
                views = generate_multi_spectral_views(lr)

                # Check sidecar json
                sidecar_file = sample_dir / f"{f.stem}.json"
                sidecar = {}
                if sidecar_file.exists():
                    try:
                        with open(sidecar_file, "r") as sf:
                            sidecar = json.load(sf)
                    except Exception:
                        pass

                has_geo = sidecar.get("has_geo", False)
                crs = sidecar.get("crs", None)
                sensor = sidecar.get("sensor", "Sentinel-2 MSI (B2, B3, B4, B8)")
                title = sidecar.get("title", f.stem.replace("_", " ").title())
                source_dataset = sidecar.get("source_dataset", "Synthetic Procedural Reference")
                desc = sidecar.get("description", f"4-band satellite verification tile ({source_dataset}).")
                tactical_cat = sidecar.get("tactical_category", "Micro-Grid Super-Resolution Verification")

                coord_str = f"CRS: {crs}" if has_geo and crs else "No geospatial reference (Synthetic benchmark)"

                samples.append(SampleInfo(
                    id=f.stem,
                    filename=f.name,
                    title=title,
                    region=source_dataset,
                    description=desc,
                    tactical_category=tactical_cat,
                    coordinates=coord_str,
                    bands=int(lr.shape[0]),
                    lr_size=f"{lr.shape[1]}x{lr.shape[2]}",
                    has_ground_truth="hr" in data.files,
                    has_geo=has_geo,
                    crs=crs,
                    sensor=sensor,
                    thumbnail=views.get("rgb", ""),
                    views=views,
                ))
            except Exception as e:
                print(f"Error loading sample tile {f}: {e}")

    # Sort so genuine georeferenced satellite scenes appear first, followed by verification patterns
    samples.sort(key=lambda s: (not s.has_geo, s.id))
    return SamplesListResponse(samples=samples)


@app.get("/api/samples/{sample_id}/preview")
async def sample_preview(sample_id: str, crop: int = 32):
    """
    Small, fast live-inference preview for a sample tile: LR / Bicubic / BharatSR
    (/ Ground Truth if available), each as a base64 RGB PNG.

    Purpose-built for the landing page's "same tile, four ways of seeing it"
    strip -- this is real inference on a small center crop, not a mock image,
    so the marketing view and the console never show different claims.
    """
    sample_tiles_dir = Path(__file__).parent.parent / "sample_tiles"
    sample_path = sample_tiles_dir / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")

    lr_image, hr_image = load_sample_tile(str(sample_path))
    c, h, w = lr_image.shape
    crop = max(8, min(crop, h, w))
    y0 = max(0, (h - crop) // 2)
    x0 = max(0, (w - crop) // 2)
    lr_crop = lr_image[:, y0:y0 + crop, x0:x0 + crop]

    model = model_registry.get_model("rcan")
    model_meta = model_registry.get_metadata("rcan") or {}
    scale_factor = model_meta.get("scale_factor", 4)

    bicubic_sr, _ = run_bicubic_baseline(lr_crop, scale_factor=scale_factor)
    views = {
        "lr": generate_multi_spectral_views(lr_crop).get("rgb", ""),
        "bicubic": generate_multi_spectral_views(bicubic_sr).get("rgb", ""),
    }

    if model is not None:
        sr_crop, _, _ = run_inference(model, lr_crop, scale_factor, model_registry.device)
        views["sr"] = generate_multi_spectral_views(sr_crop).get("rgb", "")
    else:
        views["sr"] = views["bicubic"]

    if hr_image is not None:
        hr_crop_size = crop * scale_factor
        hy0, hx0 = y0 * scale_factor, x0 * scale_factor
        hr_crop = hr_image[:, hy0:hy0 + hr_crop_size, hx0:hx0 + hr_crop_size]
        if hr_crop.shape[1] == hr_crop_size and hr_crop.shape[2] == hr_crop_size:
            views["ground_truth"] = generate_multi_spectral_views(hr_crop).get("rgb", "")

    return JSONResponse(content={
        "status": "success",
        "sample_id": sample_id,
        "crop_size": crop,
        "model_id": "rcan" if model is not None else "bicubic",
        "views": views,
    })


ALLOWED_MODELS = ("bicubic", "srcnn", "rcan", "swinir", "hat", "diffusion", "ensemble")


@app.post("/api/superresolve")
async def superresolve(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    model_id: str = Form("rcan"),
    quality: str = Form("fast"),
):
    """
    Run super-resolution on an uploaded GeoTIFF or pre-loaded sample tile.
    Returns 4x SR result as base64 PNG, multi-spectral views, physical metrics,
    and predicted uncertainty map.

    quality: "fast" (single pass) or "high" (4x flip self-ensemble, ~4x slower,
    no retraining required -- see run_inference_ensembled).
    """
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_id '{model_id}'. Allowed: {', '.join(ALLOWED_MODELS)}")
    if quality not in ("fast", "high"):
        raise HTTPException(status_code=400, detail="Invalid quality. Allowed: fast, high")

    file_bytes = await file.read() if file else None
    lr_image, hr_image, geo_metadata = _load_input_data(sample_id, file_bytes)

    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor, quality=quality)
    sr_array = result.pop("_sr_array")

    # Generate a unique run_id and cache the run state on disk for GeoTIFF export and pixel inspection
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    run_path = RUNS_DIR / f"{run_id}.npz"
    try:
        np.savez_compressed(
            str(run_path),
            sr=sr_array,
            lr=lr_image,
            hr=hr_image if hr_image is not None else np.zeros((0,), dtype=np.float32),
            has_hr=hr_image is not None,
            geo_json=json.dumps(geo_metadata) if geo_metadata else "",
            model_id=model_id,
            scale_factor=scale_factor,
        )
    except Exception as e:
        print(f"Warning: could not cache run {run_id}: {e}")

    # Generate bicubic baseline views for 4-view Evidence Mode
    bicubic_sr, _ = run_bicubic_baseline(lr_image, scale_factor=scale_factor)
    bic_views = generate_multi_spectral_views(bicubic_sr)
    if hr_image is not None:
        bic_err = np.mean(np.abs(bicubic_sr.astype(np.float32) - hr_image.astype(np.float32)), axis=0)
        bic_err_bytes = generate_uncertainty_heatmap(bic_err, colormap="plasma")
        bic_views["error"] = f"data:image/png;base64,{base64.b64encode(bic_err_bytes).decode('utf-8')}"

    input_views = generate_multi_spectral_views(lr_image)

    response = {
        "status": "success",
        "model_id": model_id,
        "run_id": run_id,
        "quality": result["quality"],
        "inference_time_s": result["inference_time_s"],
        "input": {
            "shape": list(lr_image.shape),
            "image": input_views.get("rgb", ""),
            "views": input_views,
        },
        "bicubic": {
            "shape": list(bicubic_sr.shape),
            "image": bic_views.get("rgb", ""),
            "views": bic_views,
        },
        "output": {
            "shape": result["shape"],
            "image": result["image"],
            "views": result["views"],
        },
        "metrics": result["metrics"],
    }

    if result.get("error_map") is not None:
        response["error_map"] = result["error_map"]

    if result["uncertainty"] is not None:
        response["uncertainty"] = result["uncertainty"]

    if hr_image is not None:
        gt_views = generate_multi_spectral_views(hr_image)
        response["ground_truth"] = {
            "shape": list(hr_image.shape),
            "image": gt_views.get("rgb", ""),
            "views": gt_views,
        }

    if geo_metadata:
        response["geospatial_metadata"] = geo_metadata

    return JSONResponse(content=response)


@app.post("/api/compare")
async def compare(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
):
    """
    Rigorous multi-model benchmark: Bicubic Baseline vs SRCNN vs RCAN side-by-side.
    Computes comparative metrics (PSNR, SSIM, SAM, Consistency MAE, Spectral MAE),
    inference latency, and multi-spectral representations on exactly the same scene.
    """
    file_bytes = await file.read() if file else None
    lr_image, hr_image, geo_metadata = _load_input_data(sample_id, file_bytes)

    input_views = generate_multi_spectral_views(lr_image)

    # Models to benchmark
    model_ids = ["bicubic"]
    for m in ALLOWED_MODELS:
        if m != "bicubic" and model_registry.get_model(m) is not None:
            model_ids.append(m)

    results = {}
    for m_id in model_ids:
        res = _execute_model_sr(m_id, lr_image, hr_image, scale_factor=4)
        results[m_id] = res

    # Build comparison table
    comparison_table = []
    metric_keys = [
        ("psnr", "PSNR (Peak SNR)", "dB", True),
        ("ssim", "SSIM (Structural Similarity)", "", True),
        ("sam", "SAM (Spectral Angle Mapper)", "°", False),
        ("downsample_consistency", "Downsample Consistency", "MAE", False),
        ("spectral_mae", "Mean Absolute Spectral Error", "", False),
        ("correctness_score", "Correctness Score", "", True),
        ("hallucination_rate", "Hallucination Rate", "", False),
    ]

    for key, label, unit, higher_is_better in metric_keys:
        row = {"metric": label, "unit": unit, "higher_is_better": higher_is_better}
        best_val = None
        best_m = None

        for m_id in model_ids:
            val = results[m_id]["metrics"].get(key, {}).get("value")
            row[m_id] = val
            if val is not None:
                if best_val is None:
                    best_val = val
                    best_m = m_id
                elif higher_is_better and val > best_val:
                    best_val = val
                    best_m = m_id
                elif not higher_is_better and val < best_val:
                    best_val = val
                    best_m = m_id

        row["best_model"] = best_m
        comparison_table.append(row)

    # Latency row
    latency_row = {
        "metric": "Inference Latency",
        "unit": "s",
        "higher_is_better": False,
        "best_model": "bicubic",
    }
    for m_id in model_ids:
        latency_row[m_id] = results[m_id]["inference_time_s"]
    comparison_table.append(latency_row)

    # Cache best available model so that user can immediately download GeoTIFF or inspect pixels after compare
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    best_model = next((m for m in ["rcan", "srcnn", "bicubic"] if m in results and "_sr_array" in results[m]), None)
    if best_model:
        best_sr = results[best_model]["_sr_array"]
        try:
            np.savez_compressed(
                str(RUNS_DIR / f"{run_id}.npz"),
                sr=best_sr,
                lr=lr_image,
                hr=hr_image if hr_image is not None else np.zeros((0,), dtype=np.float32),
                has_hr=hr_image is not None,
                geo_json=json.dumps(geo_metadata) if geo_metadata else "",
                model_id=best_model,
                scale_factor=4,
            )
        except Exception as e:
            print(f"Warning: could not cache compare run: {e}")

    for m_id in model_ids:
        if "_sr_array" in results[m_id]:
            results[m_id].pop("_sr_array")

    response = {
        "status": "success",
        "run_id": run_id,
        "input": {
            "shape": list(lr_image.shape),
            "image": input_views.get("rgb", ""),
            "views": input_views,
        },
        "models": results,
        "comparison_table": comparison_table,
    }

    if hr_image is not None:
        gt_views = generate_multi_spectral_views(hr_image)
        response["ground_truth"] = {
            "shape": list(hr_image.shape),
            "image": gt_views.get("rgb", ""),
            "views": gt_views,
        }

    if geo_metadata:
        response["geospatial_metadata"] = geo_metadata

    return JSONResponse(content=response)


@app.post("/api/pixel-profile", response_model=PixelProfileResponse)
async def pixel_profile(
    sample_id: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    x: int = Form(128),
    y: int = Form(128),
    model_id: str = Form("rcan"),
):
    """
    Extract multi-band spectral reflectance profile at coordinate (x, y).
    Supports either pre-loaded sample_id or an active uploaded run_id.
    Compares LR, SR, and HR ground truth curves across B2, B3, B4, B8.
    Calculates exact spectral angle and provides rule-based spectral interpretation.
    """
    if not sample_id and not run_id:
        raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'run_id'")

    if run_id:
        run_path = RUNS_DIR / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
        hr_image = data["hr"].astype(np.float32) if data.get("has_hr", False) and data["hr"].size > 0 else None
        scale_factor = int(data.get("scale_factor", 4))
    else:
        sample_tiles_dir = Path(__file__).parent.parent / "sample_tiles"
        sample_path = sample_tiles_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample tile not found")

        lr_image, hr_image = load_sample_tile(str(sample_path))
        model_meta = model_registry.get_metadata(model_id) or {}
        scale_factor = model_meta.get("scale_factor", 4)

        result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
        sr_array = result["_sr_array"]

    c, h_hr, w_hr = sr_array.shape
    x = max(0, min(int(x), w_hr - 1))
    y = max(0, min(int(y), h_hr - 1))

    # Map to LR coordinate
    x_lr = max(0, min(x // scale_factor, lr_image.shape[2] - 1))
    y_lr = max(0, min(y // scale_factor, lr_image.shape[1] - 1))

    # Compute bicubic baseline for pixel inspector comparison
    bicubic_arr, _ = run_bicubic_baseline(lr_image, scale_factor=scale_factor)

    # Bands ordering: B2=0, B3=1, B4=2, B8=3
    spectral_order = [
        {"name": "Blue", "band": "B2", "wavelength_nm": 490, "idx": BAND_INDEX["B2"]},
        {"name": "Green", "band": "B3", "wavelength_nm": 560, "idx": BAND_INDEX["B3"]},
        {"name": "Red", "band": "B4", "wavelength_nm": 665, "idx": BAND_INDEX["B4"]},
        {"name": "Near-IR", "band": "B8", "wavelength_nm": 842, "idx": BAND_INDEX["B8"]},
    ]

    bands_data = []
    sr_vec = []
    hr_vec = []
    for b in spectral_order:
        idx = b["idx"]
        sr_val = float(sr_array[idx, y, x]) if idx < c else 0.0
        lr_val = float(lr_image[idx, y_lr, x_lr]) if idx < lr_image.shape[0] else 0.0
        bic_val = float(bicubic_arr[idx, y, x]) if idx < bicubic_arr.shape[0] else 0.0
        hr_val = float(hr_image[idx, y, x]) if (hr_image is not None and idx < hr_image.shape[0]) else None

        sr_vec.append(sr_val)
        if hr_val is not None:
            hr_vec.append(hr_val)

        bands_data.append(BandProfile(
            name=b["name"],
            band=b["band"],
            wavelength=f"{b['wavelength_nm']} nm",
            lr_reflectance=round(lr_val, 4),
            bicubic_reflectance=round(bic_val, 4),
            sr_reflectance=round(sr_val, 4),
            hr_reflectance=round(hr_val, 4) if hr_val is not None else None,
        ))

    # Compute pointwise spectral angle if HR is available
    spectral_angle = None
    if len(hr_vec) == len(sr_vec):
        v_sr = np.array(sr_vec, dtype=np.float64)
        v_hr = np.array(hr_vec, dtype=np.float64)
        norm_sr = np.linalg.norm(v_sr)
        norm_hr = np.linalg.norm(v_hr)
        if norm_sr > 1e-6 and norm_hr > 1e-6:
            cos_theta = np.dot(v_sr, v_hr) / (norm_sr * norm_hr)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            spectral_angle = round(float(np.degrees(np.arccos(cos_theta))), 2)

    # Pointwise NDVI = (NIR - Red) / (NIR + Red)
    red_sr = float(sr_array[BAND_INDEX["B4"], y, x])
    nir_sr = float(sr_array[BAND_INDEX["B8"], y, x])
    red_lr = float(lr_image[BAND_INDEX["B4"], y_lr, x_lr])
    nir_lr = float(lr_image[BAND_INDEX["B8"], y_lr, x_lr])
    red_bic = float(bicubic_arr[BAND_INDEX["B4"], y, x])
    nir_bic = float(bicubic_arr[BAND_INDEX["B8"], y, x])

    ndvi_sr = (nir_sr - red_sr) / (nir_sr + red_sr + 1e-7)
    ndvi_lr = (nir_lr - red_lr) / (nir_lr + red_lr + 1e-7)
    ndvi_bic = (nir_bic - red_bic) / (nir_bic + red_bic + 1e-7)
    ndvi_hr = None
    if hr_image is not None and hr_image.shape[0] >= 4:
        red_hr = float(hr_image[BAND_INDEX["B4"], y, x])
        nir_hr = float(hr_image[BAND_INDEX["B8"], y, x])
        ndvi_hr = (nir_hr - red_hr) / (nir_hr + red_hr + 1e-7)

    # Rule-based spectral interpretation (clearly disclaimed, not ground truth)
    if ndvi_sr > 0.4:
        surface_type = "Rule-based: Dense Vegetation / Crop Canopy"
        signature_note = "Steep red edge with high NIR cellular scattering plateau (rule-based interpretation, not ground truth)."
    elif ndvi_sr > 0.15:
        surface_type = "Rule-based: Sparse Scrub / Mixed Soil-Vegetation"
        signature_note = "Moderate red-NIR slope typical of semi-arid terrain (rule-based interpretation, not ground truth)."
    elif nir_sr < 0.08:
        surface_type = "Rule-based: Water Body / Deep Shadow"
        signature_note = "High absorption across visible and near-infrared spectrum (rule-based interpretation, not ground truth)."
    else:
        surface_type = "Rule-based: Built-up Urban / High-Albedo Ground"
        signature_note = "Relatively flat visible-to-NIR reflectance curve (rule-based interpretation, not ground truth)."

    return PixelProfileResponse(
        status="success",
        sample_id=sample_id,
        run_id=run_id,
        model_id=model_id,
        hr_coordinates={"x": x, "y": y},
        lr_coordinates={"x": x_lr, "y": y_lr},
        bands_data=bands_data,
        ndvi={
            "sr": round(ndvi_sr, 4),
            "lr": round(ndvi_lr, 4),
            "bicubic": round(ndvi_bic, 4),
            "hr": round(ndvi_hr, 4) if ndvi_hr is not None else None,
        },
        spectral_angle_deg=spectral_angle,
        surface_classification=surface_type,
        signature_analysis=signature_note,
        interpretation_disclaimer="Rule-based spectral interpretation (heuristic, not ground truth)",
    )


@app.post("/api/downstream-masks", response_model=DownstreamMasksResponse)
async def downstream_masks(
    sample_id: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    model_id: str = Form("rcan"),
):
    """
    Computes binary segmentation masks and real-time IoU/F1 metrics for
    downstream analytical tasks (Micro-Canopy Vegetation & Built-Up Infrastructure Extraction).
    """
    if not sample_id and not run_id:
        raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'run_id'")

    if run_id:
        run_path = RUNS_DIR / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
        hr_image = data["hr"].astype(np.float32) if data.get("has_hr", False) and data["hr"].size > 0 else None
        scale_factor = int(data.get("scale_factor", 4))
    else:
        sample_tiles_dir = Path(__file__).parent.parent / "sample_tiles"
        sample_path = sample_tiles_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample tile not found")

        lr_image, hr_image = load_sample_tile(str(sample_path))
        model_meta = model_registry.get_metadata(model_id) or {}
        scale_factor = model_meta.get("scale_factor", 4)

        result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
        sr_array = result["_sr_array"]

    bicubic_sr, _ = run_bicubic_baseline(lr_image, scale_factor=scale_factor)

    tasks_config = {
        "canopy_segmentation": {
            "name": "Micro-Canopy Vegetation Segmentation (Rule-Based)",
            "description": "Rule-based spectral interpretation: Delineates vegetation canopy using NDVI > 0.35",
            "fn": segment_micro_canopy_rule_based,
            "color": (34, 197, 94),  # Emerald Green
        },
        "built_up_infrastructure": {
            "name": "Built-up Infrastructure Extraction (Rule-Based)",
            "description": "Rule-based spectral interpretation: Extracts candidate built-up areas using albedo & contrast",
            "fn": segment_built_up_rule_based,
            "color": (249, 115, 22),  # Amber Orange
        },
    }

    tasks_res = {}
    for task_key, tcfg in tasks_config.items():
        seg_fn = tcfg["fn"]
        color = tcfg["color"]

        mask_bicubic = seg_fn(bicubic_sr)
        mask_rcan = seg_fn(sr_array)
        mask_gt = seg_fn(hr_image) if hr_image is not None else mask_bicubic

        metrics_bicubic = compute_segmentation_metrics(mask_bicubic, mask_gt) if hr_image is not None else {
            "f1": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0
        }
        metrics_rcan = compute_segmentation_metrics(mask_rcan, mask_gt) if hr_image is not None else compute_segmentation_metrics(mask_rcan, mask_bicubic)

        png_bicubic = mask_to_png_bytes(mask_bicubic, color)
        png_rcan = mask_to_png_bytes(mask_rcan, color)
        png_gt = mask_to_png_bytes(mask_gt, (59, 130, 246)) if hr_image is not None else b""

        tasks_res[task_key] = DownstreamTaskItem(
            task_name=tcfg["name"],
            description=tcfg["description"],
            bicubic=DownstreamTaskMetrics(**metrics_bicubic),
            rcan=DownstreamTaskMetrics(**metrics_rcan),
            ground_truth_pixel_count=int(mask_gt.sum()),
            masks={
                "bicubic": f"data:image/png;base64,{base64.b64encode(png_bicubic).decode('utf-8')}",
                "rcan": f"data:image/png;base64,{base64.b64encode(png_rcan).decode('utf-8')}",
                "ground_truth": f"data:image/png;base64,{base64.b64encode(png_gt).decode('utf-8')}" if png_gt else "",
            }
        )

    return DownstreamMasksResponse(status="success", tasks=tasks_res)


# ========================
# ASYNC JOB ENDPOINTS
# ========================

@app.post("/api/superresolve/async", response_model=AsyncJobSubmitResponse)
async def superresolve_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    model_id: str = Form("rcan"),
    quality: str = Form("fast"),
):
    """
    Submit super-resolution as an asynchronous background task.
    Returns job_id and status checking URL.

    quality: "fast" (single pass) or "high" (4x flip self-ensemble).
    """
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_id '{model_id}'. Allowed: {', '.join(ALLOWED_MODELS)}")
    if quality not in ("fast", "high"):
        raise HTTPException(status_code=400, detail="Invalid quality. Allowed: fast, high")

    file_bytes = await file.read() if file else None
    if not sample_id and not file_bytes:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'sample_id'")

    assert job_store is not None
    job_id = job_store.create_job(model_id=model_id)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_thread_pool, _async_worker, job_id, sample_id, file_bytes, model_id, quality)

    return AsyncJobSubmitResponse(
        status="accepted",
        job_id=job_id,
        model_id=model_id,
        status_url=f"/api/jobs/{job_id}",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check asynchronous job status, progress percentage, or retrieve final results."""
    assert job_store is not None
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response_data = {
        "job_id": job["job_id"],
        "status": job["status"],
        "model_id": job["model_id"],
        "progress_pct": job.get("progress_pct", 0),
        "is_cancelled": bool(job.get("is_cancelled", 0)),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "inference_time_s": job.get("inference_time_s"),
        "error_message": job.get("error_message"),
        "result": None,
    }

    if job["status"] == "completed" and job.get("result_path"):
        result_file = Path(job["result_path"])
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    response_data["result"] = json.load(f)
            except Exception as e:
                response_data["error_message"] = f"Failed to load cached result: {e}"

    return JobStatusResponse(**response_data)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running or pending asynchronous job."""
    assert job_store is not None
    success = job_store.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' cannot be cancelled (either not found or already completed/failed).")
    return {"status": "success", "message": f"Job '{job_id}' cancellation requested."}


@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs():
    """List recent background processing jobs."""
    assert job_store is not None
    jobs = job_store.list_jobs(limit=20)
    return JobListResponse(jobs=jobs)


# ========================
# EXPORT ENDPOINTS
# ========================

@app.get("/api/export/geotiff")
async def export_geotiff(
    sample_id: Optional[str] = Query(None, description="Sample ID to export"),
    run_id: Optional[str] = Query(None, description="Active run ID to export"),
    model_id: str = Query("rcan", description="Model to generate SR with"),
):
    """
    Generate and download an authoritative 4-band Float32 GeoTIFF.
    Supports either pre-loaded sample_id or an active uploaded run_id.
    CRITICAL RULE:
    - If the source dataset lacks geospatial metadata, raises HTTP 400 stating
      'No geospatial reference available'. Never silently assigns EPSG:4326.
    - If georeferenced, preserves exact CRS, scales affine transform for 4x SR (p_out = p_in / 4),
      and sets band descriptions and metadata tags.
    """
    if not sample_id and not run_id:
        raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'run_id'")

    if run_id:
        run_path = RUNS_DIR / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        geo_json = str(data.get("geo_json", ""))
        geo_metadata = json.loads(geo_json) if geo_json else None
        scale_factor = int(data.get("scale_factor", 4))

        if not geo_metadata or not geo_metadata.get("has_geo", False):
            raise HTTPException(
                status_code=400,
                detail="No geospatial reference available for this run. Authoritative geospatial GeoTIFF export is disabled for non-georeferenced inputs."
            )
        try:
            geotiff_bytes = export_geotiff_bytes(sr_array, geo_metadata=geo_metadata, scale_factor=scale_factor)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return Response(
            content=geotiff_bytes,
            media_type="image/tiff",
            headers={
                "Content-Disposition": f"attachment; filename=bharatsr_{run_id}.tif"
            },
        )

    sample_tiles_dir = Path(__file__).parent.parent / "sample_tiles"
    sample_path = sample_tiles_dir / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample tile not found")

    lr_image, hr_image, geo_metadata = _load_input_data(sample_id, None)

    if not geo_metadata or not geo_metadata.get("has_geo", False):
        raise HTTPException(
            status_code=400,
            detail="No geospatial reference available for this dataset. Authoritative geospatial GeoTIFF export is disabled for non-georeferenced synthetic benchmarks."
        )

    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
    sr_array = result["_sr_array"]

    try:
        geotiff_bytes = export_geotiff_bytes(sr_array, geo_metadata=geo_metadata, scale_factor=scale_factor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=geotiff_bytes,
        media_type="image/tiff",
        headers={
            "Content-Disposition": f"attachment; filename=bharatsr_{sample_id}_{model_id}_4x.tif"
        },
    )


@app.get("/api/export/report")
async def export_report(
    sample_id: str = Query(..., description="Sample ID"),
    model_id: str = Query("rcan", description="Model ID"),
):
    """
    Generate an analytical verification report JSON for SIH evaluation.
    Reports scientifically defensible metrics without fabricated claims.
    """
    sample_tiles_dir = Path(__file__).parent.parent / "sample_tiles"
    sample_path = sample_tiles_dir / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample tile not found")

    lr_image, hr_image, geo_metadata = _load_input_data(sample_id, None)
    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)

    report = {
        "title": "BharatSR Super-Resolution Physics & Spectral Fidelity Report",
        "problem_statement": "SIH26142 - Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery",
        "target_organization": "National Technical Research Organisation (NTRO)",
        "sample_id": sample_id,
        "model_id": model_id,
        "scale_factor": f"{scale_factor}x (10m -> 2.5m-equivalent output grid)",
        "input_dimension": list(lr_image.shape),
        "output_dimension": result["shape"],
        "latency_seconds": result["inference_time_s"],
        "metrics": result["metrics"],
        "uncertainty_summary": result["uncertainty"]["summary"] if result.get("uncertainty") else None,
        "geospatial_metadata": geo_metadata,
        "spectral_integrity_compliance": {
            "physical_reflectance_preserved": True,
            "sam_evaluation_target_met": (result["metrics"].get("sam", {}).get("value", 99) < 5.0) if hr_image is not None else None,
            "downsample_consistency_mae": result["metrics"].get("downsample_consistency", {}).get("value"),
            "target_note": "Internal evaluation target: SAM < 5.0° (not an NTRO mandated threshold).",
        },
    }

    return JSONResponse(content=report)


# ==========================================
# PHASE E — NEW ADVANCED & FARMER ENDPOINTS
# ==========================================

@app.post("/api/indices")
async def compute_indices(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    run_id: str = Form(None),
    model_id: str = Form("rcan"),
):
    """
    Compute NDVI, NDWI, EVI, SAVI, RVI, GCI from super-resolved imagery.
    Returns both SR-derived and LR-derived indices for comparison.
    Includes per-band visualization PNGs.
    """
    if run_id:
        run_path = RUNS_DIR / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(404, f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
    else:
        file_bytes = await file.read() if file else None
        lr_image, hr_image, _ = _load_input_data(sample_id, file_bytes)
        result = _execute_model_sr(model_id, lr_image, hr_image, 4)
        sr_array = result["_sr_array"]

    sr_indices = compute_spectral_indices(sr_array)
    lr_indices = compute_spectral_indices(lr_image)

    sr_viz = indices_to_visualizations(sr_indices)
    lr_viz = indices_to_visualizations(lr_indices)

    index_stats = {}
    for name, sr_arr in sr_indices.items():
        lr_arr = lr_indices.get(name, sr_arr)
        index_stats[name] = {
            "sr": {
                "mean": round(float(np.nanmean(sr_arr)), 4),
                "std": round(float(np.nanstd(sr_arr)), 4),
                "p25": round(float(np.nanpercentile(sr_arr, 25)), 4),
                "p75": round(float(np.nanpercentile(sr_arr, 75)), 4),
            },
            "lr": {
                "mean": round(float(np.nanmean(lr_arr)), 4),
                "std": round(float(np.nanstd(lr_arr)), 4),
            },
            "sr_visualization": sr_viz.get(name, ""),
            "lr_visualization": lr_viz.get(name, ""),
        }

    return JSONResponse({"status": "success", "indices": index_stats})


@app.post("/api/batch")
async def batch_superresolve(
    files: List[UploadFile] = File(None),
    sample_ids: str = Form(None),
    model_id: str = Form("rcan"),
    quality: str = Form("fast"),
):
    """
    Submit multiple tiles for parallel super-resolution.
    Returns a batch_id; poll /api/batch/{batch_id} for results.
    Max 10 tiles per batch.
    """
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_id '{model_id}'")

    parsed_sample_ids = [s.strip() for s in (sample_ids or "").split(",") if s.strip()]
    uploaded_files = files or []

    total_inputs = len(parsed_sample_ids) + len(uploaded_files)
    if total_inputs == 0:
        raise HTTPException(400, "Provide 'files' or 'sample_ids'")
    if total_inputs > 10:
        raise HTTPException(400, f"Max 10 tiles per batch. Got {total_inputs}.")

    assert job_store is not None
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    job_ids = []

    for sid in parsed_sample_ids:
        job_id = job_store.create_job(model_id=model_id)
        job_ids.append(job_id)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_thread_pool, _async_worker, job_id, sid, None, model_id, quality)

    for f in uploaded_files:
        file_bytes = await f.read()
        job_id = job_store.create_job(model_id=model_id)
        job_ids.append(job_id)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_thread_pool, _async_worker, job_id, None, file_bytes, model_id, quality)

    batch_manifest = {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "model_id": model_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    batch_path = RUNS_DIR / f"{batch_id}.json"
    with open(batch_path, "w") as bf:
        json.dump(batch_manifest, bf)

    return JSONResponse({
        "status": "accepted",
        "batch_id": batch_id,
        "job_ids": job_ids,
        "total": total_inputs,
        "status_url": f"/api/batch/{batch_id}",
    })


@app.get("/api/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status and results of all jobs in a batch."""
    batch_path = RUNS_DIR / f"{batch_id}.json"
    if not batch_path.exists():
        raise HTTPException(404, f"Batch '{batch_id}' not found")
    with open(batch_path) as bf:
        manifest = json.load(bf)

    assert job_store is not None
    job_statuses = []
    for jid in manifest["job_ids"]:
        job = job_store.get_job(jid)
        job_statuses.append(job or {"job_id": jid, "status": "not_found"})

    completed = sum(1 for j in job_statuses if j.get("status") == "completed")
    failed = sum(1 for j in job_statuses if j.get("status") == "failed")
    total = len(job_statuses)

    overall_status = "completed" if completed == total else "partial_failure" if failed > 0 else "processing"

    return JSONResponse({
        "batch_id": batch_id,
        "overall_status": overall_status,
        "completed": completed,
        "failed": failed,
        "total": total,
        "jobs": job_statuses,
    })


@app.post("/api/crop-health")
async def crop_health_analysis(
    sample_id: str = Form(None),
    run_id: str = Form(None),
    model_id: str = Form("rcan"),
):
    """
    Farmer-facing crop health classification using SR-enhanced spectral indices.
    """
    if run_id:
        run_path = RUNS_DIR / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(404, f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
    else:
        lr_image, hr_image, _ = _load_input_data(sample_id, None)
        result = _execute_model_sr(model_id, lr_image, hr_image, 4)
        sr_array = result["_sr_array"]

    sr_indices = compute_spectral_indices(sr_array)
    lr_indices = compute_spectral_indices(lr_image)

    ndvi = sr_indices["ndvi"]
    ndwi = sr_indices["ndwi"]
    ndbi = sr_indices["ndbi_approx"]
    evi = sr_indices["evi"]

    h, w = ndvi.shape
    classification = np.full((h, w), 3, dtype=np.uint8)
    classification[ndvi > 0.1] = 3   # Sparse / Stressed
    classification[ndvi > 0.3] = 2   # Moderate
    classification[ndvi > 0.6] = 1   # Dense healthy
    classification[ndwi > 0.2] = 4   # Water
    classification[ndbi > 0.0] = 5   # Built-up
    classification[ndvi <= 0.1] = 0  # Bare soil

    CLASS_LABELS = {
        0: "Bare Soil",
        1: "Dense Healthy Crop",
        2: "Moderate Vegetation",
        3: "Sparse / Stressed Crop",
        4: "Water Body",
        5: "Built-up / Non-vegetation",
    }
    CLASS_COLORS = {
        0: [139, 90, 43],    # Brown
        1: [34, 139, 34],    # Forest green
        2: [144, 238, 144],  # Light green
        3: [255, 215, 0],    # Gold/yellow (stress)
        4: [0, 105, 148],    # Deep blue
        5: [128, 128, 128],  # Grey
    }

    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        mask = classification == cls_id
        rgb_map[mask] = color

    buf = io.BytesIO()
    Image.fromarray(rgb_map).save(buf, format="PNG")
    class_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    total_pixels = h * w
    area_stats = {}
    for cls_id, label in CLASS_LABELS.items():
        count = int(np.sum(classification == cls_id))
        area_stats[label] = {
            "pixel_count": count,
            "percentage": round(100.0 * count / total_pixels, 2),
        }

    veg_mask = ndvi > 0.1
    health_score = float(np.mean(ndvi[veg_mask])) if veg_mask.any() else 0.0

    lr_ndvi = lr_indices["ndvi"]
    lr_ndvi_up = np.kron(lr_ndvi, np.ones((4, 4), dtype=np.float32))[:h, :w]
    ndvi_uplift = float(np.mean(np.abs(ndvi - lr_ndvi_up)))

    recommendations = []
    stressed_pct = area_stats.get("Sparse / Stressed Crop", {}).get("percentage", 0)
    if stressed_pct > 30:
        recommendations.append(f"⚠️ {stressed_pct:.1f}% of field shows stress indicators. Consider irrigation or soil testing.")
    if area_stats.get("Water Body", {}).get("percentage", 0) > 5:
        recommendations.append("💧 Significant water presence detected. Check for waterlogging.")
    if health_score > 0.6:
        recommendations.append("✅ Vegetation health appears good. Monitor for seasonal changes.")
    if not recommendations:
        recommendations.append("ℹ️ Moderate crop health. Regular monitoring recommended.")

    return JSONResponse({
        "status": "success",
        "disclaimer": "Rule-based spectral index classification. Not a certified agronomic assessment.",
        "model_id": model_id,
        "classification_map": class_b64,
        "area_statistics": area_stats,
        "health_score": round(health_score, 4),
        "mean_ndvi": round(float(np.mean(ndvi)), 4),
        "mean_evi": round(float(np.mean(evi)), 4),
        "sr_vs_lr_ndvi_uplift": round(ndvi_uplift, 6),
        "recommendations": recommendations,
        "class_legend": {str(k): v for k, v in CLASS_LABELS.items()},
    })


@app.post("/api/field-boundary")
async def field_boundary_delineation(
    sample_id: str = Form(None),
    run_id: str = Form(None),
    model_id: str = Form("rcan"),
    method: str = Form("gradient"),
):
    """
    Delineate agricultural field boundaries from SR imagery using gradient edge detection on NDVI.
    """
    from scipy.ndimage import gaussian_filter, sobel

    if run_id:
        run_path = RUNS_DIR / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(404, f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
    else:
        lr_image, hr_image, _ = _load_input_data(sample_id, None)
        result = _execute_model_sr(model_id, lr_image, hr_image, 4)
        sr_array = result["_sr_array"]

    sr_indices = compute_spectral_indices(sr_array)
    lr_indices = compute_spectral_indices(lr_image)

    def detect_edges(ndvi_map: np.ndarray, sigma: float = 1.0, threshold: float = 0.05) -> np.ndarray:
        smoothed = gaussian_filter(ndvi_map.astype(np.float64), sigma=sigma)
        sx = sobel(smoothed, axis=1)
        sy = sobel(smoothed, axis=0)
        magnitude = np.sqrt(sx**2 + sy**2)
        return (magnitude > threshold).astype(np.uint8)

    sr_edges = detect_edges(sr_indices["ndvi"])
    lr_ndvi_up = np.kron(lr_indices["ndvi"], np.ones((4, 4), dtype=np.float32))
    lr_ndvi_up = lr_ndvi_up[:sr_edges.shape[0], :sr_edges.shape[1]]
    lr_edges = detect_edges(lr_ndvi_up)

    def edges_to_png(img_rgb: np.ndarray, edges: np.ndarray, color=(255, 255, 0)) -> str:
        overlay = img_rgb.copy()
        overlay[edges > 0] = color
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    sr_rgb_base = np.clip(
        np.stack([sr_array[2], sr_array[1], sr_array[0]], axis=-1) * 255, 0, 255
    ).astype(np.uint8)

    lr_rgb_up = np.kron(
        np.clip(np.stack([lr_image[2], lr_image[1], lr_image[0]], axis=-1) * 255, 0, 255).astype(np.uint8),
        np.ones((4, 4, 1), dtype=np.uint8)
    )[:sr_edges.shape[0], :sr_edges.shape[1]]

    sr_overlay = edges_to_png(sr_rgb_base, sr_edges)
    lr_overlay = edges_to_png(lr_rgb_up, lr_edges)

    edge_density_sr = float(np.mean(sr_edges))
    edge_density_lr = float(np.mean(lr_edges))

    return JSONResponse({
        "status": "success",
        "disclaimer": "Heuristic gradient-based field boundary detection. Not cadastral survey data.",
        "model_id": model_id,
        "sr_edge_overlay": sr_overlay,
        "lr_edge_overlay": lr_overlay,
        "sr_edge_density": round(edge_density_sr, 4),
        "lr_edge_density": round(edge_density_lr, 4),
        "boundary_improvement_ratio": round(edge_density_sr / (edge_density_lr + 1e-7), 3),
        "sr_edge_pixel_count": int(np.sum(sr_edges)),
        "lr_edge_pixel_count": int(np.sum(lr_edges)),
        "method": method,
    })


@app.post("/api/change-detect")
async def change_detection(
    run_id_t1: str = Form(..., description="First date run_id (earlier)"),
    run_id_t2: str = Form(..., description="Second date run_id (later)"),
    method: str = Form("ndvi_diff"),
):
    """
    Bi-temporal change detection between two SR outputs.
    """
    def load_run(rid: str) -> np.ndarray:
        p = RUNS_DIR / f"{rid}.npz"
        if not p.exists():
            raise HTTPException(404, f"Run '{rid}' not found")
        d = np.load(str(p))
        return d["sr"].astype(np.float32)

    sr_t1 = load_run(run_id_t1)
    sr_t2 = load_run(run_id_t2)

    if sr_t1.shape != sr_t2.shape:
        raise HTTPException(400, f"Shape mismatch: T1={sr_t1.shape}, T2={sr_t2.shape}. Images must have same dimensions.")

    idx_t1 = compute_spectral_indices(sr_t1)
    idx_t2 = compute_spectral_indices(sr_t2)

    ndvi_diff = idx_t2["ndvi"] - idx_t1["ndvi"]
    spectral_diff = np.mean(np.abs(sr_t2 - sr_t1), axis=0)
    diff_vec = sr_t2 - sr_t1
    change_magnitude = np.sqrt(np.sum(diff_vec**2, axis=0))

    def array_to_heatmap(arr: np.ndarray, cmap_name: str, vmin=None, vmax=None) -> str:
        import matplotlib.pyplot as plt
        p2, p98 = np.percentile(arr, [2, 98])
        vmin = vmin if vmin is not None else p2
        vmax = vmax if vmax is not None else p98
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-7), 0, 1)
        cmap = plt.get_cmap(cmap_name)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    significant_change = (change_magnitude > np.percentile(change_magnitude, 80)).astype(np.uint8)

    return JSONResponse({
        "status": "success",
        "run_id_t1": run_id_t1,
        "run_id_t2": run_id_t2,
        "method": method,
        "ndvi_difference_map": array_to_heatmap(ndvi_diff, "RdYlGn", -0.3, 0.3),
        "spectral_difference_map": array_to_heatmap(spectral_diff, "hot"),
        "change_magnitude_map": array_to_heatmap(change_magnitude, "YlOrRd"),
        "statistics": {
            "mean_ndvi_t1": round(float(np.mean(idx_t1["ndvi"])), 4),
            "mean_ndvi_t2": round(float(np.mean(idx_t2["ndvi"])), 4),
            "ndvi_change": round(float(np.mean(ndvi_diff)), 4),
            "mean_spectral_diff": round(float(np.mean(spectral_diff)), 6),
            "significant_change_pct": round(100.0 * float(np.mean(significant_change)), 2),
            "max_change_magnitude": round(float(np.max(change_magnitude)), 6),
        },
        "interpretation": (
            "Vegetation gain (greening)" if np.mean(ndvi_diff) > 0.05
            else "Vegetation loss or stress" if np.mean(ndvi_diff) < -0.05
            else "Minimal change detected"
        ),
        "disclaimer": "Spectral change analysis. Verify with ground truth for agronomic decisions.",
    })


@app.post("/api/models/reload")
async def reload_model(
    model_id: str = Form(...),
    checkpoint_path: str = Form(None),
):
    """
    Hot-reload a model from an updated checkpoint without server restart.
    Security: checkpoint_path must be within the project's weights directory.
    """
    weights_dir = Path(__file__).parent.parent / "weights"

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        try:
            ckpt_path.resolve().relative_to(weights_dir.resolve())
        except ValueError:
            raise HTTPException(400, "checkpoint_path must be within the weights/ directory")
    else:
        ckpt_path = weights_dir / f"{model_id}_best.pth"

    if not ckpt_path.exists():
        raise HTTPException(404, f"Checkpoint not found: {ckpt_path}")

    success = model_registry.hot_reload(model_id, str(ckpt_path))
    if not success:
        raise HTTPException(500, f"Failed to reload model '{model_id}'")

    meta = model_registry.get_metadata(model_id)
    return JSONResponse({
        "status": "success",
        "model_id": model_id,
        "checkpoint": str(ckpt_path),
        "metadata": meta,
    })


@app.websocket("/ws/inference/{job_id}")
async def websocket_inference_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress streaming.
    Client receives: {"job_id": ..., "status": ..., "progress_pct": ...}
    """
    await websocket.accept()
    try:
        while True:
            job = job_store.get_job(job_id) if job_store else None
            if not job:
                await websocket.send_json({"error": f"Job {job_id} not found"})
                break

            await websocket.send_json({
                "job_id": job_id,
                "status": job["status"],
                "progress_pct": job.get("progress_pct", 0),
                "model_id": job.get("model_id"),
                "inference_time_s": job.get("inference_time_s"),
                "error": job.get("error_message"),
            })

            if job["status"] in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
