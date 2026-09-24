"""
BharatSR — Core Inference Router
Handles super-resolution, comparative benchmarking, pixel profiling, and downstream masks.
"""

import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.config import Settings
from backend.app.deps import (
    get_settings,
    get_model_registry,
    get_job_store,
    get_thread_pool,
    verify_api_key,
    check_rate_limit,
    check_upload_size,
    read_uploaded_file_capped,
)
from backend.app.schemas import (
    SuperResolveResponse,
    AsyncJobSubmitResponse,
    CompareResponse,
    PixelProfileResponse,
    DownstreamMasksResponse,
    BandProfile,
    DownstreamTaskItem,
    DownstreamTaskMetrics,
)
from backend.app.services.inference import ModelRegistry, run_bicubic_baseline
from backend.app.services.job_store import JobStore
from backend.app.services.preprocessing import (
    generate_multi_spectral_views,
    load_sample_tile,
    BAND_INDEX,
)
from backend.app.services.sr_pipeline import (
    load_input_data,
    execute_model_sr,
    async_worker,
    mask_to_png_bytes,
)
from backend.app.models_ml.uncertainty import generate_uncertainty_heatmap
from evaluation.downstream_task import (
    segment_micro_canopy_rule_based,
    segment_built_up_rule_based,
    compute_segmentation_metrics,
)
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.inference")
router = APIRouter(tags=["inference"])

ALLOWED_MODELS = ("bicubic", "srcnn", "rcan", "swinir", "hat", "diffusion", "ensemble")


def _run_superresolve_sync(
    model_id: str,
    lr_image: np.ndarray,
    hr_image: Optional[np.ndarray],
    geo_metadata: Optional[Dict[str, Any]],
    scale_factor: int,
    quality: str,
    runs_dir: Path,
) -> Dict[str, Any]:
    """Synchronous CPU worker for superresolve."""
    result = execute_model_sr(model_id, lr_image, hr_image, scale_factor, quality=quality)
    sr_array = result.array

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    run_path = runs_dir / f"{run_id}.npz"
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
        logger.warning(f"Could not cache run {run_id}: {e}")

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
        "quality": result.response_dict["quality"],
        "inference_time_s": result.response_dict["inference_time_s"],
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
            "shape": result.response_dict["shape"],
            "image": result.response_dict["image"],
            "views": result.response_dict["views"],
        },
        "metrics": result.response_dict["metrics"],
    }

    if result.response_dict.get("error_map") is not None:
        response["error_map"] = result.response_dict["error_map"]

    if result.response_dict.get("uncertainty") is not None:
        response["uncertainty"] = result.response_dict["uncertainty"]

    if hr_image is not None:
        gt_views = generate_multi_spectral_views(hr_image)
        response["ground_truth"] = {
            "shape": list(hr_image.shape),
            "image": gt_views.get("rgb", ""),
            "views": gt_views,
        }

    if geo_metadata:
        response["geospatial_metadata"] = geo_metadata

    return response


@router.post(
    "/api/superresolve",
    response_model=SuperResolveResponse,
    dependencies=[Depends(check_rate_limit), Depends(check_upload_size), Depends(verify_api_key)],
)
async def superresolve(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    model_id: str = Form("rcan"),
    quality: str = Form("fast"),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Run super-resolution on an uploaded GeoTIFF or pre-loaded sample tile.
    Returns 4x SR result as base64 PNG, multi-spectral views, physical metrics,
    and predicted uncertainty map.
    """
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id '{model_id}'. Allowed: {', '.join(ALLOWED_MODELS)}",
        )
    if quality not in ("fast", "high"):
        raise HTTPException(status_code=400, detail="Invalid quality. Allowed: fast, high")

    file_bytes = await read_uploaded_file_capped(file, settings.max_image_bytes)
    sample_tiles_dir = Path(settings.sample_tiles_dir)
    lr_image, hr_image, geo_metadata = load_input_data(sample_id, file_bytes, sample_tiles_dir)

    model_meta = registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)
    runs_dir = Path(settings.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Async correctness: offload CPU PyTorch & metric calculations to threadpool
    response = await asyncio.to_thread(
        _run_superresolve_sync,
        model_id,
        lr_image,
        hr_image,
        geo_metadata,
        scale_factor,
        quality,
        runs_dir,
    )
    return JSONResponse(content=response)


@router.post(
    "/api/superresolve/async",
    response_model=AsyncJobSubmitResponse,
    dependencies=[Depends(check_rate_limit), Depends(check_upload_size), Depends(verify_api_key)],
)
async def superresolve_async(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    model_id: str = Form("rcan"),
    quality: str = Form("fast"),
    settings: Settings = Depends(get_settings),
    store: JobStore = Depends(get_job_store),
    pool=Depends(get_thread_pool),
):
    """
    Submit super-resolution as an asynchronous background task.
    Returns job_id and status checking URL.
    """
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id '{model_id}'. Allowed: {', '.join(ALLOWED_MODELS)}",
        )
    if quality not in ("fast", "high"):
        raise HTTPException(status_code=400, detail="Invalid quality. Allowed: fast, high")

    file_bytes = await read_uploaded_file_capped(file, settings.max_image_bytes)
    if not sample_id and not file_bytes:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'sample_id'")

    job_id = store.create_job(model_id=model_id)
    runs_dir = Path(settings.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sample_tiles_dir = Path(settings.sample_tiles_dir)

    loop = asyncio.get_event_loop()
    if pool is not None:
        loop.run_in_executor(
            pool,
            async_worker,
            job_id,
            sample_id,
            file_bytes,
            model_id,
            quality,
            store,
            runs_dir,
            sample_tiles_dir,
        )
    else:
        asyncio.create_task(
            asyncio.to_thread(
                async_worker,
                job_id,
                sample_id,
                file_bytes,
                model_id,
                quality,
                store,
                runs_dir,
                sample_tiles_dir,
            )
        )

    return AsyncJobSubmitResponse(
        status="accepted",
        job_id=job_id,
        model_id=model_id,
        status_url=f"/api/jobs/{job_id}",
    )


def _run_compare_sync(
    lr_image: np.ndarray,
    hr_image: Optional[np.ndarray],
    geo_metadata: Optional[Dict[str, Any]],
    registry: ModelRegistry,
    runs_dir: Path,
) -> Dict[str, Any]:
    """Synchronous CPU worker for multi-model comparison."""
    input_views = generate_multi_spectral_views(lr_image)

    model_ids = ["bicubic"]
    for m in ALLOWED_MODELS:
        if m != "bicubic" and registry.get_model(m) is not None:
            model_ids.append(m)

    results = {}
    sr_arrays = {}
    for m_id in model_ids:
        res = execute_model_sr(m_id, lr_image, hr_image, scale_factor=4)
        results[m_id] = res.response_dict
        sr_arrays[m_id] = res.array

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

    latency_row = {
        "metric": "Inference Latency",
        "unit": "s",
        "higher_is_better": False,
        "best_model": "bicubic",
    }
    for m_id in model_ids:
        latency_row[m_id] = results[m_id]["inference_time_s"]
    comparison_table.append(latency_row)

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    best_model = next((m for m in ["rcan", "srcnn", "bicubic"] if m in sr_arrays), None)
    if best_model:
        best_sr = sr_arrays[best_model]
        try:
            np.savez_compressed(
                str(runs_dir / f"{run_id}.npz"),
                sr=best_sr,
                lr=lr_image,
                hr=hr_image if hr_image is not None else np.zeros((0,), dtype=np.float32),
                has_hr=hr_image is not None,
                geo_json=json.dumps(geo_metadata) if geo_metadata else "",
                model_id=best_model,
                scale_factor=4,
            )
        except Exception as e:
            logger.warning(f"Could not cache compare run: {e}")

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

    return response


@router.post(
    "/api/compare",
    response_model=CompareResponse,
    dependencies=[Depends(check_rate_limit), Depends(check_upload_size), Depends(verify_api_key)],
)
async def compare(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Rigorous multi-model benchmark: Bicubic Baseline vs SRCNN vs RCAN side-by-side.
    """
    file_bytes = await read_uploaded_file_capped(file, settings.max_image_bytes)
    sample_tiles_dir = Path(settings.sample_tiles_dir)
    lr_image, hr_image, geo_metadata = load_input_data(sample_id, file_bytes, sample_tiles_dir)
    runs_dir = Path(settings.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    response = await asyncio.to_thread(
        _run_compare_sync,
        lr_image,
        hr_image,
        geo_metadata,
        registry,
        runs_dir,
    )
    return JSONResponse(content=response)


def _run_pixel_profile_sync(
    sample_id: Optional[str],
    run_id: Optional[str],
    x: int,
    y: int,
    model_id: str,
    runs_dir: Path,
    sample_tiles_dir: Path,
    registry: ModelRegistry,
) -> PixelProfileResponse:
    """Synchronous CPU worker for pixel profile calculation."""
    if run_id:
        run_path = runs_dir / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
        hr_image = data["hr"].astype(np.float32) if data.get("has_hr", False) and data["hr"].size > 0 else None
        scale_factor = int(data.get("scale_factor", 4))
    else:
        sample_path = sample_tiles_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample tile not found")

        lr_image, hr_image = load_sample_tile(str(sample_path))
        model_meta = registry.get_metadata(model_id) or {}
        scale_factor = model_meta.get("scale_factor", 4)

        result = execute_model_sr(model_id, lr_image, hr_image, scale_factor)
        sr_array = result.array

    c, h_hr, w_hr = sr_array.shape
    x = max(0, min(int(x), w_hr - 1))
    y = max(0, min(int(y), h_hr - 1))

    x_lr = max(0, min(x // scale_factor, lr_image.shape[2] - 1))
    y_lr = max(0, min(y // scale_factor, lr_image.shape[1] - 1))

    bicubic_arr, _ = run_bicubic_baseline(lr_image, scale_factor=scale_factor)

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


@router.post(
    "/api/pixel-profile",
    response_model=PixelProfileResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def pixel_profile(
    sample_id: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    x: int = Form(128),
    y: int = Form(128),
    model_id: str = Form("rcan"),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Extract multi-band spectral reflectance profile at coordinate (x, y).
    """
    if not sample_id and not run_id:
        raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'run_id'")

    runs_dir = Path(settings.runs_dir)
    sample_tiles_dir = Path(settings.sample_tiles_dir)

    return await asyncio.to_thread(
        _run_pixel_profile_sync,
        sample_id,
        run_id,
        x,
        y,
        model_id,
        runs_dir,
        sample_tiles_dir,
        registry,
    )


def _run_downstream_masks_sync(
    sample_id: Optional[str],
    run_id: Optional[str],
    model_id: str,
    runs_dir: Path,
    sample_tiles_dir: Path,
    registry: ModelRegistry,
) -> DownstreamMasksResponse:
    """Synchronous CPU worker for downstream tasks."""
    if run_id:
        run_path = runs_dir / f"{run_id}.npz"
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        data = np.load(str(run_path))
        sr_array = data["sr"].astype(np.float32)
        lr_image = data["lr"].astype(np.float32)
        hr_image = data["hr"].astype(np.float32) if data.get("has_hr", False) and data["hr"].size > 0 else None
        scale_factor = int(data.get("scale_factor", 4))
    else:
        sample_path = sample_tiles_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample tile not found")

        lr_image, hr_image = load_sample_tile(str(sample_path))
        model_meta = registry.get_metadata(model_id) or {}
        scale_factor = model_meta.get("scale_factor", 4)

        result = execute_model_sr(model_id, lr_image, hr_image, scale_factor)
        sr_array = result.array

    bicubic_sr, _ = run_bicubic_baseline(lr_image, scale_factor=scale_factor)

    tasks_config = {
        "canopy_segmentation": {
            "name": "Micro-Canopy Vegetation Segmentation (Rule-Based)",
            "description": "Rule-based spectral interpretation: Delineates vegetation canopy using NDVI > 0.35",
            "fn": segment_micro_canopy_rule_based,
            "color": (34, 197, 94),
        },
        "built_up_infrastructure": {
            "name": "Built-up Infrastructure Extraction (Rule-Based)",
            "description": "Rule-based spectral interpretation: Extracts candidate built-up areas using albedo & contrast",
            "fn": segment_built_up_rule_based,
            "color": (249, 115, 22),
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


@router.post(
    "/api/downstream-masks",
    response_model=DownstreamMasksResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def downstream_masks(
    sample_id: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    model_id: str = Form("rcan"),
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Computes binary segmentation masks and real-time IoU/F1 metrics for
    downstream analytical tasks (Micro-Canopy Vegetation & Built-Up Infrastructure Extraction).
    """
    if not sample_id and not run_id:
        raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'run_id'")

    runs_dir = Path(settings.runs_dir)
    sample_tiles_dir = Path(settings.sample_tiles_dir)

    return await asyncio.to_thread(
        _run_downstream_masks_sync,
        sample_id,
        run_id,
        model_id,
        runs_dir,
        sample_tiles_dir,
        registry,
    )
