"""
BharatSR — Advanced Spectral Analysis Router
Provides spectral indices, farmer crop health classification,
field boundary delineation, and bi-temporal change detection.
"""

import asyncio
import base64
import io
import re
from pathlib import Path

from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.config import Settings
from backend.app.deps import (
    get_settings,
    get_model_registry,
    check_rate_limit,
    check_upload_size,
    read_uploaded_file_capped,
)
from backend.app.schemas import (
    SpectralIndicesResponse,
    CropHealthResponse,
    FieldBoundaryResponse,
    ChangeDetectionResponse,
)
from backend.app.services.inference import ModelRegistry
from backend.app.services.preprocessing import compute_spectral_indices, indices_to_visualizations
from backend.app.services.sr_pipeline import load_input_data, execute_model_sr
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.analysis")
router = APIRouter(tags=["analysis"])


@router.post(
    "/api/indices",
    response_model=SpectralIndicesResponse,
    dependencies=[Depends(check_rate_limit), Depends(check_upload_size)],
)
async def compute_indices(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    run_id: str = Form(None),
    model_id: str = Form("rcan"),
    settings: Settings = Depends(get_settings),
):
    """
    Compute NDVI, NDWI, EVI, SAVI, RVI, NDBI (approx), GCI from super-resolved imagery.
    Returns both SR-derived and LR-derived indices for comparison with PNG heatmaps.
    """
    runs_dir = Path(settings.runs_dir)
    sample_tiles_dir = Path(settings.sample_tiles_dir)

    def _sync_calc():
        if run_id:
            run_path = runs_dir / f"{run_id}.npz"
            if not run_path.exists():
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
            data = np.load(str(run_path))
            sr_array = data["sr"].astype(np.float32)
            lr_image = data["lr"].astype(np.float32)
        else:
            lr_image, hr_image, _ = load_input_data(sample_id, file_bytes, sample_tiles_dir)
            result = execute_model_sr(model_id, lr_image, hr_image, scale_factor=4)
            sr_array = result.array

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
        return {"status": "success", "indices": index_stats}

    file_bytes = await read_uploaded_file_capped(file, settings.max_image_bytes)
    res = await asyncio.to_thread(_sync_calc)
    return JSONResponse(content=res)


@router.post(
    "/api/crop-health",
    response_model=CropHealthResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def crop_health_analysis(
    sample_id: str = Form(None),
    run_id: str = Form(None),
    model_id: str = Form("rcan"),
    settings: Settings = Depends(get_settings),
):
    """
    Farmer-facing crop health classification using SR-enhanced spectral indices.
    Classifies pixels into 6 agronomic categories and provides actionable recommendations.
    """
    runs_dir = Path(settings.runs_dir)
    sample_tiles_dir = Path(settings.sample_tiles_dir)

    def _sync_calc():
        if run_id:
            run_path = runs_dir / f"{run_id}.npz"
            if not run_path.exists():
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
            data = np.load(str(run_path))
            sr_array = data["sr"].astype(np.float32)
            lr_image = data["lr"].astype(np.float32)
        else:
            lr_image, hr_image, _ = load_input_data(sample_id, None, sample_tiles_dir)
            result = execute_model_sr(model_id, lr_image, hr_image, scale_factor=4)
            sr_array = result.array

        sr_indices = compute_spectral_indices(sr_array)
        lr_indices = compute_spectral_indices(lr_image)

        ndvi = sr_indices["ndvi"]
        ndwi = sr_indices["ndwi"]
        ndbi = sr_indices["ndbi_approx"]
        evi = sr_indices["evi"]

        h, w = ndvi.shape
        classification = np.full((h, w), 3, dtype=np.uint8)
        classification[ndvi > 0.1] = 3
        classification[ndvi > 0.3] = 2
        classification[ndvi > 0.6] = 1
        classification[ndwi > 0.2] = 4
        classification[ndbi > 0.0] = 5
        classification[ndvi <= 0.1] = 0

        CLASS_LABELS = {
            0: "Bare Soil",
            1: "Dense Healthy Crop",
            2: "Moderate Vegetation",
            3: "Sparse / Stressed Crop",
            4: "Water Body",
            5: "Built-up / Non-vegetation",
        }
        CLASS_COLORS = {
            0: [139, 90, 43],
            1: [34, 139, 34],
            2: [144, 238, 144],
            3: [255, 215, 0],
            4: [0, 105, 148],
            5: [128, 128, 128],
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
            recommendations.append(
                f"⚠️ {stressed_pct:.1f}% of field shows stress indicators. Consider irrigation or soil testing."
            )
        if area_stats.get("Water Body", {}).get("percentage", 0) > 5:
            recommendations.append("💧 Significant water presence detected. Check for waterlogging.")
        if health_score > 0.6:
            recommendations.append("✅ Vegetation health appears good. Monitor for seasonal changes.")
        if not recommendations:
            recommendations.append("ℹ️ Moderate crop health. Regular monitoring recommended.")

        return {
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
        }

    res = await asyncio.to_thread(_sync_calc)
    return JSONResponse(content=res)


@router.post(
    "/api/field-boundary",
    response_model=FieldBoundaryResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def field_boundary_delineation(
    sample_id: str = Form(None),
    run_id: str = Form(None),
    model_id: str = Form("rcan"),
    method: str = Form("gradient"),
    settings: Settings = Depends(get_settings),
):
    """
    Delineate agricultural field boundaries from SR imagery using gradient edge detection on NDVI.
    """
    runs_dir = Path(settings.runs_dir)
    sample_tiles_dir = Path(settings.sample_tiles_dir)

    def _sync_calc():
        if run_id:
            run_path = runs_dir / f"{run_id}.npz"
            if not run_path.exists():
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
            data = np.load(str(run_path))
            sr_array = data["sr"].astype(np.float32)
            lr_image = data["lr"].astype(np.float32)
        else:
            lr_image, hr_image, _ = load_input_data(sample_id, None, sample_tiles_dir)
            result = execute_model_sr(model_id, lr_image, hr_image, scale_factor=4)
            sr_array = result.array

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
            np.ones((4, 4, 1), dtype=np.uint8),
        )[:sr_edges.shape[0], :sr_edges.shape[1]]

        sr_overlay = edges_to_png(sr_rgb_base, sr_edges)
        lr_overlay = edges_to_png(lr_rgb_up, lr_edges)

        edge_density_sr = float(np.mean(sr_edges))
        edge_density_lr = float(np.mean(lr_edges))

        return {
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
        }

    res = await asyncio.to_thread(_sync_calc)
    return JSONResponse(content=res)


@router.post(
    "/api/change-detect",
    response_model=ChangeDetectionResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def change_detection(
    run_id_t1: str = Form(..., description="First date run_id (earlier)"),
    run_id_t2: str = Form(..., description="Second date run_id (later)"),
    method: str = Form("ndvi_diff"),
    settings: Settings = Depends(get_settings),
):
    """
    Bi-temporal change detection between two SR outputs.
    """
    runs_dir = Path(settings.runs_dir)

    sample_tiles_dir = Path(settings.sample_tiles_dir)

    def _sync_calc():
        def load_sr_data(target_id: str) -> np.ndarray:
            clean_id = Path(target_id).name
            if clean_id != target_id or not re.match(r"^[a-zA-Z0-9_-]+$", target_id):
                raise HTTPException(
                    status_code=404,
                    detail=f"Target '{target_id}' not found in active runs or sample catalog.",
                )
            p = (runs_dir / f"{target_id}.npz").resolve()
            try:
                p.relative_to(runs_dir.resolve())
                if p.exists():
                    d = np.load(str(p))
                    if "sr" in d:
                        return d["sr"].astype(np.float32)
            except Exception as e:
                logger.warning(f"Error loading run {target_id}: {e}")

            # Fallback: check if target_id is a pre-loaded sample_id
            sample_path = (sample_tiles_dir / f"{target_id}.npz").resolve()
            try:
                sample_path.relative_to(sample_tiles_dir.resolve())
                if sample_path.exists():
                    lr_img, hr_img, _ = load_input_data(target_id, None, sample_tiles_dir)
                    res = execute_model_sr("rcan", lr_img, hr_img, scale_factor=4)
                    return res.array.astype(np.float32)
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Error loading fallback sample {target_id}: {e}")

            raise HTTPException(
                status_code=404,
                detail=f"Target '{target_id}' not found in active runs or sample catalog.",
            )


        sr_t1 = load_sr_data(run_id_t1)
        sr_t2 = load_sr_data(run_id_t2)

        # Ensure spatial dimension compatibility via center crop or overlap
        if sr_t1.shape != sr_t2.shape:
            min_h = min(sr_t1.shape[1], sr_t2.shape[1])
            min_w = min(sr_t1.shape[2], sr_t2.shape[2])
            sr_t1 = sr_t1[:, :min_h, :min_w]
            sr_t2 = sr_t2[:, :min_h, :min_w]

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

        return {
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
                "Vegetation gain (greening)"
                if np.mean(ndvi_diff) > 0.05
                else "Vegetation loss or stress"
                if np.mean(ndvi_diff) < -0.05
                else "Minimal change detected"
            ),
            "disclaimer": "Spectral change analysis. Verify with ground truth for agronomic decisions.",
        }

    res = await asyncio.to_thread(_sync_calc)
    return JSONResponse(content=res)
