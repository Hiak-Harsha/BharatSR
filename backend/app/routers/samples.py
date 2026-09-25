"""
BharatSR — Samples Router
Provides discovery and previews of pre-loaded Sentinel-2 sample tiles.
Uses in-memory caching and non-blocking asyncio.to_thread offloading.
"""

import asyncio
from pathlib import Path
import json
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from backend.app.schemas import SamplesListResponse, SampleInfo
from backend.app.config import Settings
from backend.app.deps import get_settings, get_model_registry
from backend.app.services.inference import ModelRegistry, run_bicubic_baseline, run_inference
from backend.app.services.preprocessing import generate_multi_spectral_views, load_sample_tile
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.samples")
router = APIRouter(tags=["samples"])

_cached_samples: Optional[List[SampleInfo]] = None
_cached_samples_mtime: float = 0.0


def _load_samples_sync(sample_dir_path: str) -> List[SampleInfo]:
    """Synchronous CPU worker with in-memory caching based on directory mtime."""
    global _cached_samples, _cached_samples_mtime
    sample_dir = Path(sample_dir_path)
    if not sample_dir.exists():
        return []

    npz_files = list(sample_dir.glob("sample_*.npz"))
    current_mtime = max((f.stat().st_mtime for f in npz_files), default=0.0)

    if _cached_samples is not None and current_mtime <= _cached_samples_mtime:
        return _cached_samples

    samples = []
    for f in sorted(npz_files):
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
                thumbnail=views.get("composite", views.get("rgb", "")),
                views=views,
            ))
        except Exception as e:
            logger.error(f"Error loading sample tile {f}: {e}")

    # Sort so genuine georeferenced satellite scenes appear first
    samples.sort(key=lambda s: (not s.has_geo, s.id))
    _cached_samples = samples
    _cached_samples_mtime = current_mtime
    return samples


@router.get("/api/samples", response_model=SamplesListResponse)
async def list_samples(settings: Settings = Depends(get_settings)):
    """
    List pre-loaded sample tiles with honest geographic and provenance metadata.
    Never blocks the main asyncio event loop.
    """
    samples = await asyncio.to_thread(_load_samples_sync, settings.sample_tiles_dir)
    return SamplesListResponse(samples=samples)


@router.get("/api/samples/{sample_id}/preview")
async def sample_preview(
    sample_id: str,
    crop: int = 32,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Small, fast live-inference preview for a sample tile: LR / Bicubic / BharatSR
    (/ Ground Truth if available), each as a base64 RGB PNG.
    """
    def _sync_preview():
        sample_tiles_dir = Path(settings.sample_tiles_dir)
        sample_path = sample_tiles_dir / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")

        lr_image, hr_image = load_sample_tile(str(sample_path))
        c, h, w = lr_image.shape
        crop_size = max(8, min(crop, h, w))
        y0 = max(0, (h - crop_size) // 2)
        x0 = max(0, (w - crop_size) // 2)
        lr_crop = lr_image[:, y0:y0 + crop_size, x0:x0 + crop_size]

        model = registry.get_model("rcan")
        model_meta = registry.get_metadata("rcan") or {}
        scale_factor = model_meta.get("scale_factor", 4)

        bicubic_sr, _ = run_bicubic_baseline(lr_crop, scale_factor=scale_factor)
        views = {
            "lr": generate_multi_spectral_views(lr_crop).get("rgb", ""),
            "bicubic": generate_multi_spectral_views(bicubic_sr).get("rgb", ""),
        }

        if model is not None:
            sr_crop, _, _ = run_inference(model, lr_crop, scale_factor, registry.device)
            views["sr"] = generate_multi_spectral_views(sr_crop).get("rgb", "")

        if hr_image is not None:
            h_hr, w_hr = hr_image.shape[1], hr_image.shape[2]
            crop_hr = crop_size * scale_factor
            y0_hr = max(0, (h_hr - crop_hr) // 2)
            x0_hr = max(0, (w_hr - crop_hr) // 2)
            hr_crop = hr_image[:, y0_hr:y0_hr + crop_hr, x0_hr:x0_hr + crop_hr]
            views["ground_truth"] = generate_multi_spectral_views(hr_crop).get("rgb", "")

        return {"sample_id": sample_id, "crop_size": crop_size, "views": views}

    return await asyncio.to_thread(_sync_preview)
