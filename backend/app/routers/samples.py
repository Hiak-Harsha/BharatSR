"""
BharatSR — Samples Router
Provides discovery and previews of pre-loaded Sentinel-2 sample tiles.
"""

from pathlib import Path
import json
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


@router.get("/api/samples", response_model=SamplesListResponse)
def list_samples(settings: Settings = Depends(get_settings)):
    """
    List pre-loaded sample tiles with honest geographic and provenance metadata.
    Never fabricates coordinates or real city names for synthetic procedural samples.
    """
    sample_dir = Path(settings.sample_tiles_dir)
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
                logger.error(f"Error loading sample tile {f}: {e}")

    # Sort so genuine georeferenced satellite scenes appear first, followed by verification patterns
    samples.sort(key=lambda s: (not s.has_geo, s.id))
    return SamplesListResponse(samples=samples)


@router.get("/api/samples/{sample_id}/preview")
def sample_preview(
    sample_id: str,
    crop: int = 32,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Small, fast live-inference preview for a sample tile: LR / Bicubic / BharatSR
    (/ Ground Truth if available), each as a base64 RGB PNG.
    """
    sample_tiles_dir = Path(settings.sample_tiles_dir)
    sample_path = sample_tiles_dir / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")

    lr_image, hr_image = load_sample_tile(str(sample_path))
    c, h, w = lr_image.shape
    crop = max(8, min(crop, h, w))
    y0 = max(0, (h - crop) // 2)
    x0 = max(0, (w - crop) // 2)
    lr_crop = lr_image[:, y0:y0 + crop, x0:x0 + crop]

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
        crop_hr = crop * scale_factor
        y0_hr = max(0, (h_hr - crop_hr) // 2)
        x0_hr = max(0, (w_hr - crop_hr) // 2)
        hr_crop = hr_image[:, y0_hr:y0_hr + crop_hr, x0_hr:x0_hr + crop_hr]
        views["ground_truth"] = generate_multi_spectral_views(hr_crop).get("rgb", "")

    return {"sample_id": sample_id, "crop_size": crop, "views": views}
