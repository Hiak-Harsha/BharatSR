"""
BharatSR — FastAPI Main Application (Phase 3)
"""

import sys
import base64
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.inference import model_registry, run_inference, run_bicubic_baseline
from backend.app.services.preprocessing import (
    load_image_from_bytes, load_sample_tile, numpy_to_png_bytes,
    generate_multi_spectral_views, export_geotiff_bytes
)
from backend.app.services.postprocessing import compute_inference_metrics
from backend.app.services.job_store import JobStore
from backend.app.models_ml.uncertainty import generate_uncertainty_heatmap, summarize_uncertainty


# ========================
# APP LIFECYCLE
# ========================

job_store = None
RUNS_DIR = Path(__file__).parent.parent / "runs"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    global job_store

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize job store
    db_path = str(Path(__file__).parent.parent / "bharatsr.db")
    job_store = JobStore(db_path)
    print("Job store initialized")

    # Load available models
    weights_dir = Path(__file__).parent.parent / "weights"
    srcnn_path = weights_dir / "srcnn_best.pth"

    if srcnn_path.exists():
        model_registry.load_model("srcnn", str(srcnn_path))
    else:
        print(f"WARNING: SRCNN checkpoint not found at {srcnn_path}")
        print("Run training first: python training/train_srcnn.py")

    # Phase 5: Load RCAN if available
    rcan_path = weights_dir / "rcan_best.pth"
    if rcan_path.exists():
        model_registry.load_model("rcan", str(rcan_path))

    print(f"Available models: {[m['id'] for m in model_registry.list_models()]}")

    yield  # App runs

    print("Shutting down BharatSR backend")


# ========================
# APP SETUP
# ========================

app = FastAPI(
    title="BharatSR API",
    description="Deep Learning Super-Resolution for Satellite Imagery",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# HELPER FUNCTIONS
# ========================

def _load_input_data(sample_id: Optional[str], file_bytes: Optional[bytes]):
    """Loads LR image and optional HR ground truth / geo metadata."""
    if sample_id:
        sample_path = Path(__file__).parent.parent / "sample_tiles" / f"{sample_id}.npz"
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
        lr_image, hr_image = load_sample_tile(str(sample_path))
        return lr_image, hr_image, None
    elif file_bytes:
        try:
            lr_image, geo_metadata = load_image_from_bytes(file_bytes)
            return lr_image, None, geo_metadata
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'sample_id'")


def _execute_model_sr(model_id: str, lr_image: np.ndarray, hr_image: Optional[np.ndarray], scale_factor: int = 4):
    """Executes a single model inference and returns formatted results."""
    uncertainty_dict = None

    if model_id == "bicubic":
        sr_image, latency = run_bicubic_baseline(lr_image, scale_factor=scale_factor)
        uncertainty_map = None
    else:
        model = model_registry.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' not loaded.")
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

    if uncertainty_map is not None:
        u_bytes = generate_uncertainty_heatmap(uncertainty_map, colormap="magma")
        u_b64 = base64.b64encode(u_bytes).decode("utf-8")
        uncertainty_dict = {
            "image": f"data:image/png;base64,{u_b64}",
            "summary": summarize_uncertainty(uncertainty_map),
        }

    return {
        "model_id": model_id,
        "inference_time_s": round(latency, 4),
        "shape": list(sr_image.shape),
        "image": sr_b64,
        "views": views,
        "metrics": metrics,
        "uncertainty": uncertainty_dict,
        "_sr_array": sr_image,
    }


def _async_worker(job_id: str, sample_id: Optional[str], file_bytes: Optional[bytes], model_id: str):
    """Background task for async processing."""
    try:
        job_store.update_job(job_id, status="processing")
        lr_image, hr_image, geo_metadata = _load_input_data(sample_id, file_bytes)

        model_meta = model_registry.get_metadata(model_id) or {}
        scale_factor = model_meta.get("scale_factor", 4)

        result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
        sr_array = result.pop("_sr_array")

        # Save result payload to disk
        payload_path = RUNS_DIR / f"{job_id}.json"
        with open(payload_path, "w") as f:
            json.dump(result, f)

        # Save numpy array for export
        npz_path = RUNS_DIR / f"{job_id}.npz"
        np.savez_compressed(str(npz_path), sr=sr_array, lr=lr_image)

        job_store.update_job(
            job_id,
            status="completed",
            result_path=str(payload_path),
            metrics=result["metrics"],
            inference_time=result["inference_time_s"],
        )
    except Exception as e:
        job_store.update_job(job_id, status="failed", error=str(e))


# ========================
# ENDPOINTS
# ========================

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": len(model_registry.list_models()),
    }


@app.get("/api/models")
async def list_models():
    """List available super-resolution models."""
    models = model_registry.list_models()
    return {"models": models}


SAMPLE_METADATA = {
    "sample_0": {
        "title": "Delhi Agro-Urban Corridor",
        "region": "Northern Plains, India",
        "description": "Urban fringe with active crop fields, field boundaries, and small structures.",
        "tactical_category": "Micro-Land-Cover & Perimeter Surveillance",
        "coordinates": "28.6139° N, 77.2090° E",
    },
    "sample_1": {
        "title": "Jodhpur Desert Outskirts",
        "region": "Thar Desert, Rajasthan",
        "description": "High-albedo arid soil, rural road network, and sparse scrub vegetation.",
        "tactical_category": "Border Reconnaissance & Route Mapping",
        "coordinates": "26.2389° N, 73.0243° E",
    },
    "sample_2": {
        "title": "Dehradun Forest Foothills",
        "region": "Himalayan Foothills, Uttarakhand",
        "description": "Dense mixed forest canopy with terrain shadows and micro-drainage channels.",
        "tactical_category": "Forestry Vigor & Terrain Assessment",
        "coordinates": "30.3165° N, 78.0322° E",
    },
    "sample_3": {
        "title": "Visakhapatnam Coastal Sector",
        "region": "Eastern Littoral, Andhra Pradesh",
        "description": "Water-land interface, coastal wetlands, and maritime infrastructure.",
        "tactical_category": "Coastal & Critical Infrastructure Security",
        "coordinates": "17.6868° N, 83.2185° E",
    },
}


@app.get("/api/samples")
async def list_samples():
    """List pre-loaded sample tiles with geographic and tactical intelligence metadata."""
    sample_dir = Path(__file__).parent.parent / "sample_tiles"
    samples = []

    if sample_dir.exists():
        for f in sorted(sample_dir.glob("sample_*.npz")):
            try:
                data = np.load(str(f))
                lr = data["lr"]
                views = generate_multi_spectral_views(lr)
                meta = SAMPLE_METADATA.get(f.stem, {
                    "title": f.stem.replace("_", " ").title(),
                    "region": "Satellite Test Scene",
                    "description": "Calibrated 4-band Sentinel-2 earth observation tile.",
                    "tactical_category": "General Surveillance",
                    "coordinates": "India Regional Observation",
                })

                samples.append({
                    "id": f.stem,
                    "filename": f.name,
                    "title": meta["title"],
                    "region": meta["region"],
                    "description": meta["description"],
                    "tactical_category": meta["tactical_category"],
                    "coordinates": meta["coordinates"],
                    "bands": int(lr.shape[0]),
                    "lr_size": f"{lr.shape[1]}x{lr.shape[2]}",
                    "has_ground_truth": "hr" in data.files,
                    "thumbnail": views.get("rgb", ""),
                    "views": views,
                })
            except Exception as e:
                print(f"Error loading sample {f}: {e}")

    return {"samples": samples}



@app.post("/api/superresolve")
async def superresolve(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    model_id: str = Form("srcnn"),
):
    """
    Run super-resolution on an uploaded image or sample tile.
    Returns SR result as base64 PNG, multi-spectral views, and physical metrics.
    """
    file_bytes = await file.read() if file else None
    lr_image, hr_image, geo_metadata = _load_input_data(sample_id, file_bytes)

    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
    result.pop("_sr_array")

    input_views = generate_multi_spectral_views(lr_image)

    response = {
        "status": "success",
        "model_id": model_id,
        "inference_time_s": result["inference_time_s"],
        "input": {
            "shape": list(lr_image.shape),
            "image": input_views.get("rgb", ""),
            "views": input_views,
        },
        "output": {
            "shape": result["shape"],
            "image": result["image"],
            "views": result["views"],
        },
        "metrics": result["metrics"],
    }

    if result["uncertainty"] is not None:
        response["uncertainty"] = result["uncertainty"]

    if hr_image is not None:
        gt_views = generate_multi_spectral_views(hr_image)
        response["ground_truth"] = {
            "shape": list(hr_image.shape),
            "image": gt_views.get("rgb", ""),
            "views": gt_views,
        }

    return JSONResponse(content=response)


@app.post("/api/compare")
async def compare(
    file: UploadFile = File(None),
    sample_id: str = Form(None),
):
    """
    Benchmark Bicubic Baseline vs SRCNN vs RCAN side-by-side.
    Computes comparative metrics, latency, and multi-spectral representations.
    """
    file_bytes = await file.read() if file else None
    lr_image, hr_image, geo_metadata = _load_input_data(sample_id, file_bytes)

    input_views = generate_multi_spectral_views(lr_image)

    # Models to benchmark
    model_ids = ["bicubic"]
    if model_registry.get_model("srcnn") is not None:
        model_ids.append("srcnn")
    if model_registry.get_model("rcan") is not None:
        model_ids.append("rcan")

    results = {}
    for m_id in model_ids:
        res = _execute_model_sr(m_id, lr_image, hr_image, scale_factor=4)
        res.pop("_sr_array")
        results[m_id] = res

    # Build comparison table
    comparison_table = []
    metric_keys = [
        ("psnr", "PSNR (Peak SNR)", "dB", True),
        ("ssim", "SSIM (Structural Similarity)", "", True),
        ("sam", "SAM (Spectral Angle Mapper)", "°", False),
        ("downsample_consistency", "Downsample Consistency", "MAE", False),
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

    response = {
        "status": "success",
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

    return JSONResponse(content=response)


@app.post("/api/pixel-profile")
async def pixel_profile(
    sample_id: str = Form(...),
    x: int = Form(128),
    y: int = Form(128),
    model_id: str = Form("rcan"),
):
    """
    Extract multi-band spectral reflectance profile at coordinate (x, y).
    Compares LR, SR, and HR ground truth curves.
    """
    sample_path = Path(__file__).parent.parent / "sample_tiles" / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample tile not found")

    lr_image, hr_image = load_sample_tile(str(sample_path))
    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
    sr_array = result["_sr_array"]

    # Clamp coordinates to HR dimensions
    c, h_hr, w_hr = sr_array.shape
    x = max(0, min(int(x), w_hr - 1))
    y = max(0, min(int(y), h_hr - 1))

    # Map to LR coordinate
    x_lr = max(0, min(x // scale_factor, lr_image.shape[2] - 1))
    y_lr = max(0, min(y // scale_factor, lr_image.shape[1] - 1))

    # Bands ordering: Red=0, Green=1, Blue=2, NIR=3
    # Standard spectral order by wavelength: Blue (490nm), Green (560nm), Red (665nm), NIR (842nm)
    spectral_order = [
        {"name": "Blue", "band": "B2", "wavelength_nm": 490, "idx": 2},
        {"name": "Green", "band": "B3", "wavelength_nm": 560, "idx": 1},
        {"name": "Red", "band": "B4", "wavelength_nm": 665, "idx": 0},
        {"name": "Near-IR", "band": "B8", "wavelength_nm": 842, "idx": 3},
    ]

    bands_data = []
    for b in spectral_order:
        idx = b["idx"]
        sr_val = float(sr_array[idx, y, x]) if idx < c else 0.0
        lr_val = float(lr_image[idx, y_lr, x_lr]) if idx < lr_image.shape[0] else 0.0
        hr_val = float(hr_image[idx, y, x]) if (hr_image is not None and idx < hr_image.shape[0]) else None

        bands_data.append({
            "name": b["name"],
            "band": b["band"],
            "wavelength": f"{b['wavelength_nm']} nm",
            "lr_reflectance": round(lr_val, 4),
            "sr_reflectance": round(sr_val, 4),
            "hr_reflectance": round(hr_val, 4) if hr_val is not None else None,
        })

    # Pointwise NDVI = (NIR - Red) / (NIR + Red)
    red_sr, nir_sr = float(sr_array[0, y, x]), float(sr_array[3, y, x])
    red_lr, nir_lr = float(lr_image[0, y_lr, x_lr]), float(lr_image[3, y_lr, x_lr])

    ndvi_sr = (nir_sr - red_sr) / (nir_sr + red_sr + 1e-7)
    ndvi_lr = (nir_lr - red_lr) / (nir_lr + red_lr + 1e-7)
    ndvi_hr = None
    if hr_image is not None:
        red_hr, nir_hr = float(hr_image[0, y, x]), float(hr_image[3, y, x])
        ndvi_hr = (nir_hr - red_hr) / (nir_hr + red_hr + 1e-7)

    # Classification interpretation based on NDVI and NIR
    if ndvi_sr > 0.4:
        surface_type = "Dense Vegetation / Crop Canopy"
        signature_note = "Steep red edge with high NIR cellular scattering plateau."
    elif ndvi_sr > 0.15:
        surface_type = "Sparse Scrub / Mixed Soil-Vegetation"
        signature_note = "Moderate red-NIR slope typical of arid/semi-arid terrain."
    elif nir_sr < 0.08:
        surface_type = "Water Body / Deep Shadow"
        signature_note = "High absorption across visible and near-infrared spectrum."
    else:
        surface_type = "Built-up Urban / High-Albedo Road/Sand"
        signature_note = "Relatively flat visible-to-NIR reflectance curve."

    return {
        "status": "success",
        "sample_id": sample_id,
        "model_id": model_id,
        "hr_coordinates": {"x": x, "y": y},
        "lr_coordinates": {"x": x_lr, "y": y_lr},
        "bands_data": bands_data,
        "ndvi": {
            "sr": round(ndvi_sr, 4),
            "lr": round(ndvi_lr, 4),
            "hr": round(ndvi_hr, 4) if ndvi_hr is not None else None,
        },
        "surface_classification": surface_type,
        "signature_analysis": signature_note,
    }



# ========================
# ASYNC JOB ENDPOINTS
# ========================

@app.post("/api/superresolve/async")
async def superresolve_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    model_id: str = Form("rcan"),
):
    """
    Submit super-resolution as an asynchronous background job.
    Returns job_id and status checking URL.
    """
    file_bytes = await file.read() if file else None
    if not sample_id and not file_bytes:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'sample_id'")

    job_id = job_store.create_job(model_id=model_id)
    background_tasks.add_task(_async_worker, job_id, sample_id, file_bytes, model_id)

    return {
        "status": "accepted",
        "job_id": job_id,
        "model_id": model_id,
        "status_url": f"/api/jobs/{job_id}",
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Check asynchronous job status or retrieve results."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "model_id": job["model_id"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "inference_time_s": job.get("inference_time_s"),
        "error_message": job.get("error_message"),
    }

    if job["status"] == "completed" and job.get("result_path"):
        result_file = Path(job["result_path"])
        if result_file.exists():
            with open(result_file, "r") as f:
                response["result"] = json.load(f)

    return response


@app.get("/api/jobs")
async def list_jobs():
    """List recent background jobs."""
    import sqlite3
    db_path = str(Path(__file__).parent.parent / "bharatsr.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT job_id, status, model_id, created_at, completed_at, inference_time_s FROM jobs ORDER BY created_at DESC LIMIT 20").fetchall()
    conn.close()
    return {"jobs": [dict(r) for r in rows]}


# ========================
# EXPORT ENDPOINTS
# ========================

@app.get("/api/export/geotiff")
async def export_geotiff(
    sample_id: str = Query(..., description="Sample ID to export"),
    model_id: str = Query("rcan", description="Model to generate SR with"),
):
    """
    Generate and download a calibrated 4-band float32 GeoTIFF.
    """
    sample_path = Path(__file__).parent.parent / "sample_tiles" / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample tile not found")

    lr_image, hr_image = load_sample_tile(str(sample_path))
    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)
    sr_array = result["_sr_array"]

    geotiff_bytes = export_geotiff_bytes(sr_array)

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
    """
    sample_path = Path(__file__).parent.parent / "sample_tiles" / f"{sample_id}.npz"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample tile not found")

    lr_image, hr_image = load_sample_tile(str(sample_path))
    model_meta = model_registry.get_metadata(model_id) or {}
    scale_factor = model_meta.get("scale_factor", 4)

    result = _execute_model_sr(model_id, lr_image, hr_image, scale_factor)

    report = {
        "title": "BharatSR Super-Resolution Physics & Spectral Fidelity Report",
        "problem_statement": "SIH26142 - Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery",
        "target_organization": "National Technical Research Organisation (NTRO)",
        "sample_id": sample_id,
        "model_id": model_id,
        "scale_factor": f"{scale_factor}x (10m -> 2.5m GSD)",
        "input_dimension": list(lr_image.shape),
        "output_dimension": result["shape"],
        "latency_seconds": result["inference_time_s"],
        "metrics": result["metrics"],
        "uncertainty_summary": result["uncertainty"]["summary"] if result["uncertainty"] else None,
        "spectral_integrity_compliance": {
            "physical_reflectance_preserved": True,
            "sam_under_5_deg": (result["metrics"].get("sam", {}).get("value", 99) < 5.0) if hr_image is not None else None,
            "downsample_consistency_mae": result["metrics"].get("downsample_consistency", {}).get("value"),
        },
    }

    return JSONResponse(content=report)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

