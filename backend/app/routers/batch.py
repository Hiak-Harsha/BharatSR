"""
BharatSR — Batch Processing Router
Handles parallel multi-tile batch submission and status polling.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.config import Settings
from backend.app.deps import (
    get_settings,
    get_job_store,
    get_thread_pool,
    verify_api_key,
    check_rate_limit,
    check_upload_size,
)
from backend.app.schemas import BatchSubmitResponse, BatchStatusResponse
from backend.app.services.job_store import JobStore
from backend.app.services.sr_pipeline import async_worker
from backend.app.routers.inference import ALLOWED_MODELS
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.batch")
router = APIRouter(tags=["batch"])


@router.post(
    "/api/batch",
    response_model=BatchSubmitResponse,
    dependencies=[Depends(check_rate_limit), Depends(check_upload_size), Depends(verify_api_key)],
)
async def batch_superresolve(
    files: List[UploadFile] = File(None),
    sample_ids: str = Form(None),
    model_id: str = Form("rcan"),
    quality: str = Form("fast"),
    settings: Settings = Depends(get_settings),
    store: JobStore = Depends(get_job_store),
    pool=Depends(get_thread_pool),
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
        raise HTTPException(status_code=400, detail="Provide 'files' or 'sample_ids'")
    if total_inputs > 10:
        raise HTTPException(status_code=400, detail=f"Max 10 tiles per batch. Got {total_inputs}.")

    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    job_ids = []
    runs_dir = Path(settings.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sample_tiles_dir = Path(settings.sample_tiles_dir)
    loop = asyncio.get_event_loop()

    for sid in parsed_sample_ids:
        job_id = store.create_job(model_id=model_id)
        job_ids.append(job_id)
        if pool is not None:
            loop.run_in_executor(
                pool,
                async_worker,
                job_id,
                sid,
                None,
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
                    sid,
                    None,
                    model_id,
                    quality,
                    store,
                    runs_dir,
                    sample_tiles_dir,
                )
            )

    for f in uploaded_files:
        file_bytes = await f.read()
        job_id = store.create_job(model_id=model_id)
        job_ids.append(job_id)
        if pool is not None:
            loop.run_in_executor(
                pool,
                async_worker,
                job_id,
                None,
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
                    None,
                    file_bytes,
                    model_id,
                    quality,
                    store,
                    runs_dir,
                    sample_tiles_dir,
                )
            )

    batch_manifest = {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "model_id": model_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    batch_path = runs_dir / f"{batch_id}.json"
    with open(batch_path, "w") as bf:
        json.dump(batch_manifest, bf)

    return JSONResponse(
        content={
            "status": "accepted",
            "batch_id": batch_id,
            "job_ids": job_ids,
            "total": total_inputs,
            "status_url": f"/api/batch/{batch_id}",
        }
    )


@router.get("/api/batch/{batch_id}", response_model=BatchStatusResponse)
def get_batch_status(
    batch_id: str,
    settings: Settings = Depends(get_settings),
    store: JobStore = Depends(get_job_store),
):
    """Get status and results of all jobs in a batch."""
    runs_dir = Path(settings.runs_dir)
    batch_path = runs_dir / f"{batch_id}.json"
    if not batch_path.exists():
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    with open(batch_path) as bf:
        manifest = json.load(bf)

    job_statuses = []
    for jid in manifest["job_ids"]:
        job = store.get_job(jid)
        job_statuses.append(job or {"job_id": jid, "status": "not_found"})

    completed = sum(1 for j in job_statuses if j.get("status") == "completed")
    failed = sum(1 for j in job_statuses if j.get("status") == "failed")
    total = len(job_statuses)

    overall_status = "completed" if completed == total else "partial_failure" if failed > 0 else "processing"

    return JSONResponse(
        content={
            "batch_id": batch_id,
            "overall_status": overall_status,
            "completed": completed,
            "failed": failed,
            "total": total,
            "jobs": job_statuses,
        }
    )
