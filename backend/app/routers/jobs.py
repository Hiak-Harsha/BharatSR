"""
BharatSR — Jobs & Real-Time Telemetry Router
Provides tracking, status, cancellation, and WebSocket streaming for asynchronous jobs.
"""

import asyncio
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from backend.app.schemas import JobStatusResponse, JobListResponse
from backend.app.deps import get_job_store, verify_api_key
from backend.app.services.job_store import JobStore
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.jobs")
router = APIRouter(tags=["jobs"])


@router.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, store: JobStore = Depends(get_job_store)):
    """Check asynchronous job status, progress percentage, or retrieve final results."""
    job = store.get_job(job_id)
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
                if result_file.suffix == ".json":
                    with open(result_file, "r", encoding="utf-8") as f:
                        response_data["result"] = json.load(f)
                elif result_file.suffix == ".npz":
                    companion = result_file.with_name(f"{result_file.stem}_result.json")
                    if companion.exists():
                        with open(companion, "r", encoding="utf-8") as f:
                            response_data["result"] = json.load(f)
                    else:
                        logger.info(f"Result file is array-only .npz: {result_file}")
            except Exception as e:
                logger.warning(f"Failed to load cached result from {result_file}: {e}")
                response_data["error_message"] = f"Failed to load cached result: {e}"


    return JobStatusResponse(**response_data)


@router.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_job(job_id: str, store: JobStore = Depends(get_job_store)):
    """Cancel a running or pending asynchronous job."""
    success = store.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' cannot be cancelled (either not found or already completed/failed).",
        )
    return {"status": "success", "message": f"Job '{job_id}' cancellation requested."}


@router.get("/api/jobs", response_model=JobListResponse)
def list_jobs(store: JobStore = Depends(get_job_store)):
    """List recent background processing jobs."""
    jobs = store.list_jobs(limit=20)
    return JobListResponse(jobs=jobs)


@router.websocket("/ws/inference/{job_id}")
async def websocket_inference_progress(
    websocket: WebSocket,
    job_id: str,
):
    """
    WebSocket endpoint for real-time job progress streaming.
    Client receives: {"job_id": ..., "status": ..., "progress_pct": ...}
    Capped at 600 iterations (~5 minutes) to prevent zombie loops.
    """
    await websocket.accept()
    # Resolve job_store from app state
    store: JobStore = getattr(websocket.app.state, "job_store", None)
    if store is None:
        await websocket.send_json({"error": "Job store service unavailable"})
        await websocket.close()
        return

    max_iterations = 600  # 600 * 0.5s = 5 minutes timeout cap
    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1
            job = store.get_job(job_id)
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

        if iteration >= max_iterations:
            await websocket.send_json({"error": "Progress streaming reached max duration timeout."})

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket telemetry error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.debug(f"Ignored error closing websocket for job {job_id}: {e}")

