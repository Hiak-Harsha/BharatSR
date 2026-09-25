"""
BharatSR — FastAPI Application Server Factory
Problem Statement SIH26142: Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery
Target: Sentinel-2 L2A (10m) -> 4x SR on 2.5m-equivalent grid (B2, B3, B4, B8)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings, Settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.core.exceptions import register_exception_handlers
from backend.app.services.inference import model_registry
from backend.app.services.job_store import JobStore

# Import feature routers
from backend.app.routers.health import router as health_router
from backend.app.routers.models import router as models_router
from backend.app.routers.samples import router as samples_router
from backend.app.routers.inference import router as inference_router
from backend.app.routers.jobs import router as jobs_router
from backend.app.routers.batch import router as batch_router
from backend.app.routers.export import router as export_router
from backend.app.routers.analysis import router as analysis_router
from backend.app.routers.admin import router as admin_router
from backend.app.routers.dataset import router as dataset_router

logger = get_logger("bharatsr.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and graceful shutdown."""
    settings = get_settings()
    setup_logging()

    logger.info(f"Starting {settings.app_name} v{settings.version}")

    # Ensure runtime directories exist
    runs_dir = Path(settings.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize persistent single-node threadpool
    thread_pool = ThreadPoolExecutor(max_workers=settings.max_workers)
    app.state.thread_pool = thread_pool

    # Initialize JobStore
    job_store = JobStore(settings.db_path)
    app.state.job_store = job_store

    # Cleanup expired jobs (>24 hours old) on startup
    deleted = job_store.cleanup_expired_jobs(max_age_hours=settings.job_ttl_hours, runs_dir=runs_dir)
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} expired job records.")

    # Load pre-trained models from weights directory
    weights_dir = Path(settings.weights_dir)
    for model_name in ["srcnn", "rcan", "swinir", "hat"]:
        ckpt_path = weights_dir / f"{model_name}_best.pth"
        if ckpt_path.exists():
            success = model_registry.load_model(model_name, str(ckpt_path))
            if success:
                logger.info(f"Loaded {model_name.upper()} checkpoint from {ckpt_path}")
        else:
            logger.info(f"{model_name.upper()} checkpoint not present at {ckpt_path}")

    loaded_models = [m["id"] for m in model_registry.list_models()]
    logger.info(f"Active models in registry on device ({model_registry.device}): {loaded_models}")

    yield

    logger.info("Initiating graceful shutdown...")
    thread_pool.shutdown(wait=False)
    logger.info("BharatSR backend shutdown complete.")


def create_app() -> FastAPI:
    """Application factory for BharatSR FastAPI service."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Scientifically Defensible Deep Learning Super-Resolution for Satellite Earth Observation (SIH26142)",
        version=settings.version,
        lifespan=lifespan,
    )

    # Strict CORS configuration from settings
    cors_origins = settings.cors_origins_list
    if not cors_origins:
        cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"^https://.*\.vercel\.app$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time", "Content-Disposition"],
    )

    @app.middleware("http")
    async def request_id_and_telemetry_middleware(request, call_next):
        import uuid
        import time
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        response = await call_next(request)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time"] = f"{latency_ms}ms"
        logger.info(
            f"[{req_id}] {request.method} {request.url.path} -> {response.status_code} ({latency_ms}ms)"
        )
        return response

    # Register standardized error handlers
    register_exception_handlers(app)

    # Register modular feature routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(samples_router)
    app.include_router(inference_router)
    app.include_router(jobs_router)
    app.include_router(batch_router)
    app.include_router(export_router)
    app.include_router(analysis_router)
    app.include_router(admin_router)
    app.include_router(dataset_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    app_settings = get_settings()
    uvicorn.run("backend.app.main:app", host=app_settings.host, port=app_settings.port, reload=True)
