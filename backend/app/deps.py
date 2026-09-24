"""
BharatSR — Dependency Injection (FastAPI Depends)
Provides settings, service instances, authentication, and security checks.
"""

from functools import lru_cache
import time
from collections import defaultdict
from typing import Optional, Dict, List
import threading

from fastapi import Depends, Header, Request, UploadFile

from backend.app.config import Settings
from backend.app.services.inference import ModelRegistry, model_registry
from backend.app.services.job_store import JobStore
from backend.app.core.exceptions import (
    AuthenticationError,
    RateLimitExceededError,
    PayloadTooLargeError,
)
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.deps")


@lru_cache
def get_settings() -> Settings:
    """Returns application settings cached as a singleton."""
    return Settings()


def get_model_registry() -> ModelRegistry:
    """Provides the global ModelRegistry instance."""
    return model_registry


def get_job_store(request: Request) -> JobStore:
    """Provides the JobStore instance stored on app state."""
    if hasattr(request.app.state, "job_store") and request.app.state.job_store is not None:
        return request.app.state.job_store
    # Fallback to direct instantiation from settings
    settings = get_settings()
    return JobStore(settings.db_path)


def get_thread_pool(request: Request):
    """Provides the persistent ThreadPoolExecutor stored on app state."""
    return getattr(request.app.state, "thread_pool", None)


# -------------------------------------------------------------
# Security: API Key Verification
# -------------------------------------------------------------
async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Verifies X-API-Key header on mutating or compute-intensive endpoints.
    Bypassed in development mode when settings.api_key is None or empty.
    """
    if not settings.api_key:
        return  # Dev/unsecured mode permitted
    if not x_api_key or x_api_key != settings.api_key:
        raise AuthenticationError("Invalid or missing X-API-Key header.")


# -------------------------------------------------------------
# Security: In-Memory Sliding-Window Rate Limiter
# -------------------------------------------------------------
class InMemoryRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter per client IP."""

    def __init__(self):
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, client_ip: str, max_requests: int, window_seconds: float = 60.0) -> bool:
        now = time.time()
        with self._lock:
            # Purge expired timestamps
            timestamps = [t for t in self._requests[client_ip] if now - t < window_seconds]
            if len(timestamps) >= max_requests:
                self._requests[client_ip] = timestamps
                return False
            timestamps.append(now)
            self._requests[client_ip] = timestamps
            return True


_rate_limiter = InMemoryRateLimiter()


async def check_rate_limit(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Enforces max_requests_per_minute based on client host IP."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.check(client_ip, settings.max_requests_per_minute):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise RateLimitExceededError(
            f"Rate limit of {settings.max_requests_per_minute} req/min exceeded. Please try again shortly."
        )


# -------------------------------------------------------------
# Security: Upload Size Enforcement
# -------------------------------------------------------------
async def check_upload_size(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Enforces max_image_bytes against Content-Length header before stream processing."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
            if length > settings.max_image_bytes:
                max_mb = settings.max_image_bytes // (1024 * 1024)
                logger.warning(f"Upload rejected: {length} bytes > {settings.max_image_bytes} max limit")
                raise PayloadTooLargeError(max_mb)
        except ValueError:
            pass


async def read_uploaded_file_capped(file: Optional[UploadFile], max_bytes: int) -> Optional[bytes]:
    """Reads uploaded file with a strict byte cap to prevent memory exhaustion."""
    if not file:
        return None
    chunk_size = 1024 * 1024
    total_read = 0
    chunks = []
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > max_bytes:
            max_mb = max_bytes // (1024 * 1024)
            logger.warning(f"Upload stream capped: {total_read} bytes > {max_bytes} max limit")
            raise PayloadTooLargeError(max_mb)
        chunks.append(chunk)
    return b"".join(chunks)

