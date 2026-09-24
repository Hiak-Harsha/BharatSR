"""
BharatSR — Core Exception Classes & Handlers
Standardizes error response contracts across the entire FastAPI backend.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from backend.app.core.logging import get_logger

logger = get_logger("bharatsr.exceptions")


class BharatSRError(Exception):
    """Base exception for all domain errors in BharatSR."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotFoundError(BharatSRError):
    """Raised when a requested model is not found in the registry or checkpoint is missing."""

    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' not loaded or checkpoint missing.", status_code=400)


class SampleNotFoundError(BharatSRError):
    """Raised when a preloaded sample tile does not exist on disk."""

    def __init__(self, sample_id: str):
        super().__init__(f"Sample '{sample_id}' not found", status_code=404)


class InvalidInputError(BharatSRError):
    """Raised when input imagery or parameters are invalid."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message, status_code=status_code)


class GeospatialError(BharatSRError):
    """Raised when geospatial coordinates or CRS operations fail."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code)


class PayloadTooLargeError(BharatSRError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, max_mb: int):
        super().__init__(f"Payload exceeds maximum allowed upload size of {max_mb}MB.", status_code=413)


class RateLimitExceededError(BharatSRError):
    """Raised when client exceeds allowed requests per minute."""

    def __init__(self, message: str = "Rate limit exceeded. Please slow down."):
        super().__init__(message, status_code=429)


class AuthenticationError(BharatSRError):
    """Raised when invalid or missing API key is provided."""

    def __init__(self, message: str = "Invalid or missing X-API-Key header."):
        super().__init__(message, status_code=401)


def register_exception_handlers(app: FastAPI) -> None:
    """Registers unified error handlers for BharatSR exceptions and FastAPI HTTPExceptions."""

    @app.exception_handler(BharatSRError)
    async def bharatsr_error_handler(request: Request, exc: BharatSRError):
        logger.warning(f"{exc.__class__.__name__} on {request.method} {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "detail": exc.message,
                "status": exc.status_code,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTPException ({exc.status_code}) on {request.method} {request.url.path}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "detail": exc.detail,
                "status": exc.status_code,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "An internal server error occurred while processing the request.",
                "status": 500,
            },
        )
