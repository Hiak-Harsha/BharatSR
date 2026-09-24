"""
BharatSR — Backend Configuration
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Application
    app_name: str = "BharatSR"
    version: str = "2.0.0"

    # Paths
    project_root: str = str(Path(__file__).resolve().parent.parent.parent)
    weights_dir: str = str(Path(__file__).resolve().parent.parent / "weights")
    sample_tiles_dir: str = str(Path(__file__).resolve().parent.parent / "sample_tiles")
    results_dir: str = str(Path(__file__).resolve().parent.parent / "results")
    runs_dir: str = str(Path(__file__).resolve().parent.parent / "runs")

    # Redis (optional, falls back to in-memory/thread pool)
    redis_url: str = ""

    # Concurrency and Limits
    max_workers: int = 4
    max_tile_size: int = 64
    max_image_bytes: int = 100 * 1024 * 1024
    job_ttl_hours: int = 24
    max_requests_per_minute: int = 30

    # Model
    default_model: str = "rcan"
    scale_factor: int = 4
    n_bands: int = 4

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Database
    db_path: str = str(Path(__file__).resolve().parent.parent / "bharatsr.db")

    class Config:
        env_file = ".env"
        env_prefix = "BHARATSR_"
        extra = "allow"


settings = Settings()
