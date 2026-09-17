"""
BharatSR — Backend Configuration
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Paths
    project_root: str = str(Path(__file__).resolve().parent.parent.parent)
    weights_dir: str = str(Path(__file__).resolve().parent.parent / "weights")
    sample_tiles_dir: str = str(Path(__file__).resolve().parent.parent / "sample_tiles")
    results_dir: str = str(Path(__file__).resolve().parent.parent / "results")

    # Model
    default_model: str = "srcnn"
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


settings = Settings()
