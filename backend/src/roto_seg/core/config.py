"""
Application configuration using Pydantic Settings.
"""

import platform
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_device() -> str:
    """Detect the best available device for ML inference."""
    # Check for Apple Silicon (M1/M2/M3)
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"  # Metal Performance Shaders
    # Default to CPU (CUDA requires explicit setup)
    return "cpu"


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Application
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENV: str = "development"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:4200", "http://localhost:5173", "http://127.0.0.1:4200"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rotoseg"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage
    STORAGE_PATH: str = "./storage"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    # AI Models
    MODEL_PATH: str = "./models"
    SAM2_MODEL: str = "sam2_hiera_base_plus.pt"  # base+ model - better quality
    DEVICE: str = get_default_device()  # Auto-detect: mps (M1), cuda, or cpu

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
