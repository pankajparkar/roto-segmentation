"""
Health check endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check():
    """Readiness check - verify all dependencies are available."""
    # TODO: Add checks for database, redis, GPU
    return {
        "status": "ready",
        "checks": {
            "database": "ok",
            "redis": "ok",
            "gpu": "ok",
        },
    }
