"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from roto_seg.api import health, projects, shots, exports
from roto_seg.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print(f"Starting Roto-Seg API v{settings.VERSION}")
    yield
    # Shutdown
    print("Shutting down Roto-Seg API")


app = FastAPI(
    title="Roto-Seg API",
    description="AI-powered rotoscoping automation system",
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
app.include_router(shots.router, prefix="/api/v1/shots", tags=["Shots"])
app.include_router(exports.router, prefix="/api/v1/exports", tags=["Exports"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Roto-Seg API",
        "version": settings.VERSION,
        "docs": "/docs",
    }
