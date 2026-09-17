"""FastAPI application setup and configuration."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging_config import logger
from app.api import routes_health, routes_system, routes_inspection, routes_results, routes_simulation
from app.services import get_inference_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the inference service once and log a clean application lifecycle."""
    logger.info("=" * 60)
    logger.info("AEROINSPECT AI - FASTAPI BACKEND")
    logger.info("=" * 60)

    inference_service = get_inference_service()
    if not inference_service.is_loaded():
        logger.error("Model failed to load at startup!")
        raise RuntimeError("Model initialization failed")

    logger.info("Backend:      http://localhost:8000")
    logger.info("API Docs:     http://localhost:8000/docs")
    logger.info("Health:       http://localhost:8000/api/v1/health")
    logger.info("")
    logger.info("Model:        YOLOv8n-Seg")
    logger.info("Model loaded: YES")
    logger.info(f"Device:       {inference_service.device}")
    logger.info(f"GPU:          {inference_service.get_device_name()}")
    logger.info("=" * 60)

    yield
    logger.info("Backend shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Autonomous Drone-Based Structural Crack Inspection Backend",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_health.router)
app.include_router(routes_system.router)
app.include_router(routes_inspection.router)
app.include_router(routes_results.router)
app.include_router(routes_simulation.router)
