"""Health check routes."""

from fastapi import APIRouter
from app.schemas import HealthResponse
from app.core.config import settings

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status.
    """
    return {
        "status": "healthy",
        "service": "AeroInspectAI Backend",
        "version": "0.1.0"
    }
