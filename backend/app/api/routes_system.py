"""System status routes."""

from fastapi import APIRouter
import torch
from ultralytics import __version__ as ultralytics_version
from app.core.config import settings
from app.services import get_inference_service

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/system/status")
async def system_status():
    """
    Get system status including model and compute info.
    
    Returns:
        System status dictionary.
    """
    inference_service = get_inference_service()
    
    # Determine GPU info
    gpu_name = "N/A"
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(settings.device)
        except Exception:
            gpu_name = "Unknown GPU"
    
    return {
        "backend": {
            "status": "running"
        },
        "model": {
            "loaded": inference_service.is_loaded(),
            "name": "YOLOv8n-Seg",
            "path": str(settings.get_model_path()),
            "classes": inference_service.classes
        },
        "compute": {
            "device": inference_service.device,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics_version,
            "vram_gb": round(
                torch.cuda.get_device_properties(settings.device).total_memory / (1024 ** 3), 2
            ) if cuda_available else 0
        }
    }
