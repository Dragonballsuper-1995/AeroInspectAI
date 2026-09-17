"""Results retrieval routes."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.services.result_service import ResultService
from app.schemas import InspectionRecord
from app.core.logging_config import logger

router = APIRouter(prefix="/api/v1", tags=["results"])
result_service = ResultService()


@router.get("/results/{inspection_id}")
async def get_inspection_result(inspection_id: str):
    """
    Get complete inspection result as JSON.
    
    Args:
        inspection_id: Inspection ID.
        
    Returns:
        Inspection result.
        
    Raises:
        HTTPException: If not found.
    """
    try:
        result = result_service.get_inspection_result(inspection_id)
        return result
    except FileNotFoundError:
        logger.warning(f"Inspection not found: {inspection_id}")
        raise HTTPException(status_code=404, detail="Inspection not found")
    except Exception as e:
        logger.error(f"Error retrieving inspection: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving inspection")


@router.get("/results/{inspection_id}/original")
async def get_original_image(inspection_id: str):
    """
    Get original uploaded image.
    
    Args:
        inspection_id: Inspection ID.
        
    Returns:
        Image file response.
        
    Raises:
        HTTPException: If not found.
    """
    try:
        image_path = result_service.get_original_image_path(inspection_id)
        return FileResponse(image_path, media_type="image/jpeg")
    except FileNotFoundError:
        logger.warning(f"Original image not found: {inspection_id}")
        raise HTTPException(status_code=404, detail="Original image not found")
    except ValueError as e:
        logger.warning(f"Invalid inspection ID: {e}")
        raise HTTPException(status_code=400, detail="Invalid inspection ID")
    except Exception as e:
        logger.error(f"Error retrieving original image: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving image")


@router.get("/results/{inspection_id}/annotated")
async def get_annotated_image(inspection_id: str):
    """
    Get annotated result image.
    
    Args:
        inspection_id: Inspection ID.
        
    Returns:
        Image file response.
        
    Raises:
        HTTPException: If not found.
    """
    try:
        image_path = result_service.get_annotated_image_path(inspection_id)
        return FileResponse(image_path, media_type="image/jpeg")
    except FileNotFoundError:
        logger.warning(f"Annotated image not found: {inspection_id}")
        raise HTTPException(status_code=404, detail="Annotated image not found")
    except ValueError as e:
        logger.warning(f"Invalid inspection ID: {e}")
        raise HTTPException(status_code=400, detail="Invalid inspection ID")
    except Exception as e:
        logger.error(f"Error retrieving annotated image: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving image")
