"""Inspection API routes."""

import time
from uuid import uuid4
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from app.core.config import settings
from app.core.logging_config import logger
from app.schemas import InspectionResult, ErrorResponse
from app.services.inspection_service import InspectionService
from app.services.result_service import ResultService
from app.utils.file_utils import ensure_directory
from app.utils.image_utils import sanitize_filename
from app.utils import image_utils

router = APIRouter(prefix="/api/v1", tags=["inspection"])
inspection_service = InspectionService()
result_service = ResultService()


@router.post("/inspect", response_model=InspectionResult)
async def inspect_image(
    file: Optional[UploadFile] = File(None),
    confidence: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None)
):
    """
    Upload image and perform crack inspection.
    
    Args:
        file: Image file to inspect.
        confidence: Optional confidence threshold override.
        iou: Optional IoU threshold override.
        imgsz: Optional image size override.
        
    Returns:
        InspectionResult with detections and annotated image.
        
    Raises:
        HTTPException: On validation or processing errors.
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Validate file was provided
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file size
        file_content = await file.read()
        file_size_bytes = len(file_content)
        max_size_bytes = settings.get_max_upload_bytes()
        
        if file_size_bytes > max_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.max_upload_mb}MB"
            )
        
        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Sanitize filename
        sanitized_filename = sanitize_filename(file.filename)
        
        # Save to temporary upload location
        upload_dir = settings.get_upload_path()
        ensure_directory(upload_dir)
        temp_file_path = upload_dir / f"{uuid4().hex}_{sanitized_filename}"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Upload saved: {temp_file_path} ({file_size_bytes} bytes)")
        
        # Validate image format and contents
        if not image_utils.is_valid_image(temp_file_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Supported: JPG, PNG, WEBP"
            )
        
        logger.info(f"Image validation passed: {sanitized_filename}")
        
        # Validate inference parameters if provided
        if confidence is not None:
            if not 0 <= confidence <= 1:
                raise HTTPException(
                    status_code=422,
                    detail="Confidence must be between 0 and 1"
                )
        
        if iou is not None:
            if not 0 <= iou <= 1:
                raise HTTPException(
                    status_code=422,
                    detail="IoU must be between 0 and 1"
                )
        
        if imgsz is not None:
            if imgsz < 32 or imgsz > 1920:
                raise HTTPException(
                    status_code=422,
                    detail="Image size must be between 32 and 1920"
                )
        
        logger.info(
            f"Processing inspection: {sanitized_filename} "
            f"(conf={confidence}, iou={iou}, imgsz={imgsz})"
        )
        
        # Run inspection
        result = inspection_service.process_inspection(
            uploaded_file_path=temp_file_path,
            original_filename=sanitized_filename,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Inspection completed in {total_time_ms:.1f}ms - "
            f"ID: {result.inspection_id}"
        )
        
        return result
    
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise HTTPException(status_code=500, detail="File handling error")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        if "Model not loaded" in str(e):
            raise HTTPException(status_code=503, detail="Model not loaded")
        raise HTTPException(status_code=500, detail="Inference error")
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        raise HTTPException(status_code=500, detail="Inspection processing failed")
    
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


@router.get("/inspections")
async def get_inspections(limit: int = Query(100, ge=1, le=500)):
    """
    Get inspection history.
    
    Args:
        limit: Maximum number of records to return.
        
    Returns:
        List of inspection records.
    """
    try:
        history = result_service.get_inspections_history(limit=limit)
        return history
    except Exception as e:
        logger.error(f"Error retrieving inspection history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving history")


@router.get("/inspections/{inspection_id}")
async def get_inspection_by_id(inspection_id: str):
    """
    Get specific inspection result by ID.
    
    Args:
        inspection_id: Inspection ID.
        
    Returns:
        Complete inspection result.
        
    Raises:
        HTTPException: If not found.
    """
    try:
        if not result_service.inspection_exists(inspection_id):
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        result = result_service.get_inspection_result(inspection_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving inspection {inspection_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving inspection")
