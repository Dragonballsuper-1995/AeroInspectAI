"""API schema definitions."""

from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")


class MaskInfo(BaseModel):
    """Segmentation mask information."""
    polygon: list[list[float]] = Field(
        ..., description="Polygon coordinates of the mask"
    )
    area_pixels: int = Field(..., description="Area in pixels")
    area_ratio: float = Field(..., description="Area ratio relative to image")


class Detection(BaseModel):
    """A single detection result."""
    id: int = Field(..., description="Detection ID")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., description="Confidence score")
    bounding_box: BoundingBox = Field(..., description="Bounding box")
    mask: Optional[MaskInfo] = Field(None, description="Segmentation mask")


class ImageInfo(BaseModel):
    """Image information in result."""
    original_url: str = Field(..., description="URL to original image")
    annotated_url: str = Field(..., description="URL to annotated image")


class SummaryInfo(BaseModel):
    """Summary statistics."""
    detections: int = Field(..., description="Number of detections")
    cracks_detected: int = Field(..., description="Number of cracks")
    average_confidence: float = Field(..., description="Average confidence")
    max_confidence: float = Field(..., description="Maximum confidence")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence")
    total_mask_area_pixels: int = Field(..., description="Total mask area in pixels")
    image_width: int = Field(..., description="Image width")
    image_height: int = Field(..., description="Image height")


class InferenceInfo(BaseModel):
    """Inference configuration and timing."""
    model: str = Field(..., description="Model name")
    device: str = Field(..., description="Compute device")
    image_size: int = Field(..., description="Input image size")
    confidence_threshold: float = Field(..., description="Confidence threshold")
    iou_threshold: float = Field(..., description="IoU threshold")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class InspectionResult(BaseModel):
    """Complete inspection result."""
    inspection_id: str = Field(..., description="Unique inspection ID")
    status: str = Field(..., description="Status (completed, failed, etc.)")
    image: ImageInfo = Field(..., description="Image information")
    summary: SummaryInfo = Field(..., description="Summary statistics")
    detections: list[Detection] = Field(..., description="Detection results")
    inference: InferenceInfo = Field(..., description="Inference info")


class InspectionRecord(BaseModel):
    """Inspection history record."""
    inspection_id: str = Field(..., description="Inspection ID")
    timestamp: str = Field(..., description="Timestamp (ISO format)")
    status: str = Field(..., description="Status")
    filename: str = Field(..., description="Original filename")
    detection_count: int = Field(..., description="Number of detections")
    average_confidence: float = Field(..., description="Average confidence")
    max_confidence: float = Field(..., description="Max confidence")
    processing_time_ms: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Version")


class ErrorResponse(BaseModel):
    """Error response."""
    error: dict = Field(..., description="Error details")


class SystemStatus(BaseModel):
    """System status response."""
    backend: dict = Field(..., description="Backend status")
    model: dict = Field(..., description="Model status")
    compute: dict = Field(..., description="Compute status")
