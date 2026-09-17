"""Inspection service for managing inspection workflows."""

import time
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import cv2

from app.core.logging_config import logger
from app.core.config import settings
from app.utils.file_utils import (
    get_unique_inspection_id,
    save_json,
    ensure_directory,
    get_timestamp
)
from app.utils import image_utils
from app.services import get_inference_service
from app.schemas import InspectionResult, ImageInfo, SummaryInfo, InferenceInfo, Detection, BoundingBox, MaskInfo


class InspectionService:
    """Service for managing inspection workflows."""
    
    def __init__(self):
        """Initialize inspection service."""
        self.inference_service = get_inference_service()
        self.inspections_dir = settings.get_inspection_path()
        self._history_lock = threading.Lock()
        ensure_directory(self.inspections_dir)
    
    def process_inspection(
        self,
        uploaded_file_path: Path,
        original_filename: str,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> InspectionResult:
        """
        Process a single inspection.
        
        Args:
            uploaded_file_path: Path to uploaded image.
            original_filename: Original filename.
            confidence: Confidence threshold override.
            iou: IoU threshold override.
            imgsz: Image size override.
            
        Returns:
            InspectionResult object.
            
        Raises:
            Exception: If inspection fails.
        """
        start_time = time.time()
        inspection_id = get_unique_inspection_id()
        
        try:
            logger.info(f"[INSPECTION] {inspection_id} - Starting inspection")
            
            # Create inspection directory
            inspection_dir = self.inspections_dir / inspection_id
            ensure_directory(inspection_dir)
            
            # Save original image
            original_image_path = inspection_dir / "original.jpg"
            cv2.imwrite(str(original_image_path), cv2.imread(str(uploaded_file_path)))
            
            # Get image dimensions
            img_width, img_height = image_utils.get_image_dimensions(uploaded_file_path)
            
            logger.info(
                f"[INSPECTION] {inspection_id} - Image: {original_filename} "
                f"({img_width}x{img_height})"
            )
            
            # Run inference
            inference_start = time.time()
            inference_results, inference_time_ms = self.inference_service.predict(
                uploaded_file_path,
                confidence=confidence,
                iou=iou,
                imgsz=imgsz
            )
            
            # Extract detection data
            detections_data = inference_results.get("detections", [])
            summary_data = inference_results.get("summary", {})
            
            logger.info(
                f"[INSPECTION] {inspection_id} - "
                f"Detections: {len(detections_data)}, "
                f"Inference: {inference_time_ms:.2f}ms"
            )
            
            # Generate annotated image
            annotated_image_path = inspection_dir / "annotated.jpg"
            original_image = image_utils.load_image(uploaded_file_path)
            annotated_image = image_utils.draw_segmentation_results(
                original_image,
                detections_data
            )
            image_utils.save_image(annotated_image, annotated_image_path)
            
            # Build detection objects
            detections_objs = []
            for det in detections_data:
                bbox = det.get("bounding_box", {})
                bbox_obj = BoundingBox(
                    x1=bbox.get("x1", 0),
                    y1=bbox.get("y1", 0),
                    x2=bbox.get("x2", 0),
                    y2=bbox.get("y2", 0)
                )
                
                mask_obj = None
                if det.get("mask"):
                    mask_data = det["mask"]
                    mask_obj = MaskInfo(
                        polygon=mask_data.get("polygon", []),
                        area_pixels=int(mask_data.get("area_pixels", 0)),
                        area_ratio=float(mask_data.get("area_ratio", 0))
                    )
                
                det_obj = Detection(
                    id=det.get("id", 0),
                    class_id=det.get("class_id", 0),
                    class_name=det.get("class_name", "unknown"),
                    confidence=float(det.get("confidence", 0)),
                    bounding_box=bbox_obj,
                    mask=mask_obj
                )
                detections_objs.append(det_obj)
            
            # Build summary
            summary_obj = SummaryInfo(
                detections=summary_data.get("detections", 0),
                cracks_detected=summary_data.get("cracks_detected", 0),
                average_confidence=float(summary_data.get("average_confidence", 0)),
                max_confidence=float(summary_data.get("max_confidence", 0)),
                min_confidence=summary_data.get("min_confidence"),
                total_mask_area_pixels=int(summary_data.get("total_mask_area_pixels", 0)),
                image_width=img_width,
                image_height=img_height
            )
            
            # Build inference info
            inference_params = inference_results.get("inference_params", {})
            device_name = self.inference_service.get_device_name()
            inference_obj = InferenceInfo(
                model="YOLOv8n-Seg",
                device=self.inference_service.device,
                image_size=inference_params.get("image_size", settings.image_size),
                confidence_threshold=float(inference_params.get("confidence_threshold", settings.confidence_threshold)),
                iou_threshold=float(inference_params.get("iou_threshold", settings.iou_threshold)),
                inference_time_ms=float(inference_time_ms)
            )
            
            # Build result
            image_obj = ImageInfo(
                original_url=f"/api/v1/results/{inspection_id}/original",
                annotated_url=f"/api/v1/results/{inspection_id}/annotated"
            )
            
            result = InspectionResult(
                inspection_id=inspection_id,
                status="completed",
                image=image_obj,
                summary=summary_obj,
                detections=detections_objs,
                inference=inference_obj
            )
            
            # Save result as JSON
            result_json_path = inspection_dir / "result.json"
            save_json(result.model_dump(), result_json_path)
            
            # Save inspection record for history
            total_time_ms = (time.time() - start_time) * 1000
            self._save_inspection_record(
                inspection_id=inspection_id,
                filename=original_filename,
                detection_count=len(detections_objs),
                average_confidence=summary_obj.average_confidence,
                max_confidence=summary_obj.max_confidence,
                processing_time_ms=total_time_ms
            )
            
            logger.info(
                f"[INSPECTION] {inspection_id} - COMPLETED "
                f"({total_time_ms:.1f}ms total) - "
                f"Detections: {len(detections_objs)}, "
                f"Avg Conf: {summary_obj.average_confidence:.3f}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"[INSPECTION] {inspection_id} - FAILED: {e}")
            inspection_dir = self.inspections_dir / inspection_id
            if inspection_dir.exists():
                shutil.rmtree(inspection_dir, ignore_errors=True)
            raise
    
    def _save_inspection_record(
        self,
        inspection_id: str,
        filename: str,
        detection_count: int,
        average_confidence: float,
        max_confidence: float,
        processing_time_ms: float
    ) -> None:
        """
        Save inspection record for history.
        
        Args:
            inspection_id: Inspection ID.
            filename: Original filename.
            detection_count: Number of detections.
            average_confidence: Average confidence.
            max_confidence: Maximum confidence.
            processing_time_ms: Processing time in milliseconds.
        """
        record = {
            "inspection_id": inspection_id,
            "timestamp": get_timestamp(),
            "status": "completed",
            "filename": filename,
            "detection_count": detection_count,
            "average_confidence": average_confidence,
            "max_confidence": max_confidence,
            "processing_time_ms": processing_time_ms
        }
        
        # Append to history file
        history_file = self.inspections_dir / "history.json"
        
        try:
            with self._history_lock:
                if history_file.exists():
                    from app.utils.file_utils import load_json
                    history = load_json(history_file)
                    if not isinstance(history, list):
                        history = []
                else:
                    history = []

                history.append(record)
                save_json(history, history_file)
        except Exception as e:
            logger.warning(f"Failed to save inspection record: {e}")
