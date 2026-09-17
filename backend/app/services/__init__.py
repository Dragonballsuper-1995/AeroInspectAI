"""Inference service for YOLO model."""

import time
import threading
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
from app.core.logging_config import logger
from app.core.config import settings


class InferenceService:
    """Service for managing YOLO model and performing inference."""
    
    def __init__(self):
        """Initialize inference service."""
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.model_path: Optional[Path] = None
        self.classes: dict[int, str] = {}
        self._predict_lock = threading.Lock()
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load YOLO model at startup.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            model_path = settings.get_model_path()
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
            self.model_path = model_path
            
            # Determine device
            if (
                torch.cuda.is_available()
                and settings.cuda_enabled
                and settings.device >= 0
            ):
                self.device = f"cuda:{settings.device}"
                logger.info(
                    f"CUDA available: {torch.cuda.get_device_name(settings.device)}"
                )
            else:
                self.device = "cpu"
                logger.warning("CUDA not available, using CPU")
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.classes = self.model.names
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            logger.info(f"Classes: {self.classes}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_device_name(self) -> str:
        """Get GPU device name if available."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(settings.device)
        except Exception:
            pass
        return "CPU"
    
    def predict(
        self,
        image_path: Path,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> Tuple[dict, float]:
        """
        Run inference on image.
        
        Args:
            image_path: Path to image file.
            confidence: Confidence threshold (overrides config if provided).
            iou: IoU threshold (overrides config if provided).
            imgsz: Image size (overrides config if provided).
            
        Returns:
            Tuple of (results_dict, inference_time_ms).
            
        Raises:
            RuntimeError: If model not loaded.
            FileNotFoundError: If image file not found.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use config defaults if not provided
        confidence = settings.confidence_threshold if confidence is None else confidence
        iou = settings.iou_threshold if iou is None else iou
        imgsz = settings.image_size if imgsz is None else imgsz
        
        try:
            logger.info(
                f"Running inference: image={image_path.name}, "
                f"conf={confidence}, iou={iou}, imgsz={imgsz}"
            )
            
            start_time = time.time()
            
            # Run inference
            with self._predict_lock, torch.no_grad():
                results = self.model.predict(
                    source=str(image_path),
                    conf=confidence,
                    iou=iou,
                    imgsz=imgsz,
                    device=self.device,
                    verbose=False,
                    half=self.device.startswith("cuda"),
                )
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Process results
            result_dict = self._process_results(results[0], confidence, iou, imgsz)
            result_dict["inference_time_ms"] = inference_time_ms
            
            logger.info(
                f"Inference completed: {len(result_dict.get('detections', []))} "
                f"detections in {inference_time_ms:.2f}ms"
            )
            
            return result_dict, inference_time_ms
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _process_results(self, result, confidence: float, iou: float, imgsz: int) -> dict:
        """
        Process YOLO results into structured format.
        
        Args:
            result: YOLO result object.
            confidence: Confidence threshold used.
            iou: IoU threshold used.
            imgsz: Image size used.
            
        Returns:
            Structured results dictionary.
        """
        detections = []
        total_mask_area = 0
        confidences = []
        
        # Get image dimensions
        img_height, img_width = result.orig_shape if hasattr(result, 'orig_shape') else (0, 0)
        image_area = img_height * img_width
        
        # Process boxes and masks
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            
            for idx, box in enumerate(boxes):
                try:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    confidences.append(conf)
                    
                    detection = {
                        "id": idx + 1,
                        "class_id": cls_id,
                        "class_name": self.classes.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
                        "bounding_box": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "mask": None
                    }
                    
                    # Extract segmentation mask if available
                    if hasattr(result, 'masks') and result.masks is not None:
                        try:
                            import cv2
                            # Ultralytics exposes polygons in original-image coordinates.
                            # result.masks.xy returns a list or array of coordinates
                            xy_data = result.masks.xy[idx]
                            
                            # Convert to numpy array if it's a list
                            if isinstance(xy_data, list):
                                polygon_array = np.array(xy_data, dtype=np.float32)
                            else:
                                polygon_array = np.asarray(xy_data, dtype=np.float32)
                            
                            # Ensure it's 2D (Nx2)
                            if len(polygon_array.shape) == 1:
                                polygon_array = polygon_array.reshape(-1, 2)
                            
                            if len(polygon_array) >= 3:
                                # Convert to list for JSON serialization
                                polygon = polygon_array.tolist()
                                # Calculate mask area using contour area
                                mask_area = int(round(abs(cv2.contourArea(polygon_array))))
                                total_mask_area += mask_area
                                area_ratio = mask_area / image_area if image_area > 0 else 0
                                
                                detection["mask"] = {
                                    "polygon": polygon,
                                    "area_pixels": mask_area,
                                    "area_ratio": area_ratio
                                }
                        except Exception as e:
                            logger.warning(f"Failed to extract mask for detection {idx}: {e}")
                    
                    detections.append(detection)
                
                except Exception as e:
                    logger.warning(f"Error processing detection {idx}: {e}")
                    continue
        
        # Calculate summary statistics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        max_confidence = float(np.max(confidences)) if confidences else 0.0
        min_confidence = float(np.min(confidences)) if confidences else 0.0
        
        return {
            "image_width": img_width,
            "image_height": img_height,
            "detections": detections,
            "summary": {
                "detections": len(detections),
                "cracks_detected": len(detections),
                "average_confidence": avg_confidence,
                "max_confidence": max_confidence,
                "min_confidence": min_confidence,
                "total_mask_area_pixels": total_mask_area,
                "image_width": img_width,
                "image_height": img_height
            },
            "inference_params": {
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "image_size": imgsz
            }
        }
    
# Global inference service instance
inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get or create inference service singleton."""
    global inference_service
    if inference_service is None:
        inference_service = InferenceService()
    return inference_service
