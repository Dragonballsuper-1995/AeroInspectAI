"""Image utilities."""

from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def is_valid_image(file_path: Path) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        file_path: Path to image file.
        
    Returns:
        True if valid image, False otherwise.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    if file_path.suffix.lower() not in supported_extensions:
        return False
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_dimensions(file_path: Path) -> Tuple[int, int]:
    """
    Get image width and height.
    
    Args:
        file_path: Path to image file.
        
    Returns:
        Tuple of (width, height).
    """
    with Image.open(file_path) as img:
        return img.size


def load_image(file_path: Path) -> np.ndarray:
    """
    Load image as numpy array (BGR format for OpenCV).
    
    Args:
        file_path: Path to image file.
        
    Returns:
        Image as numpy array in BGR format.
    """
    return cv2.imread(str(file_path))


def save_image(image: np.ndarray, file_path: Path) -> None:
    """
    Save numpy array as image.
    
    Args:
        image: Image as numpy array.
        file_path: Output path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(file_path), image)


def draw_segmentation_results(
    image: np.ndarray,
    results: list[dict],
    thickness: int = 2,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw detection boxes and segmentation masks on image.
    
    Args:
        image: Original image as numpy array.
        results: List of detection results.
        thickness: Line thickness for boxes.
        alpha: Transparency for mask overlay.
        
    Returns:
        Annotated image as numpy array (uint8).
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    annotated = image.copy()
    colors = [
        (0, 255, 0),      # Green for crack
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
    ]
    
    for idx, det in enumerate(results):
        color = colors[idx % len(colors)]
        
        # Draw segmentation mask if available
        if det.get("mask") and det["mask"].get("polygon"):
            polygon_data = det["mask"]["polygon"]
            # Ensure polygon is proper format
            if polygon_data and isinstance(polygon_data, list):
                polygon = np.array(polygon_data, dtype=np.int32)
                if polygon.size > 0 and len(polygon.shape) == 2 and polygon.shape[0] >= 3:
                    # Draw polygon outline
                    cv2.polylines(annotated, [polygon], True, color, thickness)
                    # Semi-transparent fill using overlay
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [polygon], color)
                    annotated = cv2.addWeighted(annotated, 1 - alpha, overlay, alpha, 0)
        
        # Draw bounding box
        if det.get("bounding_box"):
            bbox = det["bounding_box"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with confidence
            conf = det.get("confidence", 0.0)
            label = f"{det.get('class_name', 'unknown')} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            # Background rectangle for text
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 4),
                (x1 + text_width, y1),
                color,
                -1
            )
            # Text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 2),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness
            )
    
    # Ensure output is uint8
    return annotated if annotated.dtype == np.uint8 else annotated.astype(np.uint8)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing/replacing unsafe characters.
    
    Args:
        filename: Original filename.
        max_length: Maximum filename length.
        
    Returns:
        Sanitized filename.
    """
    import re
    
    # Remove unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # Truncate if needed
    if len(sanitized) > max_length:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        sanitized = name[: max_length - len(ext)] + ext
    
    return sanitized or "image"
