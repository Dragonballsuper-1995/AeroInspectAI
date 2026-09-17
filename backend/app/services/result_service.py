"""Result service for retrieving inspection results."""

from pathlib import Path
from typing import Optional, List
from app.core.config import settings
from app.core.logging_config import logger
from app.utils.file_utils import load_json, is_safe_path
from app.schemas import InspectionRecord


class ResultService:
    """Service for managing and retrieving inspection results."""
    
    def __init__(self):
        """Initialize result service."""
        self.inspections_dir = settings.get_inspection_path()
    
    def get_inspection_result(self, inspection_id: str) -> dict:
        """
        Get complete inspection result.
        
        Args:
            inspection_id: Inspection ID.
            
        Returns:
            Inspection result dictionary.
            
        Raises:
            FileNotFoundError: If inspection not found.
        """
        inspection_dir = self.inspections_dir / inspection_id
        result_file = inspection_dir / "result.json"
        
        if not is_safe_path(self.inspections_dir, result_file):
            raise ValueError(f"Invalid inspection ID: {inspection_id}")
        
        if not result_file.exists():
            raise FileNotFoundError(f"Inspection not found: {inspection_id}")
        
        try:
            return load_json(result_file)
        except Exception as e:
            logger.error(f"Failed to load inspection result: {e}")
            raise
    
    def get_original_image_path(self, inspection_id: str) -> Path:
        """
        Get path to original image.
        
        Args:
            inspection_id: Inspection ID.
            
        Returns:
            Path to original image.
            
        Raises:
            FileNotFoundError: If image not found.
        """
        image_path = self.inspections_dir / inspection_id / "original.jpg"
        
        if not is_safe_path(self.inspections_dir, image_path):
            raise ValueError(f"Invalid inspection ID: {inspection_id}")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Original image not found: {inspection_id}")
        
        return image_path
    
    def get_annotated_image_path(self, inspection_id: str) -> Path:
        """
        Get path to annotated image.
        
        Args:
            inspection_id: Inspection ID.
            
        Returns:
            Path to annotated image.
            
        Raises:
            FileNotFoundError: If image not found.
        """
        image_path = self.inspections_dir / inspection_id / "annotated.jpg"
        
        if not is_safe_path(self.inspections_dir, image_path):
            raise ValueError(f"Invalid inspection ID: {inspection_id}")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Annotated image not found: {inspection_id}")
        
        return image_path
    
    def get_inspections_history(self, limit: Optional[int] = 100) -> List[InspectionRecord]:
        """
        Get inspection history.
        
        Args:
            limit: Maximum number of records to return.
            
        Returns:
            List of inspection records.
        """
        history_file = self.inspections_dir / "history.json"
        
        if not history_file.exists():
            return []
        
        try:
            history = load_json(history_file)
            if not isinstance(history, list):
                return []
            
            # Apply limit
            if limit:
                history = history[-limit:]
            
            # Reverse to show newest first
            return list(reversed(history))
        
        except Exception as e:
            logger.error(f"Failed to load inspection history: {e}")
            return []
    
    def inspection_exists(self, inspection_id: str) -> bool:
        """
        Check if inspection exists.
        
        Args:
            inspection_id: Inspection ID.
            
        Returns:
            True if inspection exists, False otherwise.
        """
        result_file = self.inspections_dir / inspection_id / "result.json"
        return is_safe_path(self.inspections_dir, result_file) and result_file.exists()
