"""File utilities."""

from pathlib import Path
import json
import uuid
from datetime import datetime
from typing import Any, Optional


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path.
        
    Returns:
        Path to directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: Path) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save.
        file_path: Output path.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Path) -> Any:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file.
        
    Returns:
        Parsed JSON data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def get_unique_inspection_id() -> str:
    """
    Generate unique inspection ID.
    
    Returns:
        Inspection ID in format INS-YYYYMMDD-XXXXXX.
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    random_suffix = uuid.uuid4().hex[:8].upper()
    return f"INS-{date_str}-{random_suffix}"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO format timestamp.
    """
    return datetime.now().isoformat()


def is_safe_path(base_path: Path, target_path: Path) -> bool:
    """
    Check if target path is within base path (prevent traversal attacks).
    
    Args:
        base_path: Base directory.
        target_path: Target path to check.
        
    Returns:
        True if target is within base, False otherwise.
    """
    try:
        target_path.resolve().relative_to(base_path.resolve())
        return True
    except ValueError:
        return False


def cleanup_old_inspections(inspections_dir: Path, max_age_days: int = 30) -> None:
    """
    Clean up old inspection records and files.
    
    Args:
        inspections_dir: Directory containing inspections.
        max_age_days: Maximum age in days before deletion.
    """
    from datetime import timedelta, timezone
    
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    
    if not inspections_dir.exists():
        return
    
    for inspection_dir in inspections_dir.iterdir():
        if not inspection_dir.is_dir():
            continue
        
        try:
            mtime = datetime.fromtimestamp(
                inspection_dir.stat().st_mtime,
                tz=timezone.utc
            )
            if mtime < cutoff_time:
                # Remove directory and contents
                import shutil
                shutil.rmtree(inspection_dir)
        except Exception:
            pass
