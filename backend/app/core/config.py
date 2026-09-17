"""Configuration management using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "AeroInspectAI Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # Model configuration
    model_path: str = str(
        PROJECT_ROOT / "experiments" / "exp001_yolov8n_seg" / "runs" /
        "baseline" / "weights" / "best.pt"
    )
    dataset_yaml: str = str(
        PROJECT_ROOT / "datasets" / "aeroinspect_crack_v1" /
        "aeroinspect_crack_v1.yaml"
    )

    # Device configuration
    device: int = 0  # GPU device ID, or -1 for CPU
    cuda_enabled: bool = True

    # Inference parameters
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.70
    image_size: int = 640

    # Storage configuration
    storage_dir: str = str(BACKEND_ROOT / "storage")
    upload_dir: str = str(BACKEND_ROOT / "storage" / "uploads")
    result_dir: str = str(BACKEND_ROOT / "storage" / "results")
    inspection_dir: str = str(BACKEND_ROOT / "storage" / "inspections")

    # Upload limits
    max_upload_mb: int = 20

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Image configuration
    supported_formats: list[str] = ["jpg", "jpeg", "png", "webp"]

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        case_sensitive=False,
        env_prefix="AEROINSPECT_",
    )

    def get_model_path(self) -> Path:
        """Get model path as Path object."""
        path = Path(self.model_path)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()

    def get_dataset_yaml_path(self) -> Path:
        """Get dataset YAML path as Path object."""
        path = Path(self.dataset_yaml)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()

    def get_storage_path(self) -> Path:
        """Get storage directory as Path object."""
        return self._backend_path(self.storage_dir)

    def get_upload_path(self) -> Path:
        """Get upload directory as Path object."""
        return self._backend_path(self.upload_dir)

    def get_result_path(self) -> Path:
        """Get result directory as Path object."""
        return self._backend_path(self.result_dir)

    def get_inspection_path(self) -> Path:
        """Get inspection directory as Path object."""
        return self._backend_path(self.inspection_dir)

    @staticmethod
    def _backend_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()

    def get_max_upload_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
