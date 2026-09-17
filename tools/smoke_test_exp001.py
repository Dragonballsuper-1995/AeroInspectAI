from pathlib import Path
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")

DATASET_YAML = (
    PROJECT_ROOT
    / "datasets"
    / "aeroinspect_crack_v1"
    / "aeroinspect_crack_v1.yaml"
)


def main() -> None:

    print("CUDA:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

    model = YOLO(
        "yolov8n-seg.pt"
    )

    model.train(
        data=str(DATASET_YAML),
        epochs=1,
        imgsz=512,
        batch=2,
        device=0,
        workers=2,
        amp=True,
        seed=42,
        deterministic=True,
        pretrained=True,
        cache=False,
        project=str(
            PROJECT_ROOT
            / "experiments"
            / "exp001_smoke_test"
        ),
        name="gpu_check",
        plots=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()