from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")

DATASET_YAML = (
    PROJECT_ROOT
    / "datasets"
    / "aeroinspect_crack_v1"
    / "aeroinspect_crack_v1.yaml"
)

EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "exp001_yolov8n_seg"
)

RUN_NAME = "baseline"

MODEL_NAME = "yolov8n-seg.pt"

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 100

IMAGE_SIZE = 512

BATCH_SIZE = 4

DEVICE = 0

SEED = 42

WORKERS = 4

AMP = True

PATIENCE = 25

SAVE_PERIOD = 10

CACHE = False

# ============================================================
# GPU MONITORING
# ============================================================

GPU_MONITOR_INTERVAL = 5.0


# ============================================================
# DIRECTORY SETUP
# ============================================================

RUN_DIR = (
    EXPERIMENT_ROOT
    / "runs"
    / RUN_NAME
)

GPU_LOG = (
    EXPERIMENT_ROOT
    / "gpu_monitor.csv"
)

CONFIG_JSON = (
    EXPERIMENT_ROOT
    / "experiment_config.json"
)

SUMMARY_JSON = (
    EXPERIMENT_ROOT
    / "experiment_summary.json"
)


# ============================================================
# GPU INFORMATION
# ============================================================

def get_gpu_info() -> dict[str, Any]:

    info: dict[str, Any] = {}

    info["torch_version"] = torch.__version__

    info["cuda_available"] = (
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        info["cuda_version"] = (
            torch.version.cuda
        )

        info["gpu_name"] = (
            torch.cuda.get_device_name(0)
        )

        properties = (
            torch.cuda.get_device_properties(0)
        )

        info["gpu_memory_gb"] = (
            properties.total_memory
            / (1024 ** 3)
        )

        info["device"] = "cuda:0"

    else:

        info["cuda_version"] = None
        info["gpu_name"] = None
        info["gpu_memory_gb"] = None
        info["device"] = "cpu"

    return info


# ============================================================
# VERIFY CUDA
# ============================================================

def verify_cuda() -> None:

    print()
    print("=" * 78)
    print("CUDA / GPU VERIFICATION")
    print("=" * 78)

    gpu_info = get_gpu_info()

    print(
        f"PyTorch        : "
        f"{gpu_info['torch_version']}"
    )

    print(
        f"CUDA available : "
        f"{gpu_info['cuda_available']}"
    )

    if not gpu_info["cuda_available"]:

        raise RuntimeError(
            "CUDA is unavailable. "
            "Training will NOT continue on CPU."
        )

    print(
        f"CUDA version   : "
        f"{gpu_info['cuda_version']}"
    )

    print(
        f"GPU            : "
        f"{gpu_info['gpu_name']}"
    )

    print(
        f"VRAM           : "
        f"{gpu_info['gpu_memory_gb']:.2f} GB"
    )

    print(
        f"Device         : "
        f"{gpu_info['device']}"
    )

    # Real CUDA computation.
    tensor = torch.randn(
        2048,
        2048,
        device="cuda:0",
    )

    result = (
        tensor @ tensor.T
    ).mean()

    torch.cuda.synchronize()

    print(
        f"CUDA computation: "
        f"{result.item():.6f}"
    )

    del tensor

    torch.cuda.empty_cache()

    print(
        "GPU STATUS: READY"
    )


# ============================================================
# NVIDIA-SMI MONITOR
# ============================================================

def read_nvidia_smi() -> dict[str, Any] | None:

    command = [
        "nvidia-smi",
        "--query-gpu="
        "timestamp,"
        "memory.used,"
        "memory.total,"
        "utilization.gpu,"
        "temperature.gpu,"
        "power.draw",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]

    try:

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

        line = completed.stdout.strip()

        if not line:
            return None

        values = [
            value.strip()
            for value in line.split(",")
        ]

        if len(values) != 6:
            return None

        return {
            "timestamp": values[0],
            "memory_used_mb": float(
                values[1]
            ),
            "memory_total_mb": float(
                values[2]
            ),
            "gpu_utilization_percent": float(
                values[3]
            ),
            "temperature_c": float(
                values[4]
            ),
            "power_w": float(
                values[5]
            ),
        }

    except Exception:
        return None


# ============================================================
# GPU MONITOR THREAD
# ============================================================

class GPUMonitor:

    def __init__(
        self,
        output_file: Path,
        interval: float = 5.0,
    ) -> None:

        self.output_file = output_file
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "memory_used_mb",
                "memory_total_mb",
                "gpu_utilization_percent",
                "temperature_c",
                "power_w",
            ])

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def _run(self) -> None:

        while not self.stop_event.is_set():

            sample = read_nvidia_smi()

            if sample is not None:

                with self.output_file.open(
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        sample["timestamp"],
                        sample["memory_used_mb"],
                        sample["memory_total_mb"],
                        sample[
                            "gpu_utilization_percent"
                        ],
                        sample["temperature_c"],
                        sample["power_w"],
                    ])

            self.stop_event.wait(
                self.interval
            )

    def stop(self) -> None:

        self.stop_event.set()

        if self.thread is not None:

            self.thread.join(
                timeout=10
            )


# ============================================================
# CONFIGURATION LOGGING
# ============================================================

def save_configuration(
    gpu_info: dict[str, Any],
) -> None:

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "experiment": "Experiment 001",
        "description": (
            "AeroInspect AI baseline "
            "single-class concrete crack "
            "instance segmentation"
        ),
        "created_at": (
            datetime.now().isoformat()
        ),
        "model": MODEL_NAME,
        "dataset_yaml": str(
            DATASET_YAML
        ),
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "seed": SEED,
        "workers": WORKERS,
        "amp": AMP,
        "patience": PATIENCE,
        "save_period": SAVE_PERIOD,
        "cache": CACHE,
        "gpu": gpu_info,
    }

    with CONFIG_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )


# ============================================================
# MAIN TRAINING
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "AEROINSPECT AI — EXPERIMENT 001"
    )
    print("YOLOv8n-seg Baseline Training")
    print("=" * 78)

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    if not DATASET_YAML.exists():

        raise FileNotFoundError(
            f"Dataset YAML not found:\n"
            f"{DATASET_YAML}"
        )

    # --------------------------------------------------------
    # Verify CUDA
    # --------------------------------------------------------

    verify_cuda()

    gpu_info = get_gpu_info()

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    save_configuration(
        gpu_info
    )

    print()
    print("=" * 78)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 78)

    print(
        f"Model          : {MODEL_NAME}"
    )

    print(
        f"Dataset        : {DATASET_YAML}"
    )

    print(
        f"Epochs         : {EPOCHS}"
    )

    print(
        f"Image size     : {IMAGE_SIZE}"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Device         : CUDA:{DEVICE}"
    )

    print(
        f"AMP            : {AMP}"
    )

    print(
        f"Seed           : {SEED}"
    )

    print(
        f"Workers        : {WORKERS}"
    )

    print(
        f"Early stopping : {PATIENCE}"
    )

    print(
        f"Checkpoint      : every {SAVE_PERIOD} epochs"
    )

    # --------------------------------------------------------
    # Load pretrained segmentation model
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LOADING MODEL")
    print("=" * 78)

    model = YOLO(
        MODEL_NAME
    )

    print(
        f"Loaded: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Start GPU monitoring
    # --------------------------------------------------------

    gpu_monitor = GPUMonitor(
        output_file=GPU_LOG,
        interval=GPU_MONITOR_INTERVAL,
    )

    gpu_monitor.start()

    print(
        f"\nGPU monitoring:"
        f"\n  {GPU_LOG}"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    try:

        print()
        print("=" * 78)
        print("STARTING TRAINING")
        print("=" * 78)

        results = model.train(

            data=str(
                DATASET_YAML
            ),

            epochs=EPOCHS,

            imgsz=IMAGE_SIZE,

            batch=BATCH_SIZE,

            device=DEVICE,

            workers=WORKERS,

            amp=AMP,

            seed=SEED,

            deterministic=True,

            pretrained=True,

            patience=PATIENCE,

            save=True,

            save_period=SAVE_PERIOD,

            cache=CACHE,

            project=str(
                RUN_DIR.parent
            ),

            name=RUN_DIR.name,

            exist_ok=False,

            plots=True,

            verbose=True,
        )

    finally:

        gpu_monitor.stop()

    training_seconds = (
        time.time()
        - start_time
    )

    training_hours = (
        training_seconds
        / 3600
    )

    # --------------------------------------------------------
    # Locate best model
    # --------------------------------------------------------

    best_model = (
        RUN_DIR
        / "weights"
        / "best.pt"
    )

    last_model = (
        RUN_DIR
        / "weights"
        / "last.pt"
    )

    # --------------------------------------------------------
    # Post-training validation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("POST-TRAINING VALIDATION")
    print("=" * 78)

    if not best_model.exists():

        raise FileNotFoundError(
            f"Best model was not found:\n"
            f"{best_model}"
        )

    trained_model = YOLO(
        str(best_model)
    )

    print(
        "\nRunning validation on "
        "the validation set..."
    )

    validation_results = (
        trained_model.val(
            data=str(
                DATASET_YAML
            ),
            split="val",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            plots=True,
            verbose=True,
        )
    )

    # --------------------------------------------------------
    # FINAL TEST SET EVALUATION
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The test set is evaluated only after training.
    # It is NOT used to select epochs or hyperparameters.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL TEST SET EVALUATION")
    print("=" * 78)

    test_results = (
        trained_model.val(
            data=str(
                DATASET_YAML
            ),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            plots=True,
            verbose=True,
        )
    )

    # --------------------------------------------------------
    # GPU final state
    # --------------------------------------------------------

    final_gpu_state = (
        read_nvidia_smi()
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "experiment": "Experiment 001",
        "completed_at": (
            datetime.now().isoformat()
        ),
        "model": MODEL_NAME,
        "dataset": str(
            DATASET_YAML
        ),
        "training_seconds": (
            training_seconds
        ),
        "training_hours": (
            training_hours
        ),
        "best_model": str(
            best_model
        ),
        "last_model": str(
            last_model
        ),
        "gpu": gpu_info,
        "final_gpu_state": final_gpu_state,
        "validation_run_directory": (
            str(
                RUN_DIR
            )
        ),
        "test_evaluation_completed": True,
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPERIMENT 001 COMPLETE")
    print("=" * 78)

    print(
        f"Training time: "
        f"{training_hours:.2f} hours"
    )

    print(
        f"\nBest model:"
        f"\n  {best_model}"
    )

    print(
        f"\nLast model:"
        f"\n  {last_model}"
    )

    print(
        f"\nRun directory:"
        f"\n  {RUN_DIR}"
    )

    print(
        f"\nGPU log:"
        f"\n  {GPU_LOG}"
    )

    print(
        f"\nExperiment summary:"
        f"\n  {SUMMARY_JSON}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Validation metrics are for model development."
    )

    print(
        "Test metrics are the final held-out evaluation."
    )

    print(
        "Do NOT change the model based on the test results."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()