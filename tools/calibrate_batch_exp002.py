"""
AeroInspect AI — EXP002 GPU Batch Calibration

Measures practical batch-size limits and throughput on the RTX 4050 before
running the EXP002 hyperparameter screening.

Safety:
- TRAIN only (small fraction).
- Never evaluates TEST.
- Does not modify the dataset.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "datasets" / "aeroinspect_crack_v1" / "aeroinspect_crack_v1.yaml"

EXP_ROOT = PROJECT_ROOT / "experiments" / "exp002_hparam_screening" / "batch_calibration"
RUN_ROOT = EXP_ROOT / "runs"
REPORT_ROOT = EXP_ROOT / "reports"

CSV_PATH = REPORT_ROOT / "batch_calibration.csv"
JSON_PATH = REPORT_ROOT / "batch_calibration.json"
RECOMMENDATION_PATH = REPORT_ROOT / "batch_recommendation.json"

MODEL = "yolov8n-seg.pt"
IMAGE_SIZES = [512, 640, 768]
BATCH_CANDIDATES = [4, 8, 12, 16, 24, 32, 48]

# Small, fast calibration rather than a quality experiment.
FRACTION = 0.15
EPOCHS = 2

DEVICE = 0
WORKERS = 2
SEED = 42


def gpu_info():
    props = torch.cuda.get_device_properties(DEVICE)
    return {
        "gpu": torch.cuda.get_device_name(DEVICE),
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
    }


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize(DEVICE)


def memory_info():
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(DEVICE) / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(DEVICE) / 1024**3, 3),
    }


def is_oom(exc):
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text


def run_one(imgsz, batch):
    run_id = f"img{imgsz}_batch{batch}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 72)
    print(f"Calibration: {run_id}")
    print("-" * 72)

    clear_gpu()
    start = time.perf_counter()

    row = {
        "run_id": run_id,
        "imgsz": imgsz,
        "batch": batch,
        "fraction": FRACTION,
        "epochs": EPOCHS,
        "status": "started",
        "oom": False,
        "error": None,
        "elapsed_seconds": None,
        "approx_images_per_second": None,
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
    }

    try:
        model = YOLO(MODEL)

        model.train(
            data=str(DATA_YAML),
            task="segment",
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            fraction=FRACTION,
            workers=WORKERS,
            device=DEVICE,
            seed=SEED,
            deterministic=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.003,
            lrf=0.01,
            weight_decay=0.0005,
            momentum=0.9,
            patience=EPOCHS,
            cache=False,
            val=False,
            plots=False,
            save=False,
            verbose=False,
            project=str(RUN_ROOT),
            name=run_id,
            exist_ok=True,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
        )

        torch.cuda.synchronize(DEVICE)
        elapsed = time.perf_counter() - start
        mem = memory_info()

        approx_images = round(3717 * FRACTION) * EPOCHS
        ips = approx_images / elapsed

        row.update(
            status="success",
            elapsed_seconds=round(elapsed, 2),
            approx_images_per_second=round(ips, 2),
            **mem,
        )

        print(f"SUCCESS | {elapsed:.1f}s | ~{ips:.1f} img/s | peak {mem['peak_allocated_gb']:.3f} GB")

        del model
        clear_gpu()
        return row

    except Exception as exc:
        elapsed = time.perf_counter() - start
        mem = memory_info()

        row.update(
            status="oom" if is_oom(exc) else "failed",
            oom=is_oom(exc),
            error=repr(exc),
            elapsed_seconds=round(elapsed, 2),
            **mem,
        )

        print(f"{row['status'].upper()} | peak {mem.get('peak_allocated_gb')} GB")
        clear_gpu()
        return row


def save(rows):
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    fields = sorted({k for r in rows for k in r})
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def make_recommendation(rows):
    # Keep ~1 GB headroom on the 6 GB card.
    result = []

    for imgsz in IMAGE_SIZES:
        successful = [
            r for r in rows
            if r["imgsz"] == imgsz
            and r["status"] == "success"
            and r["peak_allocated_gb"] is not None
        ]

        safe = [r for r in successful if r["peak_allocated_gb"] <= 5.0]

        if safe:
            best = max(safe, key=lambda r: r["batch"])
            reason = "largest successful batch with <=5 GB peak allocation"
        elif successful:
            best = max(successful, key=lambda r: r["batch"])
            reason = "largest successful batch, but low memory headroom"
        else:
            best = None
            reason = "no successful batch"

        result.append({
            "imgsz": imgsz,
            "recommended_batch": best["batch"] if best else None,
            "peak_allocated_gb": best["peak_allocated_gb"] if best else None,
            "peak_reserved_gb": best["peak_reserved_gb"] if best else None,
            "images_per_second": best["approx_images_per_second"] if best else None,
            "reason": reason,
        })

    batches = [r["recommended_batch"] for r in result if r["recommended_batch"]]
    common = min(batches) if batches else None

    recommendation = {
        "per_resolution": result,
        "conservative_common_batch": common,
        "preferred_peak_allocation_gb": 5.0,
        "physical_vram_gb": 6.0,
        "test_split_evaluated": False,
        "note": (
            "Use the common batch for the initial EXP002 resolution comparison "
            "so batch size does not become a confounding variable."
        ),
    }

    RECOMMENDATION_PATH.write_text(
        json.dumps(recommendation, indent=2),
        encoding="utf-8",
    )
    return recommendation


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    info = gpu_info()

    print("=" * 72)
    print("AEROINSPECT AI — EXP002 GPU BATCH CALIBRATION")
    print("=" * 72)
    print(f"GPU : {info['gpu']}")
    print(f"VRAM: {info['vram_gb']} GB")
    print(f"PyTorch: {info['torch']}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Image sizes: {IMAGE_SIZES}")
    print(f"Batch candidates: {BATCH_CANDIDATES}")
    print(f"Fraction: {FRACTION}")
    print(f"Epochs: {EPOCHS}")
    print("\nTEST SAFETY: split=test is NEVER used.")

    rows = []

    for imgsz in IMAGE_SIZES:
        for batch in BATCH_CANDIDATES:
            row = run_one(imgsz, batch)
            rows.append(row)
            save(rows)

            # Batch sizes are increasing; once OOM occurs, larger batches at
            # this resolution are not useful.
            if row["status"] == "oom":
                print(f"Skipping larger batches at {imgsz}px after OOM.")
                break

    recommendation = make_recommendation(rows)

    print("\n" + "=" * 72)
    print("CALIBRATION COMPLETE")
    print("=" * 72)

    for item in recommendation["per_resolution"]:
        print(
            f"{item['imgsz']}px -> batch {item['recommended_batch']} | "
            f"peak {item['peak_allocated_gb']} GB | {item['reason']}"
        )

    print(
        f"\nCONSERVATIVE COMMON BATCH: "
        f"{recommendation['conservative_common_batch']}"
    )

    print(f"\nCSV: {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(f"Recommendation: {RECOMMENDATION_PATH}")


if __name__ == "__main__":
    main()
