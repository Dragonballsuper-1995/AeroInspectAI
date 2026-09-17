"""
AeroInspectAI — EXP002 Adaptive Hyperparameter Screening
=========================================================
Run from project root:
    python .\tools\run_exp002.py

Dataset:
    datasets\aeroinspect_crack_v1

Important:
- TEST split is never used in this script.
- Stages are adaptive: the best configuration from each stage is
  carried into the next stage.
- Every completed run is recorded in CSV + JSON.
- If interrupted, rerun with --resume to continue completed work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "datasets" / "aeroinspect_crack_v1"
DATA_YAML = DATASET / "aeroinspect_crack_v1.yaml"

EXP_ROOT = PROJECT_ROOT / "runs" / "exp002"
RUNS_ROOT = EXP_ROOT / "runs"
RESULTS_CSV = EXP_ROOT / "exp002_results.csv"
RESULTS_JSON = EXP_ROOT / "exp002_results.json"

MODEL = "yolov8n-seg.pt"
BATCH = 24
EPOCHS = 25
SEED = 42
DEVICE = 0

# ------------------------------------------------------------
# Stage definitions
# ------------------------------------------------------------
IMAGE_SIZES = [512, 640, 768]
LEARNING_RATES = [0.001, 0.003, 0.005]
WEIGHT_DECAYS = [0.0001, 0.0005, 0.001]

# Augmentation bundles. Values not listed remain at Ultralytics defaults.
AUGMENTATIONS = {
    "low": {
        "degrees": 3.0,
        "translate": 0.05,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.5,
        "mixup": 0.0,
    },
    "medium": {
        "degrees": 7.0,
        "translate": 0.10,
        "scale": 0.40,
        "shear": 2.0,
        "perspective": 0.0005,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 1.0,
        "mixup": 0.05,
    },
    "high": {
        "degrees": 12.0,
        "translate": 0.15,
        "scale": 0.50,
        "shear": 4.0,
        "perspective": 0.001,
        "fliplr": 0.5,
        "flipud": 0.10,
        "mosaic": 1.0,
        "mixup": 0.10,
    },
}

CSV_FIELDS = [
    "stage",
    "run_id",
    "parameter",
    "value",
    "imgsz",
    "lr0",
    "weight_decay",
    "augmentation",
    "epochs",
    "batch",
    "best_epoch",
    "best_f1",
    "best_precision",
    "best_recall",
    "best_box_map50",
    "best_seg_map50",
    "best_seg_map50_95",
    "status",
    "elapsed_sec",
    "run_dir",
]


def ensure_project_files() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def load_records() -> list[dict[str, Any]]:
    if not RESULTS_JSON.exists():
        return []
    try:
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_records(records: list[dict[str, Any]]) -> None:
    RESULTS_JSON.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def get_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        try:
            return float(value[0])
        except Exception:
            return None


def extract_metrics(model: YOLO, train_result: Any) -> dict[str, Any]:
    """
    Evaluate the trained model on VALIDATION ONLY and select the confidence
    threshold that maximizes segmentation F1.

    Ultralytics' standard validation metrics are used. F1 is calculated
    from the reported precision/recall at the selected confidence threshold.
    """
    best_f1 = -1.0
    best_p = None
    best_r = None
    best_map50 = None
    best_map5095 = None
    best_epoch = None

    # First attempt: use final validation object supplied by training.
    try:
        validator = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=best_imgsz_from_model(model, train_result),
            batch=BATCH,
            device=DEVICE,
            plots=False,
            verbose=False,
        )

        box = getattr(validator, "box", None)
        seg = getattr(validator, "seg", None)

        p = get_scalar(getattr(seg, "p", None))
        r = get_scalar(getattr(seg, "r", None))

        # Some Ultralytics versions expose arrays instead of scalars.
        if p is None or r is None:
            p_arr = getattr(seg, "p", None)
            r_arr = getattr(seg, "r", None)
            try:
                import numpy as np
                p_arr = np.asarray(p_arr).reshape(-1)
                r_arr = np.asarray(r_arr).reshape(-1)
                f1_arr = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-12)
                i = int(f1_arr.argmax())
                p = float(p_arr[i])
                r = float(r_arr[i])
                best_f1 = float(f1_arr[i])
            except Exception:
                pass

        if best_f1 < 0 and p is not None and r is not None:
            best_f1 = 2 * p * r / (p + r + 1e-12)

        best_p = p
        best_r = r

        if box is not None:
            best_map50 = get_scalar(getattr(box, "map50", None))
        if seg is not None:
            best_map5095 = get_scalar(getattr(seg, "map", None))
            best_map50 = get_scalar(getattr(seg, "map50", None)) or best_map50

    except Exception as exc:
        print(f"[WARN] Validation metric extraction failed: {exc}")

    # Best epoch from training history, where available.
    try:
        results_csv = Path(train_result.save_dir) / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]

            seg_p_cols = [c for c in df.columns if "metrics/precision(M)" in c]
            seg_r_cols = [c for c in df.columns if "metrics/recall(M)" in c]

            if seg_p_cols and seg_r_cols:
                p_series = df[seg_p_cols[0]]
                r_series = df[seg_r_cols[0]]
                f1_series = 2 * p_series * r_series / (p_series + r_series + 1e-12)
                i = int(f1_series.idxmax())

                if best_f1 < 0 or float(f1_series.loc[i]) > best_f1:
                    best_f1 = float(f1_series.loc[i])
                    best_p = float(p_series.loc[i])
                    best_r = float(r_series.loc[i])

                best_epoch = int(df.loc[i, "epoch"]) + 1
                if "metrics/mAP50(M)" in df.columns:
                    best_map50 = float(df.loc[i, "metrics/mAP50(M)"])
                if "metrics/mAP50-95(M)" in df.columns:
                    best_map5095 = float(df.loc[i, "metrics/mAP50-95(M)"])
    except Exception as exc:
        print(f"[WARN] Could not read training history: {exc}")

    return {
        "best_epoch": best_epoch,
        "best_f1": best_f1 if best_f1 >= 0 else None,
        "best_precision": best_p,
        "best_recall": best_r,
        "best_box_map50": best_map50,
        "best_seg_map50": best_map50,
        "best_seg_map50_95": best_map5095,
    }


def best_imgsz_from_model(model: YOLO, train_result: Any) -> int:
    # The current experiment configuration is stored in args.
    try:
        return int(train_result.args["imgsz"])
    except Exception:
        try:
            return int(getattr(model, "overrides", {}).get("imgsz", 640))
        except Exception:
            return 640


def run_one(
    stage: str,
    run_id: str,
    parameter: str,
    value: Any,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    if resume:
        for r in records:
            if r["run_id"] == run_id and r["status"] == "completed":
                print(f"[SKIP] {run_id} already completed.")
                return r

    print("\n" + "=" * 72)
    print(f"RUN {run_id}")
    print(f"Stage       : {stage}")
    print(f"Parameter   : {parameter} = {value}")
    print(f"imgsz       : {config['imgsz']}")
    print(f"lr0         : {config['lr0']}")
    print(f"weight_decay: {config['weight_decay']}")
    print(f"augmentation: {config['augmentation']}")
    print(f"batch       : {BATCH}")
    print("=" * 72)

    run_dir = RUNS_ROOT / run_id
    t0 = time.time()

    try:
        model = YOLO(MODEL)

        aug = AUGMENTATIONS[config["augmentation"]]

        train_kwargs = dict(
            data=str(DATA_YAML),
            imgsz=config["imgsz"],
            epochs=EPOCHS,
            batch=BATCH,
            optimizer="AdamW",
            lr0=config["lr0"],
            weight_decay=config["weight_decay"],
            device=DEVICE,
            seed=SEED,
            deterministic=True,
            project=str(RUNS_ROOT),
            name=run_id,
            exist_ok=True,
            pretrained=True,
            verbose=True,
            patience=10,
            workers=4,
            cache=False,
            plots=True,
            val=True,
            # Never use test split during training.
            split="val",
        )
        train_kwargs.update(aug)

        result = model.train(**train_kwargs)
        metrics = extract_metrics(model, result)

        row = {
            "stage": stage,
            "run_id": run_id,
            "parameter": parameter,
            "value": value,
            "imgsz": config["imgsz"],
            "lr0": config["lr0"],
            "weight_decay": config["weight_decay"],
            "augmentation": config["augmentation"],
            "epochs": EPOCHS,
            "batch": BATCH,
            **metrics,
            "status": "completed",
            "elapsed_sec": round(time.time() - t0, 2),
            "run_dir": str(run_dir),
        }

        records[:] = [r for r in records if r["run_id"] != run_id]
        records.append(row)
        save_records(records)

        print(
            f"[DONE] {run_id}: "
            f"F1={row['best_f1']}, "
            f"P={row['best_precision']}, "
            f"R={row['best_recall']}, "
            f"mAP50(seg)={row['best_seg_map50']}"
        )

        return row

    except Exception as exc:
        row = {
            "stage": stage,
            "run_id": run_id,
            "parameter": parameter,
            "value": value,
            "imgsz": config["imgsz"],
            "lr0": config["lr0"],
            "weight_decay": config["weight_decay"],
            "augmentation": config["augmentation"],
            "epochs": EPOCHS,
            "batch": BATCH,
            "status": f"failed: {type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - t0, 2),
            "run_dir": str(run_dir),
        }
        records[:] = [r for r in records if r["run_id"] != run_id]
        records.append(row)
        save_records(records)

        print(f"[FAILED] {run_id}: {exc}")
        raise


def completed_rows(records: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [
        r for r in records
        if r.get("stage") == stage
        and r.get("status") == "completed"
        and r.get("best_f1") is not None
    ]


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No successful runs available for stage selection.")
    return max(rows, key=lambda r: float(r["best_f1"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_project_files()
    records = load_records()

    print("AeroInspectAI EXP002")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset YAML : {DATA_YAML}")
    print(f"GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    print(f"Batch        : {BATCH}")
    print(f"Epochs       : {EPOCHS}")
    print("\nTEST SPLIT IS NOT USED FOR MODEL SELECTION.")

    # --------------------------------------------------------
    # STAGE A — image size
    # --------------------------------------------------------
    stage = "A_imgsz"
    rows = completed_rows(records, stage)

    if len(rows) < len(IMAGE_SIZES):
        for imgsz in IMAGE_SIZES:
            run_id = f"A_imgsz_{imgsz}"
            config = {
                "imgsz": imgsz,
                "lr0": 0.003,
                "weight_decay": 0.0005,
                "augmentation": "medium",
            }
            run_one(stage, run_id, "imgsz", imgsz, config, records, args.resume)
            rows = completed_rows(records, stage)
            if len(rows) == len(IMAGE_SIZES):
                break

    best_a = choose_best(completed_rows(records, stage))
    best_imgsz = int(best_a["imgsz"])
    print(f"\nBEST STAGE A: imgsz={best_imgsz}, F1={best_a['best_f1']}")

    # --------------------------------------------------------
    # STAGE B — learning rate
    # --------------------------------------------------------
    stage = "B_lr"
    rows = completed_rows(records, stage)

    if len(rows) < len(LEARNING_RATES):
        for lr in LEARNING_RATES:
            run_id = f"B_lr_{str(lr).replace('.', 'p')}"
            config = {
                "imgsz": best_imgsz,
                "lr0": lr,
                "weight_decay": 0.0005,
                "augmentation": "medium",
            }
            run_one(stage, run_id, "lr0", lr, config, records, args.resume)
            rows = completed_rows(records, stage)
            if len(rows) == len(LEARNING_RATES):
                break

    best_b = choose_best(completed_rows(records, stage))
    best_lr = float(best_b["lr0"])
    print(f"\nBEST STAGE B: lr0={best_lr}, F1={best_b['best_f1']}")

    # --------------------------------------------------------
    # STAGE C — augmentation
    # --------------------------------------------------------
    stage = "C_augmentation"
    rows = completed_rows(records, stage)

    if len(rows) < len(AUGMENTATIONS):
        for aug_name in AUGMENTATIONS:
            run_id = f"C_aug_{aug_name}"
            config = {
                "imgsz": best_imgsz,
                "lr0": best_lr,
                "weight_decay": 0.0005,
                "augmentation": aug_name,
            }
            run_one(stage, run_id, "augmentation", aug_name, config, records, args.resume)
            rows = completed_rows(records, stage)
            if len(rows) == len(AUGMENTATIONS):
                break

    best_c = choose_best(completed_rows(records, stage))
    best_aug = best_c["augmentation"]
    print(f"\nBEST STAGE C: augmentation={best_aug}, F1={best_c['best_f1']}")

    # --------------------------------------------------------
    # STAGE D — weight decay
    # --------------------------------------------------------
    stage = "D_weight_decay"
    rows = completed_rows(records, stage)

    if len(rows) < len(WEIGHT_DECAYS):
        for wd in WEIGHT_DECAYS:
            run_id = f"D_wd_{str(wd).replace('.', 'p')}"
            config = {
                "imgsz": best_imgsz,
                "lr0": best_lr,
                "weight_decay": wd,
                "augmentation": best_aug,
            }
            run_one(stage, run_id, "weight_decay", wd, config, records, args.resume)
            rows = completed_rows(records, stage)
            if len(rows) == len(WEIGHT_DECAYS):
                break

    best_d = choose_best(completed_rows(records, stage))

    # --------------------------------------------------------
    # FINAL EXP002 SCREENING RESULT
    # --------------------------------------------------------
    summary = {
        "experiment": "EXP002",
        "best_configuration": {
            "imgsz": int(best_d["imgsz"]),
            "lr0": float(best_d["lr0"]),
            "weight_decay": float(best_d["weight_decay"]),
            "augmentation": best_d["augmentation"],
            "batch": BATCH,
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "model": MODEL,
        },
        "validation_metrics": {
            "best_f1": best_d["best_f1"],
            "precision": best_d["best_precision"],
            "recall": best_d["best_recall"],
            "seg_map50": best_d["best_seg_map50"],
            "seg_map50_95": best_d["best_seg_map50_95"],
            "best_epoch": best_d["best_epoch"],
        },
        "test_set_used": False,
        "dataset_yaml": str(DATA_YAML),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    (EXP_ROOT / "EXP002_FINAL.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("EXP002 COMPLETE")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nResults CSV : {RESULTS_CSV}")
    print(f"Results JSON: {RESULTS_JSON}")
    print(f"Final config: {EXP_ROOT / 'EXP002_FINAL.json'}")


if __name__ == "__main__":
    main()
