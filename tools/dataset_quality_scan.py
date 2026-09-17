from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")

RAW_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "crack-seg"
)

CLEAN_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "aeroinspect_crack_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "audit_results"
    / "quality_scan"
)

REPORT_DIR = OUTPUT_ROOT / "reports"
SAMPLE_DIR = OUTPUT_ROOT / "samples"

RANDOM_SEED = 42

# ------------------------------------------------------------
# Suspicious thresholds
# ------------------------------------------------------------
#
# These are NOT automatic deletion thresholds.
# They are only used to flag samples for inspection.

MIN_POLYGON_AREA_RATIO = 0.00005
MAX_POLYGON_AREA_RATIO = 0.95

MIN_POLYGON_POINTS = 3

# Very long polygons can be legitimate for cracks, so this is
# only a warning threshold.
MAX_POLYGON_POINTS_WARNING = 2000

# ------------------------------------------------------------
# Known confirmed bad sample
# ------------------------------------------------------------
#
# The user manually confirmed that this image contains a visible
# crack while its label is empty.
#
# We exclude this from the CLEAN dataset.
# The RAW dataset is untouched.

KNOWN_BAD_SAMPLES = {
    "val/3513.rf.782300f38d0a008b3340e54b643e713e.jpg",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None


def relative_dataset_path(
    path: Path,
    root: Path,
) -> str:
    return str(
        path.relative_to(root)
    ).replace("\\", "/")


def safe_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


# ============================================================
# CUDA STATUS
# ============================================================

def report_cuda_status() -> None:
    print()
    print("=" * 78)
    print("COMPUTE ENVIRONMENT")
    print("=" * 78)

    try:
        import torch

        print(
            f"PyTorch version : {torch.__version__}"
        )

        print(
            f"CUDA available  : "
            f"{torch.cuda.is_available()}"
        )

        if torch.cuda.is_available():

            print(
                f"GPU             : "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                "NOTE: Annotation-quality scanning is "
                "intentionally performed on CPU."
            )

            print(
                "Reason: this workload is dominated by "
                "disk I/O, parsing and OpenCV geometry."
            )

        else:
            print("GPU             : unavailable")

    except ImportError:
        print(
            "PyTorch is not installed. "
            "CPU scan will continue."
        )


# ============================================================
# YOLO SEGMENTATION PARSER
# ============================================================

def parse_label(
    label_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:

    polygons = []
    errors = []

    if not label_path.exists():
        return polygons, [
            "missing_label"
        ]

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception as exc:

        return polygons, [
            f"label_read_error: {exc}"
        ]

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Empty annotation file.
    if not lines:
        return [], []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        parts = line.split()

        if len(parts) < 7:

            errors.append(
                f"line_{line_number}_too_few_values"
            )
            continue

        values = []

        malformed = False

        for part in parts:

            value = safe_float(part)

            if value is None:
                malformed = True
                break

            values.append(value)

        if malformed:

            errors.append(
                f"line_{line_number}_non_numeric"
            )
            continue

        # class + x/y pairs
        if (len(values) - 1) % 2 != 0:

            errors.append(
                f"line_{line_number}_odd_coordinate_count"
            )
            continue

        class_id = values[0]

        if class_id != int(class_id):

            errors.append(
                f"line_{line_number}_non_integer_class"
            )
            continue

        class_id = int(class_id)

        coordinates = values[1:]

        if len(coordinates) < 6:

            errors.append(
                f"line_{line_number}_less_than_3_points"
            )
            continue

        coords = np.asarray(
            coordinates,
            dtype=np.float64,
        ).reshape(-1, 2)

        # ----------------------------------------------------
        # Coordinate range
        # ----------------------------------------------------

        if np.any(coords < 0.0) or np.any(
            coords > 1.0
        ):

            errors.append(
                f"line_{line_number}_coordinates_outside_0_1"
            )
            continue

        polygons.append(
            {
                "line_number": line_number,
                "class_id": class_id,
                "coords": coords,
            }
        )

    return polygons, errors


# ============================================================
# POLYGON GEOMETRY
# ============================================================

def polygon_metrics(
    coords: np.ndarray,
) -> dict[str, Any]:

    points = (
        coords.astype(np.float32)
        .reshape(-1, 1, 2)
    )

    area = float(
        abs(cv2.contourArea(points))
    )

    perimeter = float(
        cv2.arcLength(
            points,
            closed=True,
        )
    )

    x_values = coords[:, 0]
    y_values = coords[:, 1]

    bbox_width = float(
        np.max(x_values) - np.min(x_values)
    )

    bbox_height = float(
        np.max(y_values) - np.min(y_values)
    )

    return {
        "point_count": int(len(coords)),
        "area_normalized": area,
        "perimeter_normalized": perimeter,
        "bbox_width_normalized": bbox_width,
        "bbox_height_normalized": bbox_height,
        "area_ratio": area,
    }


# ============================================================
# POLYGON SIGNATURE
# ============================================================

def polygon_signature(
    coords: np.ndarray,
) -> str:

    rounded = np.round(
        coords,
        decimals=8,
    )

    return hashlib.sha1(
        rounded.tobytes()
    ).hexdigest()


# ============================================================
# IMAGE-LEVEL VALIDATION
# ============================================================

def validate_image(
    image_path: Path,
    label_path: Path,
) -> dict[str, Any]:

    result = {
        "status": "VALID",
        "image": str(image_path),
        "label": str(label_path),
        "image_size": None,
        "polygon_count": 0,
        "hard_errors": [],
        "warnings": [],
        "class_ids": [],
        "polygon_metrics": [],
        "duplicate_polygon_count": 0,
        "empty_label": False,
        "known_bad_sample": False,
        "image_sha256": None,
    }

    relative_image = relative_dataset_path(
        image_path,
        RAW_DATASET,
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    size = read_image_size(
        image_path
    )

    if size is None:

        result["status"] = "EXCLUDE"

        result["hard_errors"].append(
            "unreadable_image"
        )

        return result

    width, height = size

    result["image_size"] = {
        "width": width,
        "height": height,
    }

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    try:

        result["image_sha256"] = sha256_file(
            image_path
        )

    except Exception as exc:

        result["warnings"].append(
            f"hash_error:{exc}"
        )

    # --------------------------------------------------------
    # Known bad sample
    # --------------------------------------------------------

    if relative_image in KNOWN_BAD_SAMPLES:

        result["known_bad_sample"] = True

    # --------------------------------------------------------
    # Label existence
    # --------------------------------------------------------

    if not label_path.exists():

        result["status"] = "EXCLUDE"

        result["hard_errors"].append(
            "missing_label"
        )

        return result

    polygons, parser_errors = parse_label(
        label_path
    )

    result["empty_label"] = (
        len(polygons) == 0
        and not parser_errors
    )

    if parser_errors:

        result["status"] = "EXCLUDE"

        result["hard_errors"].extend(
            parser_errors
        )

    # --------------------------------------------------------
    # Empty labels
    # --------------------------------------------------------

    if result["empty_label"]:

        if result["known_bad_sample"]:

            result["status"] = "EXCLUDE"

            result["hard_errors"].append(
                "confirmed_bad_empty_annotation"
            )

        else:

            result["warnings"].append(
                "empty_label"
            )

        return result

    # --------------------------------------------------------
    # Polygon checks
    # --------------------------------------------------------

    signatures = set()

    for polygon in polygons:

        class_id = polygon["class_id"]
        coords = polygon["coords"]

        result["class_ids"].append(
            class_id
        )

        # Only class 0 is expected in v1.
        if class_id != 0:

            result["status"] = "EXCLUDE"

            result["hard_errors"].append(
                f"unexpected_class_id_{class_id}"
            )

        # Point count.
        if len(coords) < MIN_POLYGON_POINTS:

            result["status"] = "EXCLUDE"

            result["hard_errors"].append(
                f"polygon_below_{MIN_POLYGON_POINTS}_points"
            )

        metrics = polygon_metrics(
            coords
        )

        result["polygon_metrics"].append(
            metrics
        )

        # ----------------------------------------------------
        # Degenerate polygon
        # ----------------------------------------------------

        if metrics["area_normalized"] <= 0:

            result["status"] = "EXCLUDE"

            result["hard_errors"].append(
                "degenerate_zero_area_polygon"
            )

        # ----------------------------------------------------
        # Suspiciously small polygon
        # ----------------------------------------------------

        if (
            metrics["area_ratio"]
            < MIN_POLYGON_AREA_RATIO
        ):

            result["warnings"].append(
                "very_small_polygon"
            )

        # ----------------------------------------------------
        # Suspiciously large polygon
        # ----------------------------------------------------

        if (
            metrics["area_ratio"]
            > MAX_POLYGON_AREA_RATIO
        ):

            result["warnings"].append(
                "very_large_polygon"
            )

        # ----------------------------------------------------
        # Very detailed polygon
        # ----------------------------------------------------

        if (
            metrics["point_count"]
            > MAX_POLYGON_POINTS_WARNING
        ):

            result["warnings"].append(
                "very_high_polygon_point_count"
            )

        # ----------------------------------------------------
        # Duplicate polygon
        # ----------------------------------------------------

        signature = polygon_signature(
            coords
        )

        if signature in signatures:

            result["duplicate_polygon_count"] += 1

        else:

            signatures.add(
                signature
            )

    result["polygon_count"] = len(
        polygons
    )

    if result["duplicate_polygon_count"]:

        result["warnings"].append(
            "duplicate_polygon_annotations"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if result["hard_errors"]:

        result["status"] = "EXCLUDE"

    elif result["warnings"]:

        result["status"] = "VALID_WITH_WARNINGS"

    else:

        result["status"] = "VALID"

    return result


# ============================================================
# FIND SPLITS
# ============================================================

def get_images_for_split(
    split: str,
) -> list[Path]:

    directory = (
        RAW_DATASET
        / "images"
        / split
    )

    if not directory.exists():
        return []

    return sorted(
        p
        for p in directory.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ============================================================
# CLEAN DATASET COPYING
# ============================================================

def copy_valid_sample(
    image_path: Path,
    label_path: Path,
    split: str,
) -> None:

    relative_image = image_path.relative_to(
        RAW_DATASET
        / "images"
        / split
    )

    destination_image = (
        CLEAN_DATASET
        / "images"
        / split
        / relative_image
    )

    relative_label = relative_image.with_suffix(
        ".txt"
    )

    destination_label = (
        CLEAN_DATASET
        / "labels"
        / split
        / relative_label
    )

    destination_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_label.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        image_path,
        destination_image,
    )

    shutil.copy2(
        label_path,
        destination_label,
    )


# ============================================================
# CLEAN DATASET YAML
# ============================================================

def create_yaml() -> Path:

    yaml_path = (
        CLEAN_DATASET
        / "aeroinspect_crack_v1.yaml"
    )

    yaml_content = f"""# AeroInspect AI
# Experiment 001
# Single-class concrete crack instance segmentation

path: {CLEAN_DATASET.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: crack
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    return yaml_path


# ============================================================
# REPORT HELPERS
# ============================================================

def flatten_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    metrics = result["polygon_metrics"]

    areas = [
        m["area_ratio"]
        for m in metrics
    ]

    point_counts = [
        m["point_count"]
        for m in metrics
    ]

    return {
        "status": result["status"],
        "image": result["image"],
        "label": result["label"],
        "width": (
            result["image_size"]["width"]
            if result["image_size"]
            else None
        ),
        "height": (
            result["image_size"]["height"]
            if result["image_size"]
            else None
        ),
        "polygon_count": result["polygon_count"],
        "class_ids": ",".join(
            str(x)
            for x in sorted(
                set(result["class_ids"])
            )
        ),
        "empty_label": result["empty_label"],
        "known_bad_sample": result["known_bad_sample"],
        "duplicate_polygon_count": (
            result["duplicate_polygon_count"]
        ),
        "min_area_ratio": (
            min(areas)
            if areas
            else None
        ),
        "max_area_ratio": (
            max(areas)
            if areas
            else None
        ),
        "min_point_count": (
            min(point_counts)
            if point_counts
            else None
        ),
        "max_point_count": (
            max(point_counts)
            if point_counts
            else None
        ),
        "hard_errors": " | ".join(
            result["hard_errors"]
        ),
        "warnings": " | ".join(
            result["warnings"]
        ),
        "sha256": result["image_sha256"],
    }


def save_json(
    results: list[dict[str, Any]],
) -> Path:

    path = (
        REPORT_DIR
        / "dataset_quality_report.json"
    )

    serializable = []

    for result in results:

        item = dict(result)

        serializable.append(
            item
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return path


def save_csv(
    results: list[dict[str, Any]],
) -> Path:

    path = (
        REPORT_DIR
        / "dataset_quality_report.csv"
    )

    rows = [
        flatten_result(
            result
        )
        for result in results
    ]

    fieldnames = list(
        rows[0].keys()
    ) if rows else [
        "status",
        "image",
        "label",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


# ============================================================
# TEXT SUMMARY
# ============================================================

def print_final_summary(
    results: list[dict[str, Any]],
) -> None:

    status_counter = Counter(
        r["status"]
        for r in results
    )

    error_counter = Counter()

    warning_counter = Counter()

    for result in results:

        for error in result["hard_errors"]:
            error_counter[error] += 1

        for warning in result["warnings"]:
            warning_counter[warning] += 1

    total_polygons = sum(
        r["polygon_count"]
        for r in results
    )

    total_duplicates = sum(
        r["duplicate_polygon_count"]
        for r in results
    )

    print()
    print("=" * 78)
    print("FINAL DATASET QUALITY SUMMARY")
    print("=" * 78)

    print(
        f"Images scanned        : "
        f"{len(results)}"
    )

    print(
        f"Polygons scanned      : "
        f"{total_polygons}"
    )

    print()

    print("STATUS")

    for status, count in sorted(
        status_counter.items()
    ):

        print(
            f"  {status:20s}: {count}"
        )

    print()

    print(
        "HARD ERRORS"
    )

    if error_counter:

        for error, count in (
            error_counter.most_common()
        ):

            print(
                f"  {error:45s}: {count}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "WARNINGS"
    )

    if warning_counter:

        for warning, count in (
            warning_counter.most_common()
        ):

            print(
                f"  {warning:45s}: {count}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        f"Duplicate polygon annotations: "
        f"{total_duplicates}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "AEROINSPECT AI — DATASET QUALITY SCAN"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Validate raw dataset
    # --------------------------------------------------------

    if not RAW_DATASET.exists():

        raise FileNotFoundError(
            f"Raw dataset not found:\n"
            f"{RAW_DATASET}"
        )

    images_root = (
        RAW_DATASET / "images"
    )

    labels_root = (
        RAW_DATASET / "labels"
    )

    if not images_root.exists():

        raise FileNotFoundError(
            f"Images directory not found:\n"
            f"{images_root}"
        )

    if not labels_root.exists():

        raise FileNotFoundError(
            f"Labels directory not found:\n"
            f"{labels_root}"
        )

    # --------------------------------------------------------
    # GPU information
    # --------------------------------------------------------

    report_cuda_status()

    # --------------------------------------------------------
    # Clean previous GENERATED outputs
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print()
        print(
            "Removing previous quality-scan output..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Re-create clean dataset
    # --------------------------------------------------------

    if CLEAN_DATASET.exists():

        print()
        print(
            "Existing AeroInspect clean dataset found."
        )

        print(
            "It will be regenerated from the RAW dataset."
        )

        shutil.rmtree(
            CLEAN_DATASET
        )

    # --------------------------------------------------------
    # Scan all splits
    # --------------------------------------------------------

    results = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        print()
        print("-" * 78)
        print(
            f"SCANNING {split.upper()}"
        )
        print("-" * 78)

        image_paths = get_images_for_split(
            split
        )

        print(
            f"Images found: "
            f"{len(image_paths)}"
        )

        kept = 0
        excluded = 0
        warnings = 0

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            relative = image_path.relative_to(
                images_root / split
            )

            label_path = (
                labels_root
                / split
                / relative.with_suffix(".txt")
            )

            result = validate_image(
                image_path,
                label_path,
            )

            results.append(
                result
            )

            if result["status"] == "EXCLUDE":

                excluded += 1

            else:

                kept += 1

                # Valid and valid-with-warnings
                copy_valid_sample(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                )

                if result["status"] == (
                    "VALID_WITH_WARNINGS"
                ):

                    warnings += 1

            if (
                index == 1
                or index % 250 == 0
                or index == len(image_paths)
            ):

                print(
                    f"  Progress: "
                    f"{index}/{len(image_paths)}"
                )

        print()
        print(
            f"Kept     : {kept}"
        )

        print(
            f"Excluded : {excluded}"
        )

        print(
            f"Warnings : {warnings}"
        )

    # --------------------------------------------------------
    # Create YAML
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("CREATING CLEAN DATASET YAML")
    print("=" * 78)

    yaml_path = create_yaml()

    print(
        f"YAML created:\n"
        f"  {yaml_path}"
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    json_path = save_json(
        results
    )

    csv_path = save_csv(
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_final_summary(
        results
    )

    # --------------------------------------------------------
    # Exclusion report
    # --------------------------------------------------------

    excluded_results = [
        r
        for r in results
        if r["status"] == "EXCLUDE"
    ]

    excluded_path = (
        REPORT_DIR
        / "excluded_samples.json"
    )

    with excluded_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            excluded_results,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print("=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)

    print(
        f"JSON report:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"\nCSV report:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"\nExcluded samples:"
    )

    print(
        f"  {excluded_path}"
    )

    print(
        f"\nClean dataset:"
    )

    print(
        f"  {CLEAN_DATASET}"
    )

    print(
        f"\nClean YAML:"
    )

    print(
        f"  {yaml_path}"
    )

    print()
    print("=" * 78)
    print(
        "RAW DATASET SAFETY CHECK"
    )
    print("=" * 78)

    print(
        "The original crack-seg dataset was NOT modified."
    )

    print(
        f"Raw dataset remains at:\n"
        f"  {RAW_DATASET}"
    )

    print()
    print(
        "QUALITY SCAN COMPLETE."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()