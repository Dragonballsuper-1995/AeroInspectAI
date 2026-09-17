from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "crack-seg"
)

IMAGE_ROOT = DATASET_ROOT / "images"
LABEL_ROOT = DATASET_ROOT / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "audit_results"
    / "yolo_validation"
)

SAMPLES_PER_SPLIT = 10

RANDOM_SEED = 42

# Confidence is not relevant to ground-truth annotations,
# but this controls visualization thickness.
LINE_THICKNESS = 2

# Alpha used for the filled segmentation overlay.
MASK_ALPHA = 0.30


# ============================================================
# GPU INITIALIZATION
# ============================================================

def initialize_gpu() -> torch.device:
    print()
    print("=" * 78)
    print("GPU / CUDA VERIFICATION")
    print("=" * 78)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print()
        print("WARNING: CUDA is NOT available.")
        print(
            "This script can still perform annotation "
            "validation on CPU."
        )

        print()
        print("Possible causes:")
        print("  - PyTorch CPU-only build")
        print("  - CUDA/PyTorch compatibility issue")
        print("  - NVIDIA driver problem")

        return torch.device("cpu")

    device = torch.device("cuda:0")

    gpu_name = torch.cuda.get_device_name(0)

    total_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU             : {gpu_name}")
    print(f"VRAM            : {total_memory:.2f} GB")
    print(f"CUDA device     : {device}")

    # --------------------------------------------------------
    # Explicit CUDA computation
    # --------------------------------------------------------

    # This performs a real GPU operation to verify that CUDA
    # tensors can be allocated and computed.
    test_tensor = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        device=device,
        dtype=torch.float32,
    )

    test_result = torch.sum(test_tensor * test_tensor)

    torch.cuda.synchronize()

    print(
        f"CUDA test result : "
        f"{test_result.item():.2f}"
    )

    print("GPU status       : READY")

    return device


# ============================================================
# GPU COORDINATE CONVERSION
# ============================================================

def normalized_to_pixel_gpu(
    polygon: List[float],
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    """
    Convert YOLO normalized polygon coordinates:

        x1 y1 x2 y2 ...

    where values are in [0,1]

    into pixel coordinates:

        x1 y1 x2 y2 ...
    """

    if len(polygon) < 6:
        raise ValueError(
            "Polygon must contain at least 3 points."
        )

    if len(polygon) % 2 != 0:
        raise ValueError(
            "Polygon must contain an even number of coordinates."
        )

    # --------------------------------------------------------
    # GPU tensor creation
    # --------------------------------------------------------

    coords = torch.tensor(
        polygon,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 2)

    scale = torch.tensor(
        [width - 1, height - 1],
        dtype=torch.float32,
        device=device,
    )

    pixel_coords = coords * scale

    # Clamp using GPU.
    pixel_coords[:, 0] = torch.clamp(
        pixel_coords[:, 0],
        0,
        width - 1,
    )

    pixel_coords[:, 1] = torch.clamp(
        pixel_coords[:, 1],
        0,
        height - 1,
    )

    torch.cuda.synchronize() if device.type == "cuda" else None

    return (
        pixel_coords
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )


# ============================================================
# YOLO LABEL PARSER
# ============================================================

def parse_yolo_segmentation(
    label_path: Path,
) -> Tuple[List[np.ndarray], List[str]]:

    polygons: List[np.ndarray] = []
    errors: List[str] = []

    if not label_path.exists():
        return polygons, [
            "Label file does not exist."
        ]

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return polygons, [
            f"Could not read label file: {exc}"
        ]

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Empty label = potentially valid negative image.
    if not lines:
        return [], []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        parts = line.split()

        try:
            values = [
                float(value)
                for value in parts
            ]
        except ValueError:
            errors.append(
                f"Line {line_number}: "
                "contains non-numeric data."
            )
            continue

        if len(values) < 7:
            errors.append(
                f"Line {line_number}: "
                f"only {len(values)} values. "
                "Expected class + at least 3 polygon points."
            )
            continue

        if len(values) % 2 == 0:
            errors.append(
                f"Line {line_number}: "
                "invalid segmentation format."
            )
            continue

        class_id = values[0]

        if int(class_id) != 0:
            errors.append(
                f"Line {line_number}: "
                f"unexpected class ID {class_id}. "
                "Expected class 0."
            )

        polygon = values[1:]

        if not polygon:
            errors.append(
                f"Line {line_number}: empty polygon."
            )
            continue

        coordinates = np.asarray(
            polygon,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Check YOLO normalized range
        # ----------------------------------------------------

        if np.any(coordinates < 0.0) or np.any(
            coordinates > 1.0
        ):
            errors.append(
                f"Line {line_number}: "
                "coordinates outside [0,1]."
            )
            continue

        polygons.append(polygon)

    return polygons, errors


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(
    image: np.ndarray,
    polygons: List[np.ndarray],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:

    annotated = image.copy()

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    height, width = image.shape[:2]

    for polygon in polygons:

        points = normalized_to_pixel_gpu(
            polygon=polygon,
            width=width,
            height=height,
            device=device,
        )

        points_cv = points.reshape(
            (-1, 1, 2)
        )

        # ----------------------------------------------------
        # Fill mask
        # ----------------------------------------------------

        cv2.fillPoly(
            mask,
            [points_cv],
            255,
        )

        # ----------------------------------------------------
        # Draw polygon outline
        # ----------------------------------------------------

        cv2.polylines(
            annotated,
            [points_cv],
            isClosed=True,
            color=(0, 255, 0),
            thickness=LINE_THICKNESS,
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x_coordinates = points[:, 0]
        y_coordinates = points[:, 1]

        x1 = int(np.min(x_coordinates))
        y1 = int(np.min(y_coordinates))
        x2 = int(np.max(x_coordinates))
        y2 = int(np.max(y_coordinates))

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            LINE_THICKNESS,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            "CRACK",
            (x1, max(y1 - 7, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, mask


# ============================================================
# CREATE THREE-PANEL VISUALIZATION
# ============================================================

def create_visualization(
    original: np.ndarray,
    annotated: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:

    # --------------------------------------------------------
    # Mask visualization
    # --------------------------------------------------------

    mask_bgr = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = original.copy()

    if np.any(mask > 0):

        colored_mask = np.zeros_like(
            original
        )

        colored_mask[:, :, 1] = mask

        overlay = cv2.addWeighted(
            original,
            1.0 - MASK_ALPHA,
            colored_mask,
            MASK_ALPHA,
            0,
        )

    # --------------------------------------------------------
    # Resize all panels to common dimensions
    # --------------------------------------------------------

    target_width = 512
    target_height = 512

    panels = []

    for panel in [
        original,
        annotated,
        overlay,
    ]:

        panels.append(
            cv2.resize(
                panel,
                (
                    target_width,
                    target_height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        )

    # --------------------------------------------------------
    # Add panel titles
    # --------------------------------------------------------

    titles = [
        "ORIGINAL",
        "ANNOTATIONS",
        "MASK OVERLAY",
    ]

    for panel, title in zip(
        panels,
        titles,
    ):

        cv2.rectangle(
            panel,
            (0, 0),
            (target_width, 35),
            (30, 30, 30),
            -1,
        )

        cv2.putText(
            panel,
            title,
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Horizontal contact sheet
    # --------------------------------------------------------

    return np.hstack(panels)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return {
            "status": "ERROR",
            "reason": "Could not read image.",
        }

    polygons, errors = parse_yolo_segmentation(
        label_path
    )

    annotated, mask = draw_annotations(
        image=image,
        polygons=polygons,
        device=device,
    )

    visualization = create_visualization(
        original=image,
        annotated=annotated,
        mask=mask,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        visualization,
    )

    if not success:
        return {
            "status": "ERROR",
            "reason": "Could not write visualization.",
        }

    return {
        "status": (
            "EMPTY"
            if len(polygons) == 0
            else "OK"
        ),
        "polygons": len(polygons),
        "errors": errors,
    }


# ============================================================
# FIND SPLIT IMAGES
# ============================================================

def get_split_images(
    split: str,
) -> list[Path]:

    split_dir = (
        IMAGE_ROOT / split
    )

    if not split_dir.exists():
        return []

    return sorted(
        p
        for p in split_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    )


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(
    split: str,
    device: torch.device,
) -> dict:

    print()
    print("-" * 78)
    print(f"VALIDATING SPLIT: {split.upper()}")
    print("-" * 78)

    image_paths = get_split_images(
        split
    )

    if not image_paths:
        print("No images found.")
        return {
            "split": split,
            "images": 0,
            "missing_labels": 0,
            "empty_labels": 0,
            "errors": 0,
        }

    label_split_dir = (
        LABEL_ROOT / split
    )

    rng = random.Random(
        RANDOM_SEED
        + hash(split) % 1000
    )

    sample_count = min(
        SAMPLES_PER_SPLIT,
        len(image_paths),
    )

    selected_images = rng.sample(
        image_paths,
        sample_count,
    )

    print(
        f"Images in split: {len(image_paths)}"
    )

    print(
        f"Visual samples : {sample_count}"
    )

    missing_labels = 0
    empty_labels = 0
    annotation_errors = 0
    total_polygons = 0

    split_output = (
        OUTPUT_ROOT / split
    )

    split_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # First: audit ALL labels
    # --------------------------------------------------------

    for image_path in image_paths:

        relative = image_path.relative_to(
            IMAGE_ROOT / split
        )

        label_path = (
            label_split_dir
            / relative.with_suffix(".txt")
        )

        if not label_path.exists():
            missing_labels += 1
            continue

        polygons, errors = parse_yolo_segmentation(
            label_path
        )

        if not polygons:
            empty_labels += 1

        annotation_errors += len(errors)
        total_polygons += len(polygons)

    # --------------------------------------------------------
    # Then: create visual samples
    # --------------------------------------------------------

    print("\nCreating visual samples...")

    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):

        relative = image_path.relative_to(
            IMAGE_ROOT / split
        )

        label_path = (
            label_split_dir
            / relative.with_suffix(".txt")
        )

        filename = (
            f"{index:02d}_"
            f"{image_path.stem}"
            ".jpg"
        )

        output_path = (
            split_output / filename
        )

        result = process_image(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            device=device,
        )

        print(
            f"  [{index:02d}/{sample_count}] "
            f"{relative}"
        )

        print(
            f"       result={result['status']} "
            f"polygons={result.get('polygons', 0)}"
        )

        if result.get("errors"):
            for error in result["errors"]:
                print(
                    f"       WARNING: {error}"
                )

    print()
    print("SPLIT SUMMARY")
    print(
        f"  Images             : {len(image_paths)}"
    )
    print(
        f"  Missing labels     : {missing_labels}"
    )
    print(
        f"  Empty labels       : {empty_labels}"
    )
    print(
        f"  Polygon count      : {total_polygons}"
    )
    print(
        f"  Annotation errors  : {annotation_errors}"
    )

    return {
        "split": split,
        "images": len(image_paths),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "total_polygons": total_polygons,
        "annotation_errors": annotation_errors,
    }


# ============================================================
# EMPTY-LABEL INSPECTION
# ============================================================

def find_empty_labels() -> list[Tuple[Path, Path]]:

    results = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        label_dir = (
            LABEL_ROOT / split
        )

        if not label_dir.exists():
            continue

        for label_path in label_dir.rglob(
            "*.txt"
        ):

            try:
                text = label_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip()
            except Exception:
                continue

            if not text:

                relative = (
                    label_path.relative_to(
                        label_dir
                    )
                )

                image_path = (
                    IMAGE_ROOT
                    / split
                    / relative.with_suffix(".jpg")
                )

                if image_path.exists():
                    results.append(
                        (
                            image_path,
                            label_path,
                        )
                    )

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "AEROINSPECT AI — YOLO ANNOTATION VALIDATOR"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        print()
        print(
            f"ERROR: Dataset not found:\n"
            f"{DATASET_ROOT}"
        )
        sys.exit(1)

    if not IMAGE_ROOT.exists():
        print(
            f"ERROR: Images directory not found:\n"
            f"{IMAGE_ROOT}"
        )
        sys.exit(1)

    if not LABEL_ROOT.exists():
        print(
            f"ERROR: Labels directory not found:\n"
            f"{LABEL_ROOT}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Initialize GPU
    # --------------------------------------------------------

    device = initialize_gpu()

    print()
    print("=" * 78)
    print("DATASET")
    print("=" * 78)

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Image root  : {IMAGE_ROOT}")
    print(f"Label root  : {LABEL_ROOT}")

    # --------------------------------------------------------
    # Create fresh output directory
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print()
        print(
            "Previous YOLO validation directory exists."
        )

        print(
            "It will be replaced because this directory "
            "contains only generated audit output."
        )

        import shutil

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate train / val / test
    # --------------------------------------------------------

    results = []

    for split in [
        "train",
        "val",
        "test",
    ]:

        results.append(
            validate_split(
                split=split,
                device=device,
            )
        )

    # --------------------------------------------------------
    # Explicit empty-label report
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("EMPTY LABEL INSPECTION")
    print("=" * 78)

    empty_labels = find_empty_labels()

    if not empty_labels:

        print("No empty label files found.")

    else:

        print(
            f"Empty labels found: "
            f"{len(empty_labels)}"
        )

        for image_path, label_path in empty_labels:

            print()
            print(
                f"Image: {image_path}"
            )

            print(
                f"Label: {label_path}"
            )

    # --------------------------------------------------------
    # Final GPU memory report
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.synchronize()

        allocated = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 2)
        )

        print()
        print("=" * 78)
        print("GPU MEMORY")
        print("=" * 78)

        print(
            f"Allocated : {allocated:.2f} MB"
        )

        print(
            f"Reserved  : {reserved:.2f} MB"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    for result in results:

        print(
            f"\n{result['split'].upper()}:"
        )

        print(
            f"  Images            : "
            f"{result['images']}"
        )

        print(
            f"  Missing labels    : "
            f"{result['missing_labels']}"
        )

        print(
            f"  Empty labels     : "
            f"{result['empty_labels']}"
        )

        print(
            f"  Polygons          : "
            f"{result['total_polygons']}"
        )

        print(
            f"  Annotation errors : "
            f"{result['annotation_errors']}"
        )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"Visualizations saved to:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Original dataset files were NOT modified."
    )

    print(
        "  YOLO coordinate conversion used CUDA "
        "when available."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "  Inspect the generated train/val/test "
        "visualizations before training."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()