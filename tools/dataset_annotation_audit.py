from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")
DATASET_ROOT = PROJECT_ROOT / "datasets"

OUTPUT_ROOT = PROJECT_ROOT / "audit_results"
REPORT_DIR = OUTPUT_ROOT / "reports"
SAMPLE_DIR = OUTPUT_ROOT / "samples"

RANDOM_SEED = 42
SAMPLES_PER_DATASET = 12

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

MASK_EXTENSIONS = IMAGE_EXTENSIONS


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TARGET_DATASETS = [
    "crack-seg",
    "crack_segmentation_dataset",
    "concreteCrackSegmentationDataset",
    "Crackseg9k",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def all_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return [
        p for p in root.rglob("*")
        if p.is_file()
    ]


def safe_open_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def image_info(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
            }
    except Exception as exc:
        return {
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "error": str(exc),
        }


def normalize_stem(stem: str) -> str:
    """
    Normalizes common suffixes used by segmentation datasets.

    Examples:
        image_001 -> image_001
        image_001_mask -> image_001
        image_001_gt -> image_001
        image_001_label -> image_001
    """

    suffixes = [
        "_mask",
        "_masks",
        "_gt",
        "_groundtruth",
        "_ground_truth",
        "_label",
        "_labels",
        "_seg",
        "_segmentation",
        "_bw",
    ]

    result = stem.lower()

    changed = True
    while changed:
        changed = False

        for suffix in suffixes:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
                break

    return result


# ============================================================
# MASK ANALYSIS
# ============================================================

def analyze_mask(mask_path: Path) -> dict[str, Any]:
    result = {
        "path": str(mask_path),
        "exists": mask_path.exists(),
        "width": None,
        "height": None,
        "mode": None,
        "unique_values": [],
        "nonzero_pixels": 0,
        "total_pixels": 0,
        "foreground_ratio": 0.0,
        "is_empty": False,
        "is_binary": False,
        "error": None,
    }

    if not mask_path.exists():
        return result

    try:
        with Image.open(mask_path) as img:
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode

            gray = img.convert("L")

            # Downsampling isn't used here because we want exact
            # foreground statistics.
            values = list(gray.getdata())

            unique = sorted(set(values))

            result["unique_values"] = unique[:100]
            result["total_pixels"] = len(values)

            nonzero = sum(v > 0 for v in values)

            result["nonzero_pixels"] = nonzero
            result["foreground_ratio"] = (
                nonzero / len(values)
                if values
                else 0.0
            )

            result["is_empty"] = nonzero == 0

            # A mask is considered binary if it only contains
            # 0 and one foreground value.
            result["is_binary"] = (
                len(unique) <= 2
                and 0 in unique
                and len(unique) >= 1
            )

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ============================================================
# YOLO LABEL ANALYSIS
# ============================================================

def parse_yolo_label(
    label_path: Path,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, Any]:

    result = {
        "path": str(label_path),
        "exists": label_path.exists(),
        "line_count": 0,
        "empty": False,
        "malformed_lines": 0,
        "valid_lines": 0,
        "class_ids": [],
        "values_per_line": [],
        "possible_format": "unknown",
        "errors": [],
    }

    if not label_path.exists():
        return result

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    result["line_count"] = len(lines)
    result["empty"] = len(lines) == 0

    if result["empty"]:
        return result

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        try:
            numbers = [float(x) for x in parts]
        except ValueError:
            result["malformed_lines"] += 1
            result["errors"].append(
                f"Line {line_number}: non-numeric content"
            )
            continue

        result["values_per_line"].append(len(numbers))

        # Standard YOLO bounding box:
        # class x_center y_center width height
        if len(numbers) == 5:

            class_id = int(numbers[0])

            result["class_ids"].append(class_id)
            result["valid_lines"] += 1

            # Check coordinate range.
            coords = numbers[1:]

            if not all(0.0 <= x <= 1.0 for x in coords):
                result["errors"].append(
                    f"Line {line_number}: YOLO box coordinates "
                    f"outside [0,1]"
                )

            if result["possible_format"] == "unknown":
                result["possible_format"] = "YOLO detection"

        # YOLO segmentation:
        # class x1 y1 x2 y2 x3 y3 ...
        elif len(numbers) >= 7 and len(numbers) % 2 == 1:

            class_id = int(numbers[0])

            result["class_ids"].append(class_id)
            result["valid_lines"] += 1

            coords = numbers[1:]

            if not all(0.0 <= x <= 1.0 for x in coords):
                result["errors"].append(
                    f"Line {line_number}: YOLO polygon coordinates "
                    f"outside [0,1]"
                )

            if result["possible_format"] == "unknown":
                result["possible_format"] = "YOLO segmentation"

        else:
            result["malformed_lines"] += 1
            result["errors"].append(
                f"Line {line_number}: {len(numbers)} numeric values; "
                f"does not match common YOLO formats"
            )

    return result


# ============================================================
# JSON ANALYSIS
# ============================================================

def inspect_json_file(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "valid_json": False,
        "root_type": None,
        "keys": [],
        "error": None,
    }

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        result["valid_json"] = True
        result["root_type"] = type(data).__name__

        if isinstance(data, dict):
            result["keys"] = sorted(
                [str(k) for k in data.keys()]
            )[:100]

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ============================================================
# DATASET STRUCTURE DETECTION
# ============================================================

def detect_structure(root: Path) -> dict[str, Any]:

    structure = {
        "has_images_directory": False,
        "has_labels_directory": False,
        "has_train_directory": False,
        "has_val_directory": False,
        "has_test_directory": False,
        "has_masks_directory": False,
        "json_count": 0,
        "txt_count": 0,
        "yaml_count": 0,
        "jpg_count": 0,
        "png_count": 0,
    }

    if not root.exists():
        return structure

    directory_names = {
        p.name.lower()
        for p in root.rglob("*")
        if p.is_dir()
    }

    structure["has_images_directory"] = "images" in directory_names
    structure["has_labels_directory"] = "labels" in directory_names
    structure["has_masks_directory"] = (
        "masks" in directory_names
        or "mask" in directory_names
        or "bw" in directory_names
    )

    structure["has_train_directory"] = "train" in directory_names
    structure["has_val_directory"] = (
        "val" in directory_names
        or "valid" in directory_names
        or "validation" in directory_names
    )
    structure["has_test_directory"] = "test" in directory_names

    for p in root.rglob("*"):
        if p.is_file():
            suffix = p.suffix.lower()

            if suffix == ".json":
                structure["json_count"] += 1
            elif suffix == ".txt":
                structure["txt_count"] += 1
            elif suffix in {".yaml", ".yml"}:
                structure["yaml_count"] += 1
            elif suffix == ".jpg":
                structure["jpg_count"] += 1
            elif suffix == ".png":
                structure["png_count"] += 1

    return structure


# ============================================================
# DATASET-SPECIFIC INSPECTION
# ============================================================

def inspect_crack_seg(root: Path) -> dict[str, Any]:

    result = {
        "dataset_type": "potential_yolo_dataset",
        "image_label_pairs": [],
        "missing_labels": [],
        "orphan_labels": [],
        "label_stats": {
            "empty_labels": 0,
            "malformed_files": 0,
            "valid_files": 0,
            "class_ids": Counter(),
            "formats": Counter(),
        },
    }

    images_root = root / "images"
    labels_root = root / "labels"

    images = image_files(images_root)

    image_stems = {
        p.relative_to(images_root).with_suffix("")
        for p in images
    }

    label_files = list(
        labels_root.rglob("*.txt")
    ) if labels_root.exists() else []

    label_stems = {
        p.relative_to(labels_root).with_suffix("")
        for p in label_files
    }

    for image_path in images:

        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")

        info = image_info(image_path)

        parsed = parse_yolo_label(
            label_path,
            info.get("width"),
            info.get("height"),
        )

        if not label_path.exists():
            result["missing_labels"].append(
                str(relative)
            )
        else:
            result["label_stats"]["valid_files"] += (
                parsed["valid_lines"] > 0
            )

            if parsed["empty"]:
                result["label_stats"]["empty_labels"] += 1

            if parsed["malformed_lines"] > 0:
                result["label_stats"]["malformed_files"] += 1

            result["label_stats"]["formats"][
                parsed["possible_format"]
            ] += 1

            for cid in parsed["class_ids"]:
                result["label_stats"]["class_ids"][cid] += 1

        result["image_label_pairs"].append({
            "image": str(relative),
            "label": (
                str(label_path.relative_to(labels_root))
                if label_path.exists()
                else None
            ),
            "image_size": info,
            "label_analysis": parsed,
        })

    orphan_stems = label_stems - image_stems

    result["orphan_labels"] = [
        str(x)
        for x in sorted(orphan_stems)
    ]

    return result


def find_candidate_masks(root: Path) -> list[Path]:

    candidates = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in MASK_EXTENSIONS:
            continue

        parent_name = p.parent.name.lower()

        stem = p.stem.lower()

        if (
            parent_name in {
                "bw",
                "mask",
                "masks",
                "gt",
                "groundtruth",
                "ground_truth",
                "labels",
                "label",
            }
            or any(
                token in stem
                for token in [
                    "mask",
                    "groundtruth",
                    "ground_truth",
                    "_gt",
                    "_seg",
                ]
            )
        ):
            candidates.append(p)

    return sorted(set(candidates))


def inspect_mask_dataset(root: Path) -> dict[str, Any]:

    result = {
        "dataset_type": "potential_mask_dataset",
        "images": [],
        "candidate_masks": [],
        "pairs": [],
        "missing_masks": [],
        "orphan_masks": [],
        "mask_stats": {
            "empty_masks": 0,
            "binary_masks": 0,
            "non_binary_masks": 0,
            "errors": 0,
            "foreground_ratios": [],
        },
    }

    candidates = find_candidate_masks(root)

    result["candidate_masks"] = [
        str(p.relative_to(root))
        for p in candidates
    ]

    # Remove obvious masks from the main image pool.
    all_imgs = image_files(root)

    mask_set = set(candidates)

    actual_images = [
        p for p in all_imgs
        if p not in mask_set
    ]

    result["images"] = [
        str(p.relative_to(root))
        for p in actual_images
    ]

    # Index candidate masks by normalized stem.
    masks_by_stem: dict[str, list[Path]] = defaultdict(list)

    for mask in candidates:
        masks_by_stem[
            normalize_stem(mask.stem)
        ].append(mask)

    used_masks: set[Path] = set()

    for image_path in actual_images:

        key = normalize_stem(image_path.stem)

        candidates_for_image = masks_by_stem.get(key, [])

        mask_path = None

        if candidates_for_image:
            # Prefer a mask in the same parent family where possible.
            mask_path = candidates_for_image[0]

        if mask_path:
            used_masks.add(mask_path)

            mask_info = analyze_mask(mask_path)

            if mask_info["is_empty"]:
                result["mask_stats"]["empty_masks"] += 1

            if mask_info["is_binary"]:
                result["mask_stats"]["binary_masks"] += 1
            else:
                result["mask_stats"]["non_binary_masks"] += 1

            if mask_info["error"]:
                result["mask_stats"]["errors"] += 1

            result["mask_stats"]["foreground_ratios"].append(
                mask_info["foreground_ratio"]
            )

            result["pairs"].append({
                "image": str(image_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
                "image_size": image_info(image_path),
                "mask_analysis": mask_info,
            })

        else:
            result["missing_masks"].append(
                str(image_path.relative_to(root))
            )

    result["orphan_masks"] = [
        str(p.relative_to(root))
        for p in candidates
        if p not in used_masks
    ]

    return result


# ============================================================
# CONTACT SHEET GENERATION
# ============================================================

def make_contact_sheet(
    items: list[tuple[Path, Path | None, str]],
    output_path: Path,
    columns: int = 3,
) -> None:

    if not items:
        return

    thumb_w = 500
    thumb_h = 430

    rows = (len(items) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (columns * thumb_w, rows * thumb_h),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, (image_path, annotation_path, title) in enumerate(items):

        row = index // columns
        col = index % columns

        x = col * thumb_w
        y = row * thumb_h

        image = safe_open_image(image_path)

        if image is None:
            continue

        image.thumbnail(
            (thumb_w - 20, 260)
        )

        paste_x = x + (thumb_w - image.width) // 2
        paste_y = y + 20

        sheet.paste(
            image,
            (paste_x, paste_y),
        )

        draw.text(
            (x + 10, y + 295),
            title[:70],
            fill="black",
        )

        if annotation_path and annotation_path.exists():

            # Display segmentation masks as a thumbnail
            try:
                with Image.open(annotation_path) as ann:
                    ann = ann.convert("L")
                    ann.thumbnail((thumb_w - 20, 100))

                    # Convert to RGB for visualization.
                    ann_rgb = Image.merge(
                        "RGB",
                        (ann, ann, ann),
                    )

                    sheet.paste(
                        ann_rgb,
                        (
                            x + 10,
                            y + 320,
                        ),
                    )

            except Exception:
                pass

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(output_path)


# ============================================================
# RANDOM SAMPLE SELECTION
# ============================================================

def choose_random_images(
    images: list[Path],
    count: int,
) -> list[Path]:

    if not images:
        return []

    rng = random.Random(RANDOM_SEED)

    if len(images) <= count:
        return images

    return rng.sample(
        images,
        count,
    )


# ============================================================
# CRACK-SEG CONTACT SHEET
# ============================================================

def create_crack_seg_samples(
    root: Path,
    output_dir: Path,
) -> None:

    images_root = root / "images"
    labels_root = root / "labels"

    images = image_files(images_root)

    chosen = choose_random_images(
        images,
        SAMPLES_PER_DATASET,
    )

    sample_items = []

    for image_path in chosen:

        relative = image_path.relative_to(images_root)

        label_path = labels_root / relative.with_suffix(".txt")

        title = (
            f"{relative} | "
            f"label={'YES' if label_path.exists() else 'NO'}"
        )

        # For YOLO labels, the raw .txt cannot be directly
        # rendered as an image. The title still records its presence.
        sample_items.append(
            (
                image_path,
                None,
                title,
            )
        )

    make_contact_sheet(
        sample_items,
        output_dir / "crack-seg_images.png",
    )


# ============================================================
# MASK DATASET CONTACT SHEET
# ============================================================

def create_mask_dataset_samples(
    root: Path,
    output_dir: Path,
) -> None:

    candidates = find_candidate_masks(root)

    mask_by_stem = defaultdict(list)

    for mask in candidates:
        mask_by_stem[
            normalize_stem(mask.stem)
        ].append(mask)

    all_imgs = image_files(root)

    actual_images = [
        p for p in all_imgs
        if p not in set(candidates)
    ]

    chosen = choose_random_images(
        actual_images,
        SAMPLES_PER_DATASET,
    )

    sample_items = []

    for image_path in chosen:

        key = normalize_stem(image_path.stem)

        matches = mask_by_stem.get(key, [])

        mask_path = matches[0] if matches else None

        title = (
            f"{image_path.name} | "
            f"mask={'YES' if mask_path else 'NO'}"
        )

        sample_items.append(
            (
                image_path,
                mask_path,
                title,
            )
        )

    make_contact_sheet(
        sample_items,
        output_dir / "mask_dataset_samples.png",
    )


# ============================================================
# CRACKSEG9K JSON/TXT DISCOVERY
# ============================================================

def inspect_generic_annotations(
    root: Path,
) -> dict[str, Any]:

    result = {
        "json_files": [],
        "txt_files": [],
        "yaml_files": [],
        "json_summary": Counter(),
        "txt_summary": Counter(),
    }

    json_files = sorted(root.rglob("*.json"))
    txt_files = sorted(root.rglob("*.txt"))
    yaml_files = sorted(
        list(root.rglob("*.yaml"))
        + list(root.rglob("*.yml"))
    )

    for path in json_files:

        info = inspect_json_file(path)

        result["json_files"].append(info)

        if info["valid_json"]:
            result["json_summary"]["valid"] += 1

            if info["root_type"]:
                result["json_summary"][
                    f"root={info['root_type']}"
                ] += 1

        else:
            result["json_summary"]["invalid"] += 1

    result["txt_files"] = [
        str(p.relative_to(root))
        for p in txt_files
    ]

    result["yaml_files"] = [
        str(p.relative_to(root))
        for p in yaml_files
    ]

    return result


# ============================================================
# MAIN DATASET AUDIT
# ============================================================

def audit_dataset(dataset_name: str) -> dict[str, Any]:

    root = DATASET_ROOT / dataset_name

    result = {
        "dataset": dataset_name,
        "path": str(root),
        "exists": root.exists(),
        "structure": {},
        "annotation_analysis": {},
    }

    if not root.exists():
        return result

    result["structure"] = detect_structure(root)

    if dataset_name.lower() == "crack-seg":
        result["annotation_analysis"] = inspect_crack_seg(root)

    elif dataset_name.lower() in {
        "concretecracksegmentationdataset",
        "crack_segmentation_dataset",
    }:
        result["annotation_analysis"] = inspect_mask_dataset(root)

    else:
        result["annotation_analysis"] = inspect_generic_annotations(
            root
        )

    return result


# ============================================================
# REPORT PRINTING
# ============================================================

def print_summary(result: dict[str, Any]) -> None:

    print()
    print("=" * 78)
    print(f"DATASET: {result['dataset']}")
    print("=" * 78)

    if not result["exists"]:
        print("Dataset does not exist.")
        return

    print(f"Path: {result['path']}")

    structure = result["structure"]

    print("\nSTRUCTURE")
    print(f"  images/ directory : {structure['has_images_directory']}")
    print(f"  labels/ directory : {structure['has_labels_directory']}")
    print(f"  masks/ directory  : {structure['has_masks_directory']}")
    print(f"  train/ directory  : {structure['has_train_directory']}")
    print(f"  val/ directory    : {structure['has_val_directory']}")
    print(f"  test/ directory   : {structure['has_test_directory']}")

    print("\nFILE COUNTS")
    print(f"  JPG  : {structure['jpg_count']}")
    print(f"  PNG  : {structure['png_count']}")
    print(f"  TXT  : {structure['txt_count']}")
    print(f"  JSON : {structure['json_count']}")
    print(f"  YAML : {structure['yaml_count']}")

    analysis = result["annotation_analysis"]

    if "label_stats" in analysis:

        stats = analysis["label_stats"]

        print("\nYOLO LABEL ANALYSIS")
        print(
            f"  Images examined   : "
            f"{len(analysis['image_label_pairs'])}"
        )
        print(
            f"  Missing labels    : "
            f"{len(analysis['missing_labels'])}"
        )
        print(
            f"  Orphan labels     : "
            f"{len(analysis['orphan_labels'])}"
        )
        print(
            f"  Empty labels      : "
            f"{stats['empty_labels']}"
        )
        print(
            f"  Malformed files   : "
            f"{stats['malformed_files']}"
        )

        print("\n  Possible formats:")
        for key, value in stats["formats"].items():
            print(f"    {key}: {value}")

        print("\n  Class IDs:")
        if stats["class_ids"]:
            for key, value in sorted(
                stats["class_ids"].items()
            ):
                print(
                    f"    class {key}: "
                    f"{value} annotation lines"
                )
        else:
            print("    none found")

    if "mask_stats" in analysis:

        stats = analysis["mask_stats"]

        print("\nMASK ANALYSIS")
        print(
            f"  Images examined   : "
            f"{len(analysis['images'])}"
        )
        print(
            f"  Candidate masks   : "
            f"{len(analysis['candidate_masks'])}"
        )
        print(
            f"  Pairs found       : "
            f"{len(analysis['pairs'])}"
        )
        print(
            f"  Missing masks     : "
            f"{len(analysis['missing_masks'])}"
        )
        print(
            f"  Orphan masks      : "
            f"{len(analysis['orphan_masks'])}"
        )
        print(
            f"  Empty masks       : "
            f"{stats['empty_masks']}"
        )
        print(
            f"  Binary masks      : "
            f"{stats['binary_masks']}"
        )
        print(
            f"  Non-binary masks  : "
            f"{stats['non_binary_masks']}"
        )

    if "json_summary" in analysis:

        print("\nGENERIC ANNOTATION ANALYSIS")

        print("  JSON:")
        for key, value in analysis["json_summary"].items():
            print(f"    {key}: {value}")

        print(
            f"  TXT files: "
            f"{len(analysis['txt_files'])}"
        )

        print(
            f"  YAML files: "
            f"{len(analysis['yaml_files'])}"
        )


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_json_report(
    results: list[dict[str, Any]],
) -> Path:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR /
        "dataset_annotation_audit.json"
    )

    # Convert Counters into normal dictionaries.
    def make_serializable(obj: Any) -> Any:

        if isinstance(obj, Counter):
            return dict(obj)

        if isinstance(obj, defaultdict):
            return dict(obj)

        if isinstance(obj, dict):
            return {
                key: make_serializable(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                make_serializable(value)
                for value in obj
            ]

        return obj

    serializable = make_serializable(
        {
            "project_root": str(PROJECT_ROOT),
            "dataset_root": str(DATASET_ROOT),
            "datasets": results,
        }
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            serializable,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print("AEROINSPECT AI — DATASET ANNOTATION AUDIT")
    print("=" * 78)
    print(f"Dataset root: {DATASET_ROOT}")

    # Create clean audit output directory.
    if OUTPUT_ROOT.exists():
        print("\nCleaning previous audit results...")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for dataset_name in TARGET_DATASETS:

        result = audit_dataset(dataset_name)

        results.append(result)

        print_summary(result)

    # --------------------------------------------------------
    # Generate sample visualizations
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("GENERATING SAMPLE VISUALIZATIONS")
    print("=" * 78)

    for dataset_name in TARGET_DATASETS:

        root = DATASET_ROOT / dataset_name

        if not root.exists():
            continue

        dataset_output = (
            SAMPLE_DIR / dataset_name
        )

        dataset_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            if dataset_name.lower() == "crack-seg":

                create_crack_seg_samples(
                    root,
                    dataset_output,
                )

                print(
                    f"  Created samples for {dataset_name}"
                )

            elif dataset_name.lower() in {
                "concretecracksegmentationdataset",
                "crack_segmentation_dataset",
            }:

                create_mask_dataset_samples(
                    root,
                    dataset_output,
                )

                print(
                    f"  Created samples for {dataset_name}"
                )

        except Exception as exc:

            print(
                f"  ERROR creating samples for "
                f"{dataset_name}: {exc}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = save_json_report(results)

    print("\n" + "=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    print(f"\nJSON report:")
    print(f"  {report_path}")

    print("\nSample visualizations:")
    print(f"  {SAMPLE_DIR}")

    print("\nIMPORTANT:")
    print(
        "No files inside the original datasets were modified."
    )


if __name__ == "__main__":
    main()