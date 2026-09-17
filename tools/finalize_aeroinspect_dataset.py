from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")

RAW_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "crack-seg"
)

FINAL_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "aeroinspect_crack_v1"
)

QUALITY_REPORT = (
    PROJECT_ROOT
    / "audit_results"
    / "quality_scan"
    / "reports"
    / "dataset_quality_report.json"
)

FINAL_REPORT_DIR = (
    PROJECT_ROOT
    / "audit_results"
    / "final_dataset"
    / "reports"
)

# Confirmed by manual inspection.
CONFIRMED_BAD_RELATIVE_PATH = (
    "images/val/"
    "3513.rf.782300f38d0a008b3340e54b643e713e.jpg"
)


SPLITS = [
    "train",
    "val",
    "test",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HASH
# ============================================================

def sha256_file(path: Path) -> str:

    hasher = hashlib.sha256()

    with path.open("rb") as f:

        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# NORMALIZED PATH
# ============================================================

def normalize_path(path: str) -> str:

    return (
        path
        .replace("\\", "/")
        .lower()
    )


# ============================================================
# LOAD QUALITY REPORT
# ============================================================

def load_quality_report() -> list[dict]:

    if not QUALITY_REPORT.exists():

        raise FileNotFoundError(
            f"Quality report not found:\n"
            f"{QUALITY_REPORT}"
        )

    with QUALITY_REPORT.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# FIND IMAGE / LABEL
# ============================================================

def get_image_path_from_report(
    result: dict,
) -> Path:

    return Path(
        result["image"]
    )


def get_label_path_from_report(
    result: dict,
) -> Path:

    return Path(
        result["label"]
    )


# ============================================================
# VERIFY RAW DATASET
# ============================================================

def verify_raw_dataset() -> None:

    if not RAW_DATASET.exists():

        raise FileNotFoundError(
            f"Raw dataset does not exist:\n"
            f"{RAW_DATASET}"
        )

    images_root = RAW_DATASET / "images"
    labels_root = RAW_DATASET / "labels"

    if not images_root.exists():

        raise FileNotFoundError(
            f"Missing images directory:\n"
            f"{images_root}"
        )

    if not labels_root.exists():

        raise FileNotFoundError(
            f"Missing labels directory:\n"
            f"{labels_root}"
        )


# ============================================================
# CLEAN FINAL DATASET
# ============================================================

def prepare_output_directory() -> None:

    if FINAL_DATASET.exists():

        print(
            "\nRemoving previous generated final dataset..."
        )

        shutil.rmtree(
            FINAL_DATASET
        )

    FINAL_DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# COPY DATASET
# ============================================================

def copy_sample(
    image_path: Path,
    label_path: Path,
    split: str,
) -> dict:

    raw_images_split = (
        RAW_DATASET
        / "images"
        / split
    )

    try:

        relative_image = image_path.relative_to(
            raw_images_split
        )

    except ValueError as exc:

        raise ValueError(
            f"Image does not belong to expected split:\n"
            f"{image_path}\n"
            f"Expected root:\n"
            f"{raw_images_split}"
        ) from exc

    destination_image = (
        FINAL_DATASET
        / "images"
        / split
        / relative_image
    )

    destination_label = (
        FINAL_DATASET
        / "labels"
        / split
        / relative_image.with_suffix(".txt")
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

    return {
        "split": split,
        "raw_image": str(image_path),
        "raw_label": str(label_path),
        "final_image": str(destination_image),
        "final_label": str(destination_label),
        "sha256": sha256_file(image_path),
    }


# ============================================================
# WRITE YAML
# ============================================================

def write_yaml() -> Path:

    yaml_path = (
        FINAL_DATASET
        / "aeroinspect_crack_v1.yaml"
    )

    content = f"""# ============================================================
# AeroInspect AI
# Experiment 001
# Single-class concrete crack instance segmentation
# ============================================================

path: {FINAL_DATASET.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: crack
"""

    yaml_path.write_text(
        content,
        encoding="utf-8",
    )

    return yaml_path


# ============================================================
# CROSS-SPLIT DUPLICATE CHECK
# ============================================================

def check_cross_split_duplicates(
    manifest: list[dict],
) -> dict:

    hashes = defaultdict(list)

    for item in manifest:

        hashes[
            item["sha256"]
        ].append(
            {
                "split": item["split"],
                "image": item["final_image"],
                "raw_image": item["raw_image"],
            }
        )

    duplicate_groups = {}

    for digest, entries in hashes.items():

        splits = {
            entry["split"]
            for entry in entries
        }

        if len(entries) > 1 and len(splits) > 1:

            duplicate_groups[digest] = entries

    return duplicate_groups


# ============================================================
# SAVE REPORT
# ============================================================

def save_json(
    filename: str,
    data,
) -> Path:

    FINAL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        FINAL_REPORT_DIR
        / filename
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "AEROINSPECT AI — FINAL DATASET FINALIZATION"
    )
    print("=" * 78)

    verify_raw_dataset()

    quality_results = load_quality_report()

    print(
        f"\nQuality-report records: "
        f"{len(quality_results)}"
    )

    prepare_output_directory()

    manifest = []

    excluded = []

    warning_samples = []

    # --------------------------------------------------------
    # Process every image from quality report
    # --------------------------------------------------------

    for result in quality_results:

        image_path = get_image_path_from_report(
            result
        )

        label_path = get_label_path_from_report(
            result
        )

        # Determine split from path.
        normalized = normalize_path(
            str(image_path)
        )

        split = None

        for candidate in SPLITS:

            token = (
                f"/images/{candidate}/"
            )

            if token in normalized:

                split = candidate
                break

        if split is None:

            print(
                "\nWARNING: Could not determine split:"
            )

            print(
                image_path
            )

            excluded.append(
                {
                    "image": str(image_path),
                    "reason": "unknown_split",
                }
            )

            continue

        relative_from_raw = normalize_path(
            image_path.relative_to(
                RAW_DATASET
            ).as_posix()
        )

        # ----------------------------------------------------
        # Confirmed bad sample
        # ----------------------------------------------------

        if (
            relative_from_raw
            == CONFIRMED_BAD_RELATIVE_PATH
        ):

            print(
                "\nEXCLUDING CONFIRMED BAD SAMPLE:"
            )

            print(
                f"  {image_path}"
            )

            excluded.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "split": split,
                    "reason": (
                        "confirmed_visible_crack_with_empty_label"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Hard errors
        # ----------------------------------------------------

        if result.get("hard_errors"):

            excluded.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "split": split,
                    "reason": result["hard_errors"],
                }
            )

            continue

        # ----------------------------------------------------
        # File existence
        # ----------------------------------------------------

        if not image_path.exists():

            excluded.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "split": split,
                    "reason": "missing_image",
                }
            )

            continue

        if not label_path.exists():

            excluded.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "split": split,
                    "reason": "missing_label",
                }
            )

            continue

        # ----------------------------------------------------
        # Keep valid sample
        # ----------------------------------------------------

        record = copy_sample(
            image_path=image_path,
            label_path=label_path,
            split=split,
        )

        manifest.append(
            record
        )

        if result.get("warnings"):

            warning_samples.append(
                {
                    "image": str(image_path),
                    "split": split,
                    "warnings": result["warnings"],
                }
            )

    # --------------------------------------------------------
    # Cross-split duplicate detection
    # --------------------------------------------------------

    cross_split_duplicates = (
        check_cross_split_duplicates(
            manifest
        )
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    counts = {
        split: 0
        for split in SPLITS
    }

    for item in manifest:

        counts[
            item["split"]
        ] += 1

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    yaml_path = write_yaml()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_path = save_json(
        "manifest.json",
        manifest,
    )

    excluded_path = save_json(
        "excluded_samples.json",
        excluded,
    )

    warning_path = save_json(
        "warning_samples.json",
        warning_samples,
    )

    duplicate_path = save_json(
        "cross_split_duplicates.json",
        cross_split_duplicates,
    )

    summary = {
        "raw_dataset": str(RAW_DATASET),
        "final_dataset": str(FINAL_DATASET),
        "raw_records_from_quality_report": len(
            quality_results
        ),
        "final_image_counts": counts,
        "total_final_images": len(manifest),
        "excluded_count": len(excluded),
        "warning_sample_count": len(
            warning_samples
        ),
        "cross_split_duplicate_groups": len(
            cross_split_duplicates
        ),
        "confirmed_bad_sample": (
            CONFIRMED_BAD_RELATIVE_PATH
        ),
        "yaml": str(yaml_path),
        "manifest": str(manifest_path),
        "excluded_report": str(excluded_path),
        "warning_report": str(warning_path),
        "cross_split_duplicate_report": (
            str(duplicate_path)
        ),
    }

    summary_path = save_json(
        "final_dataset_summary.json",
        summary,
    )

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL DATASET SUMMARY")
    print("=" * 78)

    print(
        f"Train: "
        f"{counts['train']}"
    )

    print(
        f"Validation: "
        f"{counts['val']}"
    )

    print(
        f"Test: "
        f"{counts['test']}"
    )

    print(
        f"Total: "
        f"{len(manifest)}"
    )

    print(
        f"Excluded: "
        f"{len(excluded)}"
    )

    print(
        f"Samples with warnings: "
        f"{len(warning_samples)}"
    )

    print(
        f"Cross-split duplicate groups: "
        f"{len(cross_split_duplicates)}"
    )

    print()
    print("Generated:")

    print(
        f"  YAML:\n"
        f"    {yaml_path}"
    )

    print(
        f"  Manifest:\n"
        f"    {manifest_path}"
    )

    print(
        f"  Exclusions:\n"
        f"    {excluded_path}"
    )

    print(
        f"  Warnings:\n"
        f"    {warning_path}"
    )

    print(
        f"  Cross-split duplicates:\n"
        f"    {duplicate_path}"
    )

    print(
        f"  Summary:\n"
        f"    {summary_path}"
    )

    print()
    print("=" * 78)
    print(
        "RAW DATASET SAFETY"
    )
    print("=" * 78)

    print(
        "RAW DATASET WAS NOT MODIFIED."
    )

    print(
        f"  {RAW_DATASET}"
    )

    print()
    print(
        "FINAL DATASET CREATED."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()