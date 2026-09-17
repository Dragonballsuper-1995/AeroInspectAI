from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import Counter
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")
DATASET_ROOT = PROJECT_ROOT / "datasets"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS
LABEL_EXTENSIONS = {".txt", ".json", ".xml"}


def iter_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_images(root: Path) -> dict:
    images = list(iter_files(root, IMAGE_EXTENSIONS))

    dimensions = Counter()
    broken = []
    hashes = {}

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                dimensions[f"{img.width}x{img.height}"] += 1

            digest = file_md5(image_path)
            hashes.setdefault(digest, []).append(str(image_path))

        except Exception as exc:
            broken.append({
                "path": str(image_path),
                "error": str(exc),
            })

    duplicates = {
        digest: paths
        for digest, paths in hashes.items()
        if len(paths) > 1
    }

    return {
        "image_count": len(images),
        "dimensions": dimensions.most_common(),
        "broken_images": broken,
        "duplicate_groups": duplicates,
    }


def count_extensions(root: Path) -> Counter:
    counts = Counter()

    if not root.exists():
        return counts

    for path in root.rglob("*"):
        if path.is_file():
            counts[path.suffix.lower() or "<no extension>"] += 1

    return counts


def inspect_dataset(dataset_path: Path) -> dict:
    result = {
        "dataset": dataset_path.name,
        "path": str(dataset_path),
        "extensions": dict(count_extensions(dataset_path)),
        "images": inspect_images(dataset_path),
    }

    return result


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    dataset_dirs = sorted(
        path for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    report = {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "datasets": [],
    }

    for dataset in dataset_dirs:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset.name}")
        print(f"{'=' * 70}")

        result = inspect_dataset(dataset)
        report["datasets"].append(result)

        print(f"Images: {result['images']['image_count']}")

        print("\nTop image dimensions:")
        for dimension, count in result["images"]["dimensions"][:15]:
            print(f"  {dimension}: {count}")

        print("\nFile extensions:")
        for extension, count in result["extensions"].items():
            print(f"  {extension}: {count}")

        broken_count = len(result["images"]["broken_images"])
        duplicate_count = len(result["images"]["duplicate_groups"])

        print(f"\nBroken images: {broken_count}")
        print(f"Duplicate groups: {duplicate_count}")

    output_path = PROJECT_ROOT / "dataset_audit.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Audit saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()