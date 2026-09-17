"""Run a small, repeatable dataset-to-API review smoke test.

This deliberately samples the validation split, not the held-out test split.
Each request is persisted as a normal inspection so the same results can be
opened from the frontend history page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "aeroinspect_crack_v1" / "images" / "val"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "aeroinspect_crack_v1" / "labels" / "val"


def choose_evenly(items: list[Path], limit: int) -> list[Path]:
    if limit >= len(items):
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (limit - 1))] for index in range(limit)]


def validate_result(result: dict[str, Any], base_url: str) -> None:
    width = result["summary"]["image_width"]
    height = result["summary"]["image_height"]
    assert width > 0 and height > 0
    assert result["status"] == "completed"
    assert result["summary"]["detections"] == len(result["detections"])

    for detection in result["detections"]:
        assert 0 <= detection["confidence"] <= 1
        box = detection["bounding_box"]
        assert 0 <= box["x1"] <= box["x2"] <= width
        assert 0 <= box["y1"] <= box["y2"] <= height
        mask = detection["mask"]
        if mask:
            assert mask["area_pixels"] > 0
            assert 0 < mask["area_ratio"] <= 1
            assert len(mask["polygon"]) >= 3
            assert all(0 <= x <= width and 0 <= y <= height for x, y in mask["polygon"])

    for key in ("original_url", "annotated_url"):
        image_response = requests.get(f"{base_url}{result['image'][key]}", timeout=15)
        image_response.raise_for_status()
        assert image_response.headers.get("content-type", "").startswith("image/")
        assert len(image_response.content) > 100


def run(base_url: str, limit: int, confidence: float) -> list[dict[str, Any]]:
    health = requests.get(f"{base_url}/api/v1/health", timeout=5)
    health.raise_for_status()
    assert health.json()["status"] == "healthy"

    images = sorted(path for path in DEFAULT_IMAGES.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise RuntimeError(f"No validation images found in {DEFAULT_IMAGES}")

    report: list[dict[str, Any]] = []
    for image_path in choose_evenly(images, limit):
        label_path = DEFAULT_LABELS / f"{image_path.stem}.txt"
        ground_truth_instances = len(label_path.read_text(encoding="utf-8").splitlines()) if label_path.exists() else 0
        with image_path.open("rb") as image_file:
            response = requests.post(
                f"{base_url}/api/v1/inspect",
                files={"file": (image_path.name, image_file, "image/jpeg")},
                data={"confidence": str(confidence)},
                timeout=120,
            )
        response.raise_for_status()
        result = response.json()
        validate_result(result, base_url)
        mask_count = sum(detection["mask"] is not None for detection in result["detections"])
        item = {
            "image": image_path.name,
            "inspection_id": result["inspection_id"],
            "ground_truth_instances": ground_truth_instances,
            "predictions": len(result["detections"]),
            "masks": mask_count,
            "peak_confidence": result["summary"]["max_confidence"],
            "inference_time_ms": result["inference"]["inference_time_ms"],
        }
        report.append(item)
        print(
            f"PASS {item['inspection_id']}  {image_path.name}  "
            f"GT={ground_truth_instances} predicted={item['predictions']} masks={mask_count}  "
            f"peak={item['peak_confidence']:.3f} inference={item['inference_time_ms']:.1f}ms"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    report = run(args.base_url.rstrip("/"), args.limit, args.confidence)
    print(json.dumps({"status": "passed", "cases": report}, indent=2))


if __name__ == "__main__":
    main()
