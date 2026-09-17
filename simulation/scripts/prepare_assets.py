#!/usr/bin/env python3
"""Copy five deterministic *training* images for simulator-only visual panels.

These copies are explicitly demonstration assets. The held-out test split is
never read, and this script does not alter source datasets.
"""

from __future__ import annotations

import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "datasets" / "aeroinspect_crack_v1" / "images" / "train"
TARGET = REPO / "simulation" / "gazebo" / "models" / "inspection_wall" / "materials" / "textures"


def main() -> None:
    images = sorted(path for path in SOURCE.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < 5:
        raise SystemExit(f"Expected at least five training images in {SOURCE}")
    TARGET.mkdir(parents=True, exist_ok=True)
    # Spread selections through the training set, keeping output deterministic.
    selections = [images[index * (len(images) - 1) // 4] for index in range(5)]
    for number, source in enumerate(selections, start=1):
        target = TARGET / f"panel_{number:02d}.jpg"
        shutil.copy2(source, target)
        print(f"{target.name} <- training/{source.name}")


if __name__ == "__main__":
    main()
