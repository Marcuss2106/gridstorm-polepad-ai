#!/usr/bin/env python3
"""Split each image in split_images into 4 quadrant images.

Outputs are written to the parent images directory with `_q1.._q4` suffixes.
Quadrant order:
  q1 = top-left
  q2 = top-right
  q3 = bottom-left
  q4 = bottom-right
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_split_dir = base_dir / "images" / "split_images"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=default_split_dir,
        help="Directory containing source images to split into quadrants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_split_dir.parent,
        help="Directory to store generated quadrant images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quadrant files if they already exist.",
    )
    return parser.parse_args()


def iter_images(split_dir: Path) -> list[Path]:
    files = [p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(files, key=lambda p: p.name.lower())


def quadrant_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    mid_x = width // 2
    mid_y = height // 2
    return [
        (0, 0, mid_x, mid_y),          # q1 top-left
        (mid_x, 0, width, mid_y),      # q2 top-right
        (0, mid_y, mid_x, height),     # q3 bottom-left
        (mid_x, mid_y, width, height), # q4 bottom-right
    ]


def split_image_to_quadrants(
    image_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> int:
    created = 0
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        boxes = quadrant_boxes(width, height)

        stem = image_path.stem.replace(" ", "_")
        ext = image_path.suffix.lower()

        for idx, box in enumerate(boxes, start=1):
            out_name = f"{stem}_q{idx}{ext}"
            out_path = output_dir / out_name
            if out_path.exists() and not overwrite:
                continue

            crop = image.crop(box)
            crop.save(out_path)
            created += 1
    return created


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists() or not split_dir.is_dir():
        print(f"Split images directory not found: {split_dir}")
        return 1

    source_images = iter_images(split_dir)
    if not source_images:
        print(f"No source images found in: {split_dir}")
        return 1

    total_created = 0
    for src in source_images:
        total_created += split_image_to_quadrants(src, output_dir, args.overwrite)

    print(f"Source images: {len(source_images)}")
    print(f"Quadrants expected: {len(source_images) * 4}")
    print(f"Quadrants created: {total_created}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
