#!/usr/bin/env python3
"""Minimal EasyOCR pass over original PoleTag images.

No preprocessing. No postprocessing.
Runs one OCR call per image and picks the highest-confidence raw text.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import easyocr


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=base_dir / "images",
        help="Directory containing images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=base_dir / "easyocr_models",
        help="EasyOCR model directory.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR if available.",
    )
    return parser.parse_args()


def iter_pole_images(images_dir: Path) -> list[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts and p.name.startswith("PoleTag_")
    )


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(
        ["en"],
        gpu=args.gpu,
        model_storage_directory=str(model_dir),
        download_enabled=True,
        verbose=False,
    )

    image_paths = iter_pole_images(images_dir)
    if not image_paths:
        print(f"No PoleTag images found in {images_dir}")
        return 1

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        results = reader.readtext(str(image_path), detail=1, paragraph=False)
        if results:
            best = max(results, key=lambda item: float(item[2]))
            best_text = str(best[1])
            best_confidence = float(best[2])
            raw = [
                {"text": str(item[1]), "confidence": float(item[2])}
                for item in results
            ]
        else:
            best_text = ""
            best_confidence = 0.0
            raw = []

        rows.append(
            {
                "image": image_path.name,
                "best_text": best_text,
                "best_confidence": round(best_confidence, 6),
                "raw_results": raw,
            }
        )
        print(f"{image_path.name}: {best_text or 'NO_TEXT'} ({best_confidence:.4f})")

    csv_path = output_dir / "plate_predictions2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "best_text", "best_confidence", "raw_results"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "best_text": row["best_text"],
                    "best_confidence": row["best_confidence"],
                    "raw_results": json.dumps(row["raw_results"]),
                }
            )

    json_path = output_dir / "plate_predictions2.json"
    json_path.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
