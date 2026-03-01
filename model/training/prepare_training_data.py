#!/usr/bin/env python3
"""Prepare EasyOCR fine-tuning manifests from labeled pole plate data.

This script uses full images only (no ROI cropping).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.common import (
    derive_charset,
    filter_rows,
    load_labeled_samples,
    save_charset,
    save_json,
    save_manifest,
    split_rows,
)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=model_dir / "dataset" / "plate_labels.csv",
        help="CSV produced by label_plate_dataset.py",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=model_dir / "images",
        help="Directory containing full training images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_dir / "dataset" / "prepared",
        help="Output directory for manifests and metadata.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--min-label-length",
        type=int,
        default=1,
        help="Minimum label length to keep.",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=16,
        help="Maximum label length to keep.",
    )
    parser.add_argument(
        "--allow-chars",
        default=None,
        help=(
            "Optional character allowlist. Any sample containing characters outside "
            "this set is dropped."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    labels_csv = args.labels_csv.resolve()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing_files = load_labeled_samples(labels_csv=labels_csv, images_dir=images_dir)

    filtered_rows = filter_rows(
        rows,
        min_label_length=args.min_label_length,
        max_label_length=args.max_label_length,
        allow_chars=args.allow_chars,
    )

    train_rows, val_rows, test_rows = split_rows(
        rows=filtered_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_manifest = output_dir / "train_manifest.csv"
    val_manifest = output_dir / "val_manifest.csv"
    test_manifest = output_dir / "test_manifest.csv"
    charset_path = output_dir / "charset.txt"
    stats_path = output_dir / "stats.json"

    save_manifest(train_rows, train_manifest)
    save_manifest(val_rows, val_manifest)
    save_manifest(test_rows, test_manifest)

    charset = derive_charset(filtered_rows)
    save_charset(charset, charset_path)

    dropped = len(rows) - len(filtered_rows)
    payload = {
        "labels_csv": str(labels_csv),
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "total_labeled_rows": len(rows),
        "total_missing_files": len(missing_files),
        "total_after_filter": len(filtered_rows),
        "dropped_by_filter": dropped,
        "split_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "min_label_length": args.min_label_length,
        "max_label_length": args.max_label_length,
        "allow_chars": args.allow_chars,
        "seed": args.seed,
        "charset": charset,
        "manifest_paths": {
            "train": str(train_manifest),
            "val": str(val_manifest),
            "test": str(test_manifest),
        },
        "missing_files": missing_files,
    }
    save_json(payload, stats_path)

    print("Prepared manifests:")
    print(f"- train: {train_manifest}")
    print(f"- val:   {val_manifest}")
    print(f"- test:  {test_manifest}")
    print(f"- charset: {charset_path}")
    print(f"- stats:   {stats_path}")
    print(
        "Counts -> "
        f"labeled={len(rows)} filtered={len(filtered_rows)} "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
