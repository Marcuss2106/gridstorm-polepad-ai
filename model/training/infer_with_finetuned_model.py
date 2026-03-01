#!/usr/bin/env python3
"""Run a fine-tuned EasyOCR recognizer over images with preprocessing + postprocessing.

This inference path keeps preprocessing and postprocessing behavior from plate_ocr.py,
but uses only full images (no ROI cropping).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from easyocr.recognition import AlignCollate

from training.common import (
    build_converter,
    build_model,
    choose_device,
    extract_state_dict,
    gather_variant_predictions,
    list_images,
    load_charset,
    rank_candidate_strings,
    selection_key,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_dir / "training" / "runs" / "easyocr_plate_ft" / "best.pt",
        help="Path to fine-tuned recognizer checkpoint.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=model_dir / "images",
        help="Directory of input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_dir / "output",
        help="Directory for prediction outputs.",
    )
    parser.add_argument(
        "--charset-file",
        type=Path,
        default=None,
        help="Optional charset file override.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top candidates to keep per image.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum postprocessed score to retain a candidate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=None,
        help="Override max label length from checkpoint config.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Optional comma-separated preprocessing variant names to run. "
            "Empty means all variants from plate_ocr.image_variants."
        ),
    )
    return parser.parse_args()


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "plate_predictions.csv"
    json_path = output_dir / "plate_predictions.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "best_plate",
                "best_score",
                "best_confidence",
                "best_variant",
                "plate_candidates",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return csv_path, json_path


def main() -> int:
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    device = choose_device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}

    if args.charset_file:
        charset = load_charset(args.charset_file.resolve())
    else:
        charset = payload.get("charset") or config.get("charset")
        if not charset:
            charset_path = config.get("charset_file")
            if charset_path:
                charset = load_charset(Path(charset_path).resolve())
            else:
                raise RuntimeError(
                    "Charset missing from checkpoint payload. Pass --charset-file."
                )

    recog_network = config.get("recog_network", "generation2")
    input_channel = int(config.get("input_channel", 1))
    output_channel = int(config.get("output_channel", 256))
    hidden_size = int(config.get("hidden_size", 256))
    img_h = int(config.get("img_h", 64))
    img_w = int(config.get("img_w", 256))
    max_label_length = int(
        args.max_label_length
        if args.max_label_length is not None
        else config.get("max_label_length", 16)
    )

    converter = build_converter(charset)
    model = build_model(
        recog_network=recog_network,
        input_channel=input_channel,
        output_channel=output_channel,
        hidden_size=hidden_size,
        num_class=len(converter.character),
    ).to(device)

    state_dict = strip_module_prefix(extract_state_dict(payload))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    align_collate = AlignCollate(
        imgH=img_h,
        imgW=img_w,
        keep_ratio_with_pad=True,
        adjust_contrast=0.0,
    )

    variant_filter: set[str] | None = None
    if args.variants.strip():
        variant_filter = {part.strip() for part in args.variants.split(",") if part.strip()}

    image_paths = list_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                rows.append(
                    {
                        "image": image_path.name,
                        "best_plate": "",
                        "best_score": "",
                        "best_confidence": "",
                        "best_variant": "",
                        "plate_candidates": "",
                    }
                )
                continue

            raw_predictions = gather_variant_predictions(
                model=model,
                converter=converter,
                image_bgr=image,
                align_collate=align_collate,
                device=device,
                max_label_length=max_label_length,
                variant_filter=variant_filter,
            )

            ranked_candidates = rank_candidate_strings(raw_predictions)
            filtered = [c for c in ranked_candidates if c.score >= args.min_score]
            filtered = sorted(filtered, key=selection_key, reverse=True)[: args.top_k]

            if filtered:
                best = filtered[0]
                row = {
                    "image": image_path.name,
                    "best_plate": best.text,
                    "best_score": f"{best.score:.4f}",
                    "best_confidence": f"{best.confidence:.4f}",
                    "best_variant": best.variant,
                    "plate_candidates": "|".join(c.text for c in filtered),
                }
            else:
                row = {
                    "image": image_path.name,
                    "best_plate": "",
                    "best_score": "",
                    "best_confidence": "",
                    "best_variant": "",
                    "plate_candidates": "",
                }

            rows.append(row)
            print(f"{image_path.name}: {row['plate_candidates'] or 'NO_PLATE_FOUND'}")

    csv_path, json_path = write_outputs(rows, output_dir)
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
