#!/usr/bin/env python3
"""Run EasyOCR over pole plate images and save plate text predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import certifi
import cv2
import numpy as np

# Ensure Python uses a modern CA bundle when EasyOCR downloads model files.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import easyocr


PLATE_LIKE_PATTERNS = (
    re.compile(r"^[A-Z]{1,4}\d{2,7}[A-Z]?$"),
    re.compile(r"^\d{1,4}-\d{2,6}$"),
    re.compile(r"^\d{4,8}$"),
)

COLOR_ADJUSTMENTS = (
    ("adj_b-30_c0.90_s0.85", -30, 0.90, 0.85),
    ("adj_b-15_c1.00_s0.90", -15, 1.00, 0.90),
    ("adj_b+0_c1.00_s1.00", 0, 1.00, 1.00),
    ("adj_b+15_c1.08_s1.10", 15, 1.08, 1.10),
    ("adj_b+30_c1.15_s1.20", 30, 1.15, 1.20),
    ("adj_b+10_c1.20_s0.95", 10, 1.20, 0.95),
)


@dataclass(frozen=True)
class Candidate:
    text: str
    confidence: float
    score: float
    variant: str
    bbox: list[list[float]]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=base_dir / "images",
        help="Directory containing pole plate images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output",
        help="Directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=base_dir / "easyocr_models",
        help="Directory where EasyOCR model weights are stored.",
    )
    parser.add_argument(
        "--allowlist",
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        help="Characters allowed during OCR decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Max number of candidate plate strings to keep per image.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.40,
        help="Minimum candidate score to keep in outputs.",
    )
    parser.add_argument(
        "--profile",
        choices=("mobile", "balanced", "full"),
        default="mobile",
        help="Latency/quality profile. `mobile` is fastest.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR if available.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Convert OCR text to an uppercase plate-like token."""
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = "".join(ch for ch in text if ch.isalnum() or ch == "-")
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def is_plate_like(text: str) -> bool:
    if len(text) < 4 or len(text) > 12:
        return False
    if not any(ch.isdigit() for ch in text):
        return False
    alnum_ratio = sum(ch.isalnum() for ch in text) / max(1, len(text))
    return alnum_ratio >= 0.75


def score_candidate(text: str, confidence: float) -> float:
    score = confidence
    if 4 <= len(text) <= 10:
        score += 0.10
    if any(ch.isalpha() for ch in text) and any(ch.isdigit() for ch in text):
        score += 0.10
    for pattern in PLATE_LIKE_PATTERNS:
        if pattern.fullmatch(text):
            score += 0.25
            break
    if text.count("-") > 1:
        score -= 0.08
    return score


def adjust_brightness_contrast_saturation(
    image_bgr: np.ndarray,
    brightness: int,
    contrast: float,
    saturation_scale: float,
) -> np.ndarray:
    adjusted = cv2.convertScaleAbs(image_bgr, alpha=contrast, beta=brightness)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def image_variants(image_bgr: np.ndarray, profile: str) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {"orig_color": image_bgr}
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if profile == "mobile":
        # Minimal variant set for lowest latency on constrained hardware.
        variants["clahe"] = clahe
        return variants

    if profile == "balanced":
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.update(
            {
                "gray": gray,
                "clahe": clahe,
                "binary": binary,
                "bilateral": bilateral,
            }
        )
        return variants

    for name, brightness, contrast, saturation in COLOR_ADJUSTMENTS:
        variants[name] = adjust_brightness_contrast_saturation(
            image_bgr=image_bgr,
            brightness=brightness,
            contrast=contrast,
            saturation_scale=saturation,
        )

    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    sharpen = cv2.filter2D(
        gray,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    upscaled = cv2.resize(clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.update(
        {
            "gray": gray,
            "clahe": clahe,
            "bilateral": bilateral,
            "sharpen": sharpen,
            "binary": binary,
            "adaptive": adaptive,
            "upscaled": upscaled,
        }
    )
    return variants


def readtext_params(profile: str, allowlist: str) -> dict[str, object]:
    base: dict[str, object] = {
        "detail": 1,
        "paragraph": False,
        "allowlist": allowlist,
        "min_size": 8,
        "text_threshold": 0.55,
        "low_text": 0.30,
        "link_threshold": 0.30,
        "canvas_size": 2560,
        "mag_ratio": 1.5,
    }
    if profile == "mobile":
        base.update(
            {
                "decoder": "greedy",
                "min_size": 12,
                "text_threshold": 0.65,
                "low_text": 0.35,
                "link_threshold": 0.35,
                "canvas_size": 1280,
                "mag_ratio": 1.0,
            }
        )
    elif profile == "balanced":
        base.update(
            {
                "decoder": "greedy",
                "canvas_size": 1920,
                "mag_ratio": 1.2,
            }
        )
    else:
        base.update(
            {
                "decoder": "beamsearch",
                "beamWidth": 5,
            }
        )
    return base


def extract_candidates(
    reader: easyocr.Reader,
    image_path: Path,
    allowlist: str,
    profile: str,
) -> list[Candidate]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    variants = image_variants(image, profile=profile)
    ocr_kwargs = readtext_params(profile=profile, allowlist=allowlist)
    best_by_text: dict[str, Candidate] = {}
    for variant_name, variant_img in variants.items():
        results = reader.readtext(variant_img, **ocr_kwargs)
        for bbox, raw_text, confidence in results:
            text = normalize_text(raw_text)
            if not text or not is_plate_like(text):
                continue
            score = score_candidate(text, float(confidence))
            current = best_by_text.get(text)
            candidate = Candidate(
                text=text,
                confidence=float(confidence),
                score=score,
                variant=variant_name,
                bbox=[[float(x), float(y)] for x, y in bbox],
            )
            if current is None or (
                candidate.confidence > current.confidence
                or (
                    candidate.confidence == current.confidence
                    and candidate.score > current.score
                )
            ):
                best_by_text[text] = candidate

    return sorted(
        best_by_text.values(),
        key=lambda c: (c.confidence, c.score),
        reverse=True,
    )


def write_outputs(
    records: list[dict[str, object]],
    output_dir: Path,
) -> tuple[Path, Path]:
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
        for record in records:
            writer.writerow(record)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": records,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return csv_path, json_path


def iter_images(images_dir: Path) -> Iterable[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
    )


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EASYOCR_MODULE_PATH"] = str(model_dir)
    reader = easyocr.Reader(
        ["en"],
        gpu=args.gpu,
        model_storage_directory=str(model_dir),
        download_enabled=True,
        verbose=False,
    )

    image_paths = list(iter_images(images_dir))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return 1

    print(f"Using profile: {args.profile}")

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        candidates = extract_candidates(
            reader,
            image_path,
            args.allowlist,
            profile=args.profile,
        )
        filtered = [c for c in candidates if c.score >= args.min_score][: args.top_k]
        if filtered:
            best = max(filtered, key=lambda c: (c.confidence, c.score))
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
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Model dir:  {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
