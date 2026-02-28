#!/usr/bin/env python3
"""Run EasyOCR over pole plate images and save plate text predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

TEMPLATE_PATTERNS = (
    (re.compile(r"^PD\d{5}$"), 0.40),
    (re.compile(r"^[A-Z]-\d{4}$"), 0.35),
    (re.compile(r"^\d{6}$"), 0.25),
    (re.compile(r"^\d{5}$"), 0.20),
    (re.compile(r"^[A-Z]{2,4}\d{4,6}$"), 0.15),
)

LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "C": "6",
}

DASH_PLATE_FIRST_CHAR_MAP = {
    "6": "C",
    "3": "C",
    "G": "C",
    "0": "C",
    "O": "C",
    "C": "C",
    "D": "D",
}

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


@dataclass
class AggregateCandidate:
    text: str
    best_confidence: float
    best_score: float
    best_variant: str
    best_bbox: list[list[float]]
    hit_count: int = 0
    weighted_votes: float = 0.0
    confidence_sum: float = 0.0
    variants: set[str] | None = None

    def __post_init__(self) -> None:
        if self.variants is None:
            self.variants = set()


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
    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count < 3:
        return False
    if "-" in text and not re.fullmatch(r"^[A-Z0-9]-[A-Z0-9]{4,6}$", text):
        return False
    alnum_ratio = sum(ch.isalnum() for ch in text) / max(1, len(text))
    return alnum_ratio >= 0.75


def template_bonus(text: str) -> float:
    for pattern, bonus in TEMPLATE_PATTERNS:
        if pattern.fullmatch(text):
            return bonus
    return 0.0


def score_candidate(text: str, confidence: float) -> float:
    score = confidence
    if 4 <= len(text) <= 10:
        score += 0.10
    if any(ch.isalpha() for ch in text) and any(ch.isdigit() for ch in text):
        score += 0.10
    if any(pattern.fullmatch(text) for pattern in PLATE_LIKE_PATTERNS):
        score += 0.15
    score += template_bonus(text)
    if text.count("-") > 1:
        score -= 0.08
    if "-" in text and not re.fullmatch(r"^[A-Z]-\d{4}$", text):
        score -= 0.05
    return score


def letters_to_digits(text: str) -> str:
    return "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in text)


def candidate_forms(text: str) -> list[tuple[str, float]]:
    """Generate confusion-aware normalized alternatives with penalties."""
    forms: dict[str, float] = {text: 0.0}

    # Remove one noisy leading character before common two-letter prefixes.
    if re.fullmatch(r"^[IJ1L](?:PD|IPD|JPD|[A-Z]{1,3})\d{3,7}$", text):
        forms[text[1:]] = min(forms.get(text[1:], 1.0), 0.03)

    # Normalize common noisy PD prefixes (e.g. IPD41459 -> PD41459).
    m = re.fullmatch(r"^[IJ1L]?P?D([A-Z0-9]{4,6})$", text)
    if m:
        normalized = "PD" + letters_to_digits(m.group(1))
        forms[normalized] = min(forms.get(normalized, 1.0), 0.02)

    # Normalize A-1234 style plates where first character is often confused.
    m = re.fullmatch(r"^([A-Z0-9])-([A-Z0-9]{4})$", text)
    if m:
        first, tail = m.groups()
        tail_digits = letters_to_digits(tail)
        mapped_first = DASH_PLATE_FIRST_CHAR_MAP.get(first)
        if mapped_first:
            normalized = f"{mapped_first}-{tail_digits}"
            forms[normalized] = min(forms.get(normalized, 1.0), 0.05)
        if first.isalpha():
            normalized = f"{first}-{tail_digits}"
            forms[normalized] = min(forms.get(normalized, 1.0), 0.02)

    # Preserve alpha prefix and force numeric suffix when shape matches.
    m = re.fullmatch(r"^([A-Z]{1,4})([A-Z0-9]{3,7})$", text)
    if m:
        prefix, suffix = m.groups()
        # Avoid converting plain words like WARNING into fake plate-looking strings.
        if sum(ch.isdigit() for ch in suffix) >= 2:
            normalized = prefix + letters_to_digits(suffix)
            forms[normalized] = min(forms.get(normalized, 1.0), 0.02)

    # Numeric plate fallback with OCR letter confusions coerced to digits.
    if re.fullmatch(r"^[A-Z0-9]{5,8}$", text):
        numeric = letters_to_digits(text)
        if re.fullmatch(r"^\d{5,8}$", numeric):
            forms[numeric] = min(forms.get(numeric, 1.0), 0.04)
            # Leading digit on embossed tags is frequently confused with 7.
            if len(numeric) == 6 and numeric[0] in {"5", "6", "8"}:
                leading7 = "7" + numeric[1:]
                forms[leading7] = min(forms.get(leading7, 1.0), 0.08)

    return sorted(forms.items(), key=lambda item: item[1])


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


def image_variants(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {"orig_color": image_bgr}
    for name, brightness, contrast, saturation in COLOR_ADJUSTMENTS:
        variants[name] = adjust_brightness_contrast_saturation(
            image_bgr=image_bgr,
            brightness=brightness,
            contrast=contrast,
            saturation_scale=saturation,
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
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


def detect_plate_rois(image_bgr: np.ndarray, max_rois: int = 10) -> list[np.ndarray]:
    """Find likely plate regions so OCR sees tighter crops."""
    h, w = image_bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 40 or bh < 20:
            continue
        area = bw * bh
        area_ratio = area / img_area
        if area_ratio < 0.01 or area_ratio > 0.75:
            continue
        aspect = bw / max(1, bh)
        rect_like = 1.4 <= aspect <= 9.5
        round_like = 0.75 <= aspect <= 1.35 and area_ratio >= 0.02
        if not (rect_like or round_like):
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        # Favor shape complexity close to plates/signs.
        shape_score = 1.0
        if 3 <= len(approx) <= 10:
            shape_score += 0.25
        candidates.append((area * shape_score, (x, y, bw, bh)))

    candidates.sort(reverse=True, key=lambda item: item[0])
    rois: list[np.ndarray] = []
    seen_boxes: list[tuple[int, int, int, int]] = []
    for _, (x, y, bw, bh) in candidates:
        # Basic overlap suppression.
        suppress = False
        for ox, oy, ow, oh in seen_boxes:
            ix1, iy1 = max(x, ox), max(y, oy)
            ix2, iy2 = min(x + bw, ox + ow), min(y + bh, oy + oh)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            union = (bw * bh) + (ow * oh) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0.65:
                suppress = True
                break
        if suppress:
            continue

        pad_x = int(0.08 * bw)
        pad_y = int(0.12 * bh)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rois.append(roi)
        seen_boxes.append((x, y, bw, bh))
        if len(rois) >= max_rois:
            break
    return rois


def extract_candidates(
    reader: easyocr.Reader,
    image_path: Path,
    allowlist: str,
) -> list[Candidate]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    region_images: list[tuple[str, np.ndarray]] = [("full", image)]
    for idx, roi in enumerate(detect_plate_rois(image), start=1):
        region_images.append((f"roi{idx}", roi))

    aggregates: dict[str, AggregateCandidate] = {}
    for region_name, region_image in region_images:
        variants = image_variants(region_image)
        for variant_name, variant_img in variants.items():
            results = reader.readtext(
                variant_img,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                beamWidth=5,
                allowlist=allowlist,
                min_size=8,
                text_threshold=0.55,
                low_text=0.30,
                link_threshold=0.30,
                canvas_size=2560,
                mag_ratio=1.5,
            )
            for bbox, raw_text, confidence in results:
                raw_normalized = normalize_text(raw_text)
                if not raw_normalized:
                    continue
                for text, penalty in candidate_forms(raw_normalized):
                    if not is_plate_like(text):
                        continue
                    adjusted_confidence = max(0.0, float(confidence) - penalty)
                    score = score_candidate(text, adjusted_confidence)
                    vote_weight = adjusted_confidence + (0.35 * template_bonus(text))
                    vote_weight += 0.05 if region_name != "full" else 0.0

                    agg = aggregates.get(text)
                    if agg is None:
                        agg = AggregateCandidate(
                            text=text,
                            best_confidence=adjusted_confidence,
                            best_score=score,
                            best_variant=f"{region_name}:{variant_name}",
                            best_bbox=[[float(x), float(y)] for x, y in bbox],
                        )
                        aggregates[text] = agg

                    agg.hit_count += 1
                    agg.confidence_sum += adjusted_confidence
                    agg.weighted_votes += vote_weight
                    agg.variants.add(f"{region_name}:{variant_name}")

                    if adjusted_confidence > agg.best_confidence or (
                        adjusted_confidence == agg.best_confidence
                        and score > agg.best_score
                    ):
                        agg.best_confidence = adjusted_confidence
                        agg.best_score = score
                        agg.best_variant = f"{region_name}:{variant_name}"
                        agg.best_bbox = [[float(x), float(y)] for x, y in bbox]

    candidates: list[Candidate] = []
    for agg in aggregates.values():
        variant_support = min(0.18, 0.04 * max(0, len(agg.variants) - 1))
        hit_support = min(0.22, 0.05 * math.log1p(max(0, agg.hit_count - 1)))
        vote_support = min(0.20, 0.03 * agg.weighted_votes)
        consensus_score = agg.best_score + variant_support + hit_support + vote_support
        candidates.append(
            Candidate(
                text=agg.text,
                confidence=agg.best_confidence,
                score=consensus_score,
                variant=agg.best_variant,
                bbox=agg.best_bbox,
            )
        )

    return sorted(
        candidates,
        key=lambda c: (c.score, c.confidence),
        reverse=True,
    )


def pattern_priority(text: str) -> float:
    """Prefer known plate templates over generic OCR tokens."""
    if re.fullmatch(r"^PD\d{5}$", text):
        return 5.0
    if re.fullmatch(r"^[A-Z]-\d{4}$", text):
        return 4.0
    if re.fullmatch(r"^\d{6}$", text):
        return 3.0
    if re.fullmatch(r"^\d{5}$", text):
        return 2.0
    if re.fullmatch(r"^[A-Z]{2,4}\d{4,6}$", text):
        return 1.5
    return 0.0


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

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        candidates = extract_candidates(reader, image_path, args.allowlist)
        filtered = [c for c in candidates if c.score >= args.min_score][: args.top_k]
        if filtered:
            best = max(
                filtered,
                key=lambda c: (pattern_priority(c.text), c.score, c.confidence),
            )
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
