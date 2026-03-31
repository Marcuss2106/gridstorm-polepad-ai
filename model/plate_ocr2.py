#!/usr/bin/env python3
"""Multi-engine OCR benchmark over PoleTag images.

Runs selected OCR engines across selected variants and scores predictions with
lightweight postprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

RE_PD_PLAIN = re.compile(r"^PD\d{5}$")
RE_PD_DASH = re.compile(r"^PD-\d{4,5}$")
RE_L_DASH4 = re.compile(r"^[A-Z]-\d{4}$")
RE_DIGIT5 = re.compile(r"^\d{5}$")
RE_DIGIT6 = re.compile(r"^\d{6}$")
RE_DIGIT_5_6 = re.compile(r"^\d{5,6}$")

PLATE_PATTERNS = (
    RE_PD_PLAIN,
    RE_PD_DASH,
    RE_L_DASH4,
    re.compile(r"^\d{5,7}$"),
    re.compile(r"^[A-Z]{1,3}\d{3,7}$"),
)

NON_PLATE_WORDS = {
    "WARNING",
    "CUSTOMER",
    "GENERATION",
    "CONNECTED",
    "SCE",
    "MUNI",
}

NON_PLATE_PREFIXES = (
    "MUNI",
    "SCE",
    "WARN",
    "CUSTOM",
    "GENERAT",
    "CONNECT",
)

LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

POLETAG_RE = re.compile(r"^PoleTag_(\d+)")
SUPPORTED_ENGINES = {"easyocr", "paddleocr", "tesseract"}
SUPPORTED_VARIANTS = {"raw", "text_light"}
AMBIGUOUS_OCR_CHARS = set(LETTER_TO_DIGIT.keys()) | set(LETTER_TO_DIGIT.values())


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
        default=base_dir,
        help="Directory for OCR model/cache files.",
    )
    parser.add_argument(
        "--engines",
        default="easyocr,paddleocr",
        help="Comma-separated engines: easyocr,paddleocr,tesseract",
    )
    parser.add_argument(
        "--variants",
        default="raw,text_light",
        help="Comma-separated variants: raw,text_light",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR engines that support it.",
    )
    parser.add_argument(
        "--allowlist",
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        help="Characters allowed by OCR decoder.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Path to tesseract binary (optional if tesseract is on PATH).",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=11,
        help="Tesseract page segmentation mode.",
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=1,
        help="Tesseract OCR engine mode.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=base_dir / "dataset" / "plate_labels.csv",
        help="Ground-truth CSV to report exact-match accuracy.",
    )
    parser.add_argument(
        "--save-paddle-unsure-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save images with boxes around PaddleOCR unsure characters.",
    )
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save annotated overlay images (character sub-boxes coloured by "
            "per-character confidence) to <output-dir>/overlays/."
        ),
    )
    parser.add_argument(
        "--paddle-unsure-threshold",
        type=float,
        default=0.93,
        help="Line confidence threshold below which PaddleOCR characters are marked unsure.",
    )
    parser.add_argument(
        "--paddle-max-reruns",
        type=int,
        default=1,
        help="Max PaddleOCR retry passes per image for raw variant (0 disables retries).",
    )
    parser.add_argument(
        "--paddle-time-cap-seconds",
        type=float,
        default=5.0,
        help="Soft per-image time cap for PaddleOCR raw path; retries stop once elapsed time reaches this limit.",
    )
    parser.add_argument(
        "--paddle-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to PaddleOCR input image before preprocessing (e.g. 0.6).",
    )
    return parser.parse_args()


def iter_pole_images(images_dir: Path) -> list[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts and p.name.startswith("PoleTag_")
    )


def normalize_token(text: str) -> str:
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = "".join(ch for ch in text if ch.isalnum() or ch == "-")
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def letters_to_digits(text: str) -> str:
    return "".join(LETTER_TO_DIGIT.get(ch, ch) for ch in text)


def canonicalize(text: str) -> str:
    token = normalize_token(text)
    if not token:
        return ""

    for noisy_prefix in ("IPD", "PPD", "JPD", "1PD", "LPD"):
        if token.startswith(noisy_prefix) and len(token) >= 7:
            token = "PD" + token[len(noisy_prefix) :]
            break

    token = token.lstrip("~")
    token = token.rstrip("-")

    digit_count = sum(ch.isdigit() for ch in token)
    if digit_count >= max(3, len(token) - 2):
        token = letters_to_digits(token)

    m = re.fullmatch(r"([A-Z0-9])-(\d{4})", token)
    if m:
        lead, tail = m.groups()
        if lead in {"6", "G", "0", "O"}:
            token = f"C-{tail}"

    # Fix misread PD- prefix: "20-DDDD" or "P0-DDDD" etc -> "PD-DDDD"
    m2 = re.fullmatch(r"([0-9P][0-9D])-(\d{4})", token)
    if m2:
        prefix, digits = m2.groups()
        if prefix not in ("PD",):
            token = f"PD-{digits}"

    return token


def is_noise_token(token: str, raw_upper: str) -> bool:
    if token in NON_PLATE_WORDS or raw_upper in NON_PLATE_WORDS:
        return True
    if token.startswith(NON_PLATE_PREFIXES) or raw_upper.startswith(NON_PLATE_PREFIXES):
        return True
    return False


def format_prior_bonus(token: str) -> float:
    if RE_PD_PLAIN.fullmatch(token):
        return 0.24
    if RE_PD_DASH.fullmatch(token):
        return 0.22
    if RE_DIGIT6.fullmatch(token):
        return 0.14
    if RE_DIGIT5.fullmatch(token):
        return 0.12
    if RE_L_DASH4.fullmatch(token):
        return 0.04
    return 0.0


def plate_score(text: str, confidence: float) -> float:
    token = canonicalize(text)
    if not token:
        return -1.0

    score = float(confidence)
    upper_raw = normalize_token(text)

    if is_noise_token(token, upper_raw):
        score -= 1.2

    digit_count = sum(ch.isdigit() for ch in token)
    alpha_count = sum(ch.isalpha() for ch in token)
    has_dash = "-" in token
    has_alpha = alpha_count > 0

    if digit_count == 0:
        score -= 1.0
    if token.isdigit() and len(token) <= 4:
        score -= 0.35
    if has_alpha and alpha_count > digit_count and not token.startswith("PD"):
        score -= 0.45
    if 4 <= len(token) <= 9:
        score += 0.10
    if token.isdigit() and 5 <= len(token) <= 7:
        score += 0.06
    if has_alpha and digit_count >= 2 and not has_dash:
        score += 0.08
    if has_dash and RE_L_DASH4.fullmatch(token):
        score += 0.05
    if any(pattern.fullmatch(token) for pattern in PLATE_PATTERNS):
        score += 0.12
    score += format_prior_bonus(token)

    return score


def append_note(existing: str, note: str) -> str:
    if not note:
        return existing
    if not existing:
        return note
    return f"{existing};{note}"


def _get_box_x_center(box) -> float | None:
    """Extract horizontal center from a bounding box (polygon or rect tuple)."""
    if box is None:
        return None
    try:
        import numpy as np
        arr = np.asarray(box)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            return float(arr[:, 0].mean())
        if arr.ndim == 1 and len(arr) >= 4:
            return float(arr[0] + arr[2] / 2)
    except Exception:
        pass
    return None


def merge_adjacent_tokens(results: list[tuple[object, str, float]]) -> list[tuple[object, str, float]]:
    """Try merging short digit tokens with longer digit tokens based on spatial order."""
    import numpy as np

    short_tokens = []  # (index, box, text, conf)
    long_tokens = []

    for i, (box, text, conf) in enumerate(results):
        canon = canonicalize(text)
        if not canon:
            continue
        if canon.isdigit() and len(canon) <= 2:
            short_tokens.append((i, box, canon, conf))
        elif canon.isdigit() and len(canon) >= 4:
            long_tokens.append((i, box, canon, conf))

    if not short_tokens or not long_tokens:
        return []

    merged: list[tuple[object, str, float]] = []

    for si, s_box, s_text, s_conf in short_tokens:
        s_x = _get_box_x_center(s_box)
        for li, l_box, l_text, l_conf in long_tokens:
            l_x = _get_box_x_center(l_box)

            # Determine spatial order: try both prepend and append
            combos = []
            if s_x is not None and l_x is not None:
                if s_x < l_x:
                    combos = [s_text + l_text]
                else:
                    combos = [l_text + s_text]
            else:
                combos = [s_text + l_text, l_text + s_text]

            for combo in combos:
                if any(pat.fullmatch(combo) for pat in PLATE_PATTERNS):
                    s_len = len(s_text)
                    l_len = len(l_text)
                    total = s_len + l_len
                    avg_conf = (s_conf * s_len + l_conf * l_len) / total
                    merge_bonus = 0.15 if total in (5, 6) else 0.08
                    merged_conf = min(avg_conf + merge_bonus, 0.999)
                    merged.append((None, combo, merged_conf))

    return merged


def choose_best_candidate(results: list[tuple[object, str, float]], *, skip_merge: bool = False) -> tuple[str, float, list[dict[str, object]], str]:
    if not results:
        return "", 0.0, [], ""

    # Add merged tokens to the candidate pool (skip for retry pools to avoid cross-run merges)
    merged_tokens = [] if skip_merge else merge_adjacent_tokens(results)
    all_results = list(results) + merged_tokens

    candidates: list[dict[str, object]] = []
    for item in all_results:
        text = str(item[1])
        conf = float(item[2])
        canonical = canonicalize(text)
        score = plate_score(text, conf)
        candidates.append(
            {
                "text": text,
                "canonical": canonical,
                "confidence": conf,
                "score": score,
            }
        )

    best = max(candidates, key=lambda c: float(c["score"]))
    note = ""

    if merged_tokens and str(best["text"]) in [str(m[1]) for m in merged_tokens]:
        note = append_note(note, "token_merge")

    if RE_L_DASH4.fullmatch(str(best["canonical"])):
        numeric = [c for c in candidates if RE_DIGIT_5_6.fullmatch(str(c["canonical"]))]
        if numeric:
            best_numeric = max(numeric, key=lambda c: float(c["score"]))
            if float(best_numeric["score"]) >= float(best["score"]) - 0.08:
                note = append_note(note, f"prefer_numeric_over_dash:{best['canonical']}->{best_numeric['canonical']}")
                best = best_numeric

    # Prefer 6-digit superset over 5-digit subset (more complete reading)
    # Only consider direct OCR results, not merge artifacts
    merged_texts = {str(m[1]) for m in merged_tokens} if merged_tokens else set()
    best_canon = str(best["canonical"])
    if RE_DIGIT5.fullmatch(best_canon):
        longer = [
            c for c in candidates
            if RE_DIGIT6.fullmatch(str(c["canonical"]))
            and best_canon in str(c["canonical"])
            and float(c["confidence"]) >= 0.90
            and str(c["text"]) not in merged_texts
        ]
        if longer:
            best_longer = max(longer, key=lambda c: float(c["score"]))
            note = append_note(note, f"prefer_6digit_superset:{best_canon}->{best_longer['canonical']}")
            best = best_longer

    raw = [
        {
            "text": str(c["text"]),
            "canonical": str(c["canonical"]),
            "confidence": float(c["confidence"]),
            "score": round(float(c["score"]), 6),
        }
        for c in candidates
    ]
    return str(best["canonical"]), float(best["confidence"]), raw, note


def parse_csv_list(raw_value: str, allowed: set[str], name: str) -> list[str]:
    values = [v.strip().lower() for v in raw_value.split(",") if v.strip()]
    if not values:
        raise ValueError(f"At least one {name} is required.")
    invalid = [v for v in values if v not in allowed]
    if invalid:
        raise ValueError(f"Invalid {name}: {', '.join(invalid)}. Allowed: {', '.join(sorted(allowed))}")

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def init_easyocr(model_dir: Path, use_gpu: bool):
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("easyocr is not installed. Install with: pip install easyocr") from exc

    easy_dir = model_dir / "easyocr_models"
    easy_dir.mkdir(parents=True, exist_ok=True)
    try:
        reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory=str(easy_dir),
            download_enabled=False,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"easyocr initialization failed with local cache at {easy_dir}. "
            "If models are missing, populate easyocr_models or enable internet access."
        ) from exc
    return reader, "easyocr"


def init_paddleocr(model_dir: Path, use_gpu: bool):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(model_dir / "paddle_models"))
    os.environ.setdefault("PADDLE_HOME", str(model_dir / "paddle_models"))
    os.environ.setdefault("PADDLEOCR_HOME", str(model_dir / "paddle_models"))

    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("paddleocr is not installed. Install with: pip install paddleocr paddlepaddle") from exc

    try:
        import paddle.inference as _paddle_infer
        _paddle_infer.Config.enable_mkldnn = lambda self: None  # noqa: E731
    except Exception:
        pass

    reader = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0" if use_gpu else "cpu",
    )
    return reader, "paddleocr"


def init_tesseract(tesseract_cmd: str):
    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract") from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    resolved_cmd = pytesseract.pytesseract.tesseract_cmd
    cmd_exists = Path(resolved_cmd).exists() or shutil.which(resolved_cmd) is not None
    if not cmd_exists:
        raise RuntimeError("tesseract binary not found. Install it and/or pass --tesseract-cmd /path/to/tesseract")

    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"tesseract binary is not callable: {exc}") from exc

    return pytesseract, f"tesseract {version}"


def extract_easyocr_candidates(reader: object, image_path: Path, allowlist: str) -> list[tuple[object, str, float]]:
    results = reader.readtext(
        str(image_path),
        detail=1,
        paragraph=False,
        decoder="greedy",
        allowlist=allowlist,
        min_size=8,
        canvas_size=1280,
        mag_ratio=1.0,
    )
    return [(item[0], str(item[1]), float(item[2])) for item in results]


def extract_paddle_candidates(raw_result: object) -> list[tuple[object, str, float]]:
    candidates: list[tuple[object, str, float]] = []

    def walk(node: object) -> None:
        if node is None:
            return

        if isinstance(node, dict):
            texts = node.get("rec_texts")
            scores = node.get("rec_scores")
            polys = node.get("dt_polys")
            if isinstance(texts, list):
                score_list = scores if isinstance(scores, list) else []
                poly_list = polys if isinstance(polys, list) else []
                for idx, text in enumerate(texts):
                    if text is None:
                        continue
                    conf = 0.0
                    if idx < len(score_list):
                        try:
                            conf = float(score_list[idx])
                        except (TypeError, ValueError):
                            conf = 0.0
                    box = poly_list[idx] if idx < len(poly_list) else None
                    candidates.append((box, str(text), conf))

            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    walk(value)
            return

        if isinstance(node, (list, tuple)):
            if (
                len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
                and isinstance(node[1][0], str)
            ):
                text = str(node[1][0])
                try:
                    conf = float(node[1][1])
                except (TypeError, ValueError):
                    conf = 0.0
                candidates.append((node[0], text, conf))
                return

            for item in node:
                if isinstance(item, (dict, list, tuple)):
                    walk(item)

    walk(raw_result)
    return candidates


def _to_axis_aligned_box(box: object) -> tuple[float, float, float, float] | None:
    if box is None:
        return None

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return None

    try:
        arr = np.asarray(box, dtype=float)
    except Exception:
        return None

    if arr.size == 0:
        return None

    if arr.ndim == 1 and arr.size >= 4:
        x1 = float(arr[0])
        y1 = float(arr[1])
        w_or_x2 = float(arr[2])
        h_or_y2 = float(arr[3])

        # Tesseract-style rect: left, top, width, height.
        if w_or_x2 > 0 and h_or_y2 > 0:
            return (x1, y1, x1 + w_or_x2, y1 + h_or_y2)
        # Already xyxy.
        return (x1, y1, w_or_x2, h_or_y2)

    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        xs = arr[..., 0].reshape(-1)
        ys = arr[..., 1].reshape(-1)
        if xs.size == 0 or ys.size == 0:
            return None
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    return None


def _boxed_paddle_candidate(results: list[tuple[object, str, float]]) -> tuple[object, str, float] | None:
    boxed = [
        item
        for item in results
        if item[0] is not None and str(item[1]).strip()
    ]
    if not boxed:
        return None
    return max(boxed, key=lambda item: plate_score(str(item[1]), float(item[2])))


def _unsure_char_indices(text: str, confidence: float, threshold: float) -> list[int]:
    unsure: list[int] = []
    low_confidence_line = confidence < threshold

    for idx, char in enumerate(text):
        if char in AMBIGUOUS_OCR_CHARS:
            unsure.append(idx)
            continue
        if not char.isalnum() and char != "-":
            unsure.append(idx)
            continue
        if low_confidence_line and char.isalnum():
            unsure.append(idx)

    return unsure


def _split_box_by_text(
    text: str,
    box_xyxy: tuple[float, float, float, float],
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    if not text:
        return []

    x1, y1, x2, y2 = box_xyxy
    width = max(x2 - x1, 1.0)
    count = len(text)
    char_boxes: list[tuple[int, str, tuple[float, float, float, float]]] = []

    for idx, char in enumerate(text):
        char_x1 = x1 + width * (idx / count)
        char_x2 = x1 + width * ((idx + 1) / count)
        char_boxes.append((idx, char, (char_x1, y1, char_x2, y2)))

    return char_boxes


def save_paddle_unsure_overlay(
    image_path: Path,
    output_dir: Path,
    image_name: str,
    variant: str,
    paddle_candidate: tuple[object, str, float],
    confidence_threshold: float,
) -> tuple[str, int]:
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return "", 0

    box, text, confidence = paddle_candidate
    box_xyxy = _to_axis_aligned_box(box)
    if not box_xyxy:
        return "", 0

    unsure_indices = _unsure_char_indices(str(text), float(confidence), confidence_threshold)
    if not unsure_indices:
        return "", 0

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    char_boxes = _split_box_by_text(str(text), box_xyxy)
    for idx, char, char_box in char_boxes:
        if idx in unsure_indices:
            draw.rectangle(char_box, outline=(255, 0, 0), width=2)
            label_y = max(int(char_box[1]) - 12, 0)
            draw.text((int(char_box[0]) + 1, label_y), char, fill=(255, 0, 0))
        else:
            draw.rectangle(char_box, outline=(0, 180, 255), width=1)

    draw.rectangle(box_xyxy, outline=(255, 255, 0), width=2)

    overlays_dir = output_dir / "paddle_unsure_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlays_dir / f"{Path(image_name).stem}_{variant}_unsure.png"
    image.save(out_path)
    image.close()
    return str(out_path), len(unsure_indices)


def _per_char_confidences(text: str, all_preds: list[tuple[str, float]]) -> list[float]:
    """
    Weighted-vote confidence for each character position in *text*.

    For position n:
        conf[n] = Σ(conf_i  where  pred_i[n] == text[n]  and  len(pred_i) > n)
                  ─────────────────────────────────────────────────────────────
                  Σ(conf_i  for all preds where  len(pred_i) > n)

    When the ensemble unanimously agrees → 1.0.  Disagreement reduces the score
    in proportion to the combined weight of disagreeing predictions.
    """
    confs: list[float] = []
    for n, ch in enumerate(text):
        relevant = [(t, c) for t, c in all_preds if len(t) > n]
        if not relevant:
            confs.append(0.0)
            continue
        total  = sum(c for _, c in relevant)
        agreed = sum(c for t, c in relevant if t[n] == ch)
        confs.append(round(agreed / total, 3) if total > 0 else 0.0)
    return confs


def _conf_to_rgb(conf: float) -> tuple[int, int, int]:
    """Confidence 0→1  maps to  red → yellow → green  in RGB."""
    conf = max(0.0, min(1.0, conf))
    r = int(255 * (1.0 - conf))
    g = int(255 * conf)
    return (r, g, 0)


def save_overlay_image(
    image_path: Path,
    output_dir: Path,
    image_name: str,
    engine: str,
    variant: str,
    results: list[tuple[object, str, float]],
    best_text: str,
    best_confidence: float,
) -> str:
    """
    Draw per-character bounding boxes (coloured by weighted-vote confidence)
    on *image_path* and save to ``<output_dir>/overlays/``.

    Each detected text region is divided into equal-width sub-boxes — one per
    character.  The colour of each sub-box runs red (low agreement) → yellow →
    green (full agreement) based on the weighted vote across all predictions
    from this engine+variant run.  A result banner is added at the bottom.

    Returns the saved file path as a string, or "" on failure.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return ""

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return ""

    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # All (normalised_text, conf) pairs — used for the weighted-vote
    all_preds: list[tuple[str, float]] = [
        (normalize_token(text), conf)
        for _, text, conf in results
        if normalize_token(text)
    ]

    COLOR_BEST  = (  0, 210,   0)
    COLOR_PLATE = (  0, 140, 255)
    COLOR_OTHER = (100, 100, 100)

    for box, raw_text, confidence in results:
        text     = normalize_token(raw_text)
        is_best  = bool(best_text) and text == best_text
        is_plate = bool(text) and any(p.fullmatch(text) for p in PLATE_PATTERNS)

        box_xyxy = _to_axis_aligned_box(box)
        if box_xyxy is None:
            continue

        outer_color = COLOR_BEST if is_best else (COLOR_PLATE if is_plate else COLOR_OTHER)
        thick = 3 if is_best else (2 if is_plate else 1)
        draw.rectangle(box_xyxy, outline=outer_color, width=thick)

        if text:
            char_confs = _per_char_confidences(text, all_preds)
            char_boxes = _split_box_by_text(text, box_xyxy)

            for idx, ch, char_box in char_boxes:
                ch_conf = char_confs[idx] if idx < len(char_confs) else 0.0
                sub_color = _conf_to_rgb(ch_conf)

                # Semi-transparent fill + coloured border
                draw.rectangle(char_box, fill=(*sub_color, 40), outline=sub_color, width=1)

                # Character label + confidence inside the sub-box
                cx = int((char_box[0] + char_box[2]) / 2)
                cy = int((char_box[1] + char_box[3]) / 2)
                char_label = ch
                conf_label = f"{ch_conf:.0%}"

                # White shadow then coloured text
                draw.text((cx - 4, cy - 10), char_label, fill=(0, 0, 0))
                draw.text((cx - 5, cy - 11), char_label, fill=(255, 255, 255))
                draw.text((cx - 5, cy - 11), char_label, fill=sub_color)

                draw.text((cx - 6, cy + 1), conf_label, fill=(0, 0, 0))
                draw.text((cx - 7, cy), conf_label, fill=(200, 200, 200))
        else:
            x1, y1 = int(box_xyxy[0]), int(box_xyxy[1])
            draw.text((x1 + 2, max(y1 - 14, 0)),
                      f"{raw_text}  {confidence:.2f}", fill=outer_color)

    # Result banner at the bottom
    banner_color = (0, 120, 0) if best_text else (60, 60, 60)
    draw.rectangle([(0, H - 32), (W, H)], fill=(*banner_color, 220))
    banner = (f"ID: {best_text}  ({best_confidence:.2%})" if best_text
              else "No plate ID detected")
    draw.text((8, H - 24), banner, fill=(255, 255, 255))

    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    out_path = overlays_dir / f"{stem}_{engine}_{variant}_overlay.png"
    img.save(out_path)
    img.close()
    return str(out_path)


def extract_tesseract_candidates(
    pytesseract_module: object,
    image_path: Path,
    allowlist: str,
    psm: int,
    oem: int,
) -> list[tuple[object, str, float]]:
    candidates: list[tuple[object, str, float]] = []
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for Tesseract OCR.") from exc

    config = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={allowlist}"
    image = Image.open(image_path)
    data = pytesseract_module.image_to_data(
        image,
        output_type=pytesseract_module.Output.DICT,
        config=config,
        lang="eng",
    )
    image.close()

    n = len(data.get("text", []))
    line_groups: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for i in range(n):
        text = str(data["text"][i] or "").strip()
        if not text:
            continue

        try:
            conf_raw = float(data["conf"][i])
        except (TypeError, ValueError):
            conf_raw = -1.0
        if conf_raw < 0:
            continue
        conf = conf_raw / 100.0 if conf_raw > 1.0 else conf_raw

        left = int(float(data["left"][i]))
        top = int(float(data["top"][i]))
        width = int(float(data["width"][i]))
        height = int(float(data["height"][i]))
        box = (left, top, width, height)
        candidates.append((box, text, conf))

        key = (
            str(data.get("block_num", ["0"] * n)[i]),
            str(data.get("par_num", ["0"] * n)[i]),
            str(data.get("line_num", ["0"] * n)[i]),
        )
        line_groups.setdefault(key, []).append((text, conf))

    for tokens in line_groups.values():
        if len(tokens) <= 1:
            continue
        line_text = "".join(part[0] for part in tokens)
        line_conf = sum(part[1] for part in tokens) / len(tokens)
        candidates.append((None, line_text, line_conf))

    return candidates


def build_variant_image(image_path: Path, variant: str):
    try:
        from PIL import Image, ImageEnhance, ImageOps, ImageStat
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image preprocessing.") from exc

    image = Image.open(image_path).convert("RGB")
    if variant == "raw":
        return image

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(1.7)

    mean_val = ImageStat.Stat(gray).mean[0]
    threshold = int(max(95, min(180, mean_val * 0.95)))
    bw = gray.point(lambda p: 255 if p > threshold else 0, mode="L")
    image.close()
    return bw


def pad_image(image_path: Path, pad_fraction: float = 0.08) -> "Image":
    """Add white padding around image to help OCR detect text near edges."""
    from PIL import Image, ImageOps

    img = Image.open(image_path).convert("RGB")
    pad = max(int(max(img.size) * pad_fraction), 10)
    return ImageOps.expand(img, border=pad, fill=(255, 255, 255))


def scale_image_for_paddle(image_path: Path, scale: float) -> "Image":
    """Resize image by scale factor for faster PaddleOCR passes."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if scale >= 0.999:
        return img

    new_w = max(32, int(img.width * scale))
    new_h = max(32, int(img.height * scale))
    if (new_w, new_h) == img.size:
        return img

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    img.close()
    return resized


def enhance_for_ocr_retry(image_path: Path) -> "Image":
    """Upscale + autocontrast + sharpen + pad for OCR retry on weak results."""
    from PIL import Image, ImageEnhance, ImageOps

    img = Image.open(image_path).convert("RGB")
    new_size = (int(img.width * 1.5), int(img.height * 1.5))
    img = img.resize(new_size, Image.LANCZOS)
    pad = max(int(max(img.size) * 0.08), 15)
    img = ImageOps.expand(img, border=pad, fill=(255, 255, 255))
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    return img


def enhance_clahe(image_path: Path) -> "Image":
    """CLAHE contrast enhancement + 2x upscale + bilateral filter + padding."""
    import cv2
    import numpy as np
    from PIL import Image

    img = cv2.imread(str(image_path))
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge([l_chan, a_chan, b_chan])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    pad = max(int(max(img.shape[:2]) * 0.08), 15)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def enhance_heavy_upscale(image_path: Path) -> "Image":
    """3x LANCZOS upscale + aggressive autocontrast + contrast 2.0 + sharpness 2.0 + padding."""
    from PIL import Image, ImageEnhance, ImageOps

    img = Image.open(image_path).convert("RGB")
    new_size = (int(img.width * 3), int(img.height * 3))
    img = img.resize(new_size, Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=3)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    pad = max(int(max(img.size) * 0.08), 15)
    img = ImageOps.expand(img, border=pad, fill=(255, 255, 255))
    return img


def crop_and_clahe(image_path: Path, detection_boxes: list) -> "Image | None":
    """Crop the best detection region, apply CLAHE at 4x, for character-level re-OCR."""
    import cv2
    import numpy as np
    from PIL import Image

    if not detection_boxes:
        return None

    orig = Image.open(image_path).convert("RGB")
    # Use the first valid box (typically the best detection)
    for box in detection_boxes:
        if box is None:
            continue
        arr = np.asarray(box)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue

        # Account for padding that was added by pad_image
        pad_amount = max(int(max(orig.size) * 0.08), 10)
        xs = arr[:, 0] - pad_amount
        ys = arr[:, 1] - pad_amount

        margin_x = int((xs.max() - xs.min()) * 0.15)
        margin_y = int((ys.max() - ys.min()) * 0.3)
        x1 = max(0, int(xs.min()) - margin_x)
        y1 = max(0, int(ys.min()) - margin_y)
        x2 = min(orig.width, int(xs.max()) + margin_x)
        y2 = min(orig.height, int(ys.max()) + margin_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        crop = orig.crop((x1, y1, x2, y2))
        crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        crop_cv = cv2.resize(crop_cv, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge([l_chan, a_chan, b_chan])
        crop_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        crop_cv = cv2.bilateralFilter(crop_cv, 9, 75, 75)
        pad = max(int(max(crop_cv.shape[:2]) * 0.1), 20)
        crop_cv = cv2.copyMakeBorder(crop_cv, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
        orig.close()
        return Image.fromarray(rgb)

    orig.close()
    return None


def extract_poletag_index(filename: str) -> int | None:
    m = POLETAG_RE.match(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def apply_neighbor_consensus(rows: list[dict[str, object]]) -> int:
    by_index: dict[int, dict[str, object]] = {}
    for row in rows:
        idx = extract_poletag_index(str(row["image"]))
        if idx is not None:
            by_index[idx] = row

    fixes = 0
    for idx, row in by_index.items():
        pred = str(row.get("best_text", ""))
        if not (pred.isdigit() and 4 <= len(pred) <= 5):
            continue

        neighbor_vals: list[str] = []
        for delta in (-2, -1, 1, 2):
            other = by_index.get(idx + delta)
            if not other:
                continue
            other_pred = str(other.get("best_text", ""))
            if other_pred.isdigit() and len(other_pred) == 6:
                neighbor_vals.append(other_pred)

        if len(neighbor_vals) < 2:
            continue

        consensus, count = Counter(neighbor_vals).most_common(1)[0]
        if count >= 2 and consensus.endswith(pred):
            row["best_text"] = consensus
            row["best_confidence"] = float(row.get("best_confidence", 0.0)) * 0.92
            row["postprocess_note"] = append_note(
                str(row.get("postprocess_note", "")),
                f"neighbor_consensus:{pred}->{consensus}",
            )
            fixes += 1

    return fixes


def load_truth(labels_csv: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    if not labels_csv.exists():
        return truth
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = canonicalize((row.get("plate_label") or "").strip())
            if filename.startswith("PoleTag_") and label:
                truth[filename] = label
    return truth


def run_engine_variant(
    engine: str,
    reader: object,
    variant: str,
    image_paths: list[Path],
    truth: dict[str, str],
    output_dir: Path,
    allowlist: str,
    psm: int,
    oem: int,
    save_paddle_unsure_overlays: bool,
    paddle_unsure_threshold: float,
    paddle_max_reruns: int,
    paddle_time_cap_seconds: float,
    paddle_scale: float,
    save_overlays: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix=f"ocr2_{engine}_{variant}_", dir=str(output_dir)) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for image_path in image_paths:
            image_start_time = time.perf_counter()
            paddle_source_path = image_path

            if engine == "paddleocr" and paddle_scale < 0.999:
                scaled = scale_image_for_paddle(image_path, paddle_scale)
                scale_tag = str(round(paddle_scale, 3)).replace(".", "_")
                scaled_path = tmp_dir / f"{image_path.stem}_scaled_{scale_tag}.png"
                scaled.save(scaled_path)
                scaled.close()
                paddle_source_path = scaled_path

            variant_source = paddle_source_path if engine == "paddleocr" else image_path
            variant_input = variant_source
            if variant != "raw":
                prepared = build_variant_image(variant_source, variant)
                variant_input = tmp_dir / f"{image_path.stem}_{variant}.png"
                prepared.save(variant_input)
                prepared.close()
            elif engine == "paddleocr":
                padded = pad_image(variant_source)
                variant_input = tmp_dir / f"{image_path.stem}_padded.png"
                padded.save(variant_input)
                padded.close()

            try:
                if engine == "easyocr":
                    results = extract_easyocr_candidates(reader, variant_input, allowlist)
                elif engine == "paddleocr":
                    raw_result = reader.predict(str(variant_input))
                    results = extract_paddle_candidates(raw_result)
                else:
                    results = extract_tesseract_candidates(reader, variant_input, allowlist, psm, oem)
            except RuntimeError as exc:
                print(f"{image_path.name}: OCR error: {exc}")
                results = []

            best_text, best_confidence, raw, choose_note = choose_best_candidate(results)
            unsure_overlay = ""
            unsure_char_count = 0

            if engine == "paddleocr" and save_paddle_unsure_overlays:
                boxed_candidate = _boxed_paddle_candidate(results)
                if boxed_candidate is not None:
                    unsure_overlay, unsure_char_count = save_paddle_unsure_overlay(
                        variant_input,
                        output_dir,
                        image_path.name,
                        variant,
                        boxed_candidate,
                        paddle_unsure_threshold,
                    )

            # Multi-strategy retry for PaddleOCR when initial result is weak or uncertain
            if engine == "paddleocr" and variant == "raw":
                initial_score = plate_score(best_text, best_confidence) if best_text else -1.0
                is_merged = "token_merge" in choose_note
                if (initial_score < 1.35 or not best_text or is_merged) and paddle_max_reruns > 0:
                    all_retry_candidates = list(results)
                    reruns_used = 0
                    cap_hit = False
                    time_cap_hit = False

                    scored_boxes = [
                        (plate_score(str(item[1]), float(item[2])), item[0])
                        for item in results if item[0] is not None
                    ]
                    scored_boxes.sort(key=lambda x: x[0], reverse=True)
                    detection_boxes = [box for _, box in scored_boxes]

                    retry_strategies = [
                        ("crop_clahe", lambda p: crop_and_clahe(p, detection_boxes)),
                        ("clahe", enhance_clahe),
                        ("upscale", enhance_for_ocr_retry),
                        ("heavy", enhance_heavy_upscale),
                        ("binarize", lambda p: build_variant_image(p, "text_light")),
                    ]

                    for strat_name, strat_fn in retry_strategies:
                        if reruns_used >= paddle_max_reruns:
                            cap_hit = True
                            break
                        if paddle_time_cap_seconds > 0:
                            elapsed = time.perf_counter() - image_start_time
                            if elapsed >= paddle_time_cap_seconds:
                                time_cap_hit = True
                                break
                        try:
                            enhanced = strat_fn(variant_source)
                            if enhanced is None:
                                continue
                            retry_path = tmp_dir / f"{image_path.stem}_retry_{strat_name}.png"
                            enhanced.save(retry_path)
                            enhanced.close()
                            retry_result = reader.predict(str(retry_path))
                            retry_candidates = extract_paddle_candidates(retry_result)
                            all_retry_candidates.extend(retry_candidates)
                            reruns_used += 1
                        except Exception:
                            pass
                    best2, conf2, raw2, note2 = choose_best_candidate(all_retry_candidates, skip_merge=True)
                    score2 = plate_score(best2, conf2) if best2 else -1.0
                    # Accept retry if better score, or if it found a longer digit superset
                    retry_is_superset = (
                        best2.isdigit() and best_text.isdigit()
                        and len(best2) > len(best_text)
                        and best_text in best2
                        and score2 > initial_score - 0.15
                    )
                    if score2 > initial_score or retry_is_superset:
                        best_text, best_confidence, raw, choose_note = (
                            best2, conf2, raw2, append_note(note2, "multi_retry"),
                        )
                    choose_note = append_note(choose_note, f"reruns_used:{reruns_used}")
                    if cap_hit:
                        choose_note = append_note(choose_note, f"rerun_cap:{paddle_max_reruns}")
                    if time_cap_hit:
                        choose_note = append_note(choose_note, f"time_cap:{paddle_time_cap_seconds:.1f}s")
            if engine == "paddleocr" and paddle_scale < 0.999:
                choose_note = append_note(choose_note, f"scale:{paddle_scale:.3f}")

            if save_overlays and results:
                save_overlay_image(
                    image_path,
                    output_dir,
                    image_path.name,
                    engine,
                    variant,
                    results,
                    best_text,
                    best_confidence,
                )

            rows.append(
                {
                    "image": image_path.name,
                    "engine": engine,
                    "variant": variant,
                    "best_text": best_text,
                    "best_confidence": round(best_confidence, 6),
                    "truth": truth.get(image_path.name, ""),
                    "exact_match": 0,
                    "postprocess_note": choose_note,
                    "unsure_overlay": unsure_overlay,
                    "unsure_char_count": unsure_char_count,
                    "raw_results": raw,
                }
            )

    fixes = apply_neighbor_consensus(rows)

    exact_matches = 0
    comparable = 0
    for row in rows:
        truth_label = str(row.get("truth", ""))
        is_exact = int(bool(truth_label) and str(row.get("best_text", "")) == truth_label)
        row["exact_match"] = is_exact
        if truth_label:
            comparable += 1
            exact_matches += is_exact
            note = str(row.get("postprocess_note", ""))
            note_text = f" [{note}]" if note else ""
            print(
                f"{row['image']}: {row['best_text'] or 'NO_TEXT'} "
                f"({float(row['best_confidence']):.4f}) [truth={truth_label}]{note_text}"
            )
        else:
            print(f"{row['image']}: {row['best_text'] or 'NO_TEXT'} ({float(row['best_confidence']):.4f})")

    metrics = {
        "engine": engine,
        "variant": variant,
        "comparable": comparable,
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / comparable, 6) if comparable else None,
        "neighbor_consensus_fixes": fixes,
    }
    return rows, metrics


def write_variant_outputs(output_dir: Path, rows: list[dict[str, object]], metrics: dict[str, object]) -> tuple[Path, Path]:
    suffix = f"{metrics['engine']}_{metrics['variant']}"
    csv_path = output_dir / f"plate_predictions2_{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "engine",
                "variant",
                "best_text",
                "best_confidence",
                "truth",
                "exact_match",
                "postprocess_note",
                "unsure_overlay",
                "unsure_char_count",
                "raw_results",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "engine": row["engine"],
                    "variant": row["variant"],
                    "best_text": row["best_text"],
                    "best_confidence": row["best_confidence"],
                    "truth": row["truth"],
                    "exact_match": row["exact_match"],
                    "postprocess_note": row["postprocess_note"],
                    "unsure_overlay": row.get("unsure_overlay", ""),
                    "unsure_char_count": row.get("unsure_char_count", 0),
                    "raw_results": json.dumps(row["raw_results"]),
                }
            )

    json_path = output_dir / f"plate_predictions2_{suffix}.json"
    json_path.write_text(json.dumps({"results": rows, "metrics": metrics}, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        engines = parse_csv_list(args.engines, SUPPORTED_ENGINES, "engines")
        variants = parse_csv_list(args.variants, SUPPORTED_VARIANTS, "variants")
    except ValueError as exc:
        print(exc)
        return 2
    if args.paddle_max_reruns < 0:
        print("--paddle-max-reruns must be >= 0")
        return 2
    if args.paddle_time_cap_seconds < 0:
        print("--paddle-time-cap-seconds must be >= 0")
        return 2
    if args.paddle_scale <= 0:
        print("--paddle-scale must be > 0")
        return 2

    image_paths = iter_pole_images(images_dir)
    if not image_paths:
        print(f"No PoleTag images found in {images_dir}")
        return 1

    truth = load_truth(args.labels_csv.resolve())

    initialized: dict[str, object] = {}
    for engine in engines:
        try:
            if engine == "easyocr":
                reader, label = init_easyocr(model_dir, args.gpu)
            elif engine == "paddleocr":
                reader, label = init_paddleocr(model_dir, args.gpu)
            else:
                reader, label = init_tesseract(args.tesseract_cmd)
            initialized[engine] = reader
            print(f"Initialized {label}")
        except RuntimeError as exc:
            print(f"Skipping {engine}: {exc}")

    if not initialized:
        print("No OCR engines initialized. Install dependencies and try again.")
        return 2

    summaries: list[dict[str, object]] = []
    for engine in engines:
        if engine not in initialized:
            continue
        for variant in variants:
            print(f"\n=== Engine: {engine} | Variant: {variant} ===")
            rows, metrics = run_engine_variant(
                engine,
                initialized[engine],
                variant,
                image_paths,
                truth,
                output_dir,
                args.allowlist,
                args.psm,
                args.oem,
                args.save_paddle_unsure_overlays,
                args.paddle_unsure_threshold,
                args.paddle_max_reruns,
                args.paddle_time_cap_seconds,
                args.paddle_scale,
                save_overlays=args.save_overlays,
            )
            csv_path, json_path = write_variant_outputs(output_dir, rows, metrics)
            summaries.append(metrics)

            print(f"Saved CSV:  {csv_path}")
            print(f"Saved JSON: {json_path}")
            if metrics["comparable"]:
                print(
                    f"Exact match on PoleTag truth rows: "
                    f"{metrics['exact_matches']}/{metrics['comparable']} "
                    f"({float(metrics['exact_match_rate']):.2%})"
                )

    summary_path = output_dir / "plate_predictions2_summary.json"
    summary_path.write_text(json.dumps({"variants": summaries}, indent=2), encoding="utf-8")
    print(f"\nSaved Summary: {summary_path}")
    print("Run Accuracy:")
    for metric in summaries:
        if metric["comparable"]:
            print(
                f"- {metric['engine']}/{metric['variant']}: "
                f"{metric['exact_matches']}/{metric['comparable']} "
                f"({float(metric['exact_match_rate']):.2%})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
