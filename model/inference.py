"""
Inference module for vegetation encroachment detection and pole plate OCR.

Call load_models() once before using run_ocr() or run_encroachment().
All public functions accept raw PNG/JPEG bytes from the camera and return
plain Python dicts/strings suitable for JSON serialization.
"""

from __future__ import annotations

import base64
import os
import re
import traceback
from pathlib import Path
from typing import Any

import certifi
import cv2
import numpy as np

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent

DETECT_MODEL_PATH = _MODEL_DIR / "trained_weights" / "yolo26m-detect" / "weights" / "soup.pt"
SEG_MODEL_PATH    = _MODEL_DIR / "trained_weights" / "yolo26m-seg-veg-guy" / "weights" / "soup.pt"
DETECT_FALLBACK   = "yolo26m.pt"
SEG_FALLBACK      = "yolo26m-seg.pt"
EASYOCR_MODEL_DIR = _MODEL_DIR / "easyocr_models"

# ---------------------------------------------------------------------------
# Inference constants
# ---------------------------------------------------------------------------
CONF                   = 0.25
ENCROACHMENT_THRESHOLD = 0.05

POLE_NAMES         = {"pole", "composite", "wood"}
DET_IGNORE_CLASSES = {"vegetation", "guy_guard"}

COMPONENT_MAP: dict[str, str] = {
    "transformer":  "transformer",
    "insulator":    "insulator",
    "street light": "streetlight",
}

DET_COLORS: dict[str, tuple[int, int, int]] = {
    "composite":    (  0, 200, 255),
    "wood":         ( 43, 140, 200),
    "guy_guard":    (255, 215,   0),
    "insulator":    (255, 100,   0),
    "street light": (200,   0, 200),
    "transformer":  (  0, 165, 255),
    "vegetation":   ( 50, 255,  50),
}
SEG_FILL: dict[str, tuple[int, int, int]] = {
    "vegetation": (  0, 120,   0),
    "guy_guard":  (  0, 160, 200),
}
ENCROACHMENT_COLOR = (0, 0, 220)

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
PLATE_LIKE_PATTERNS = (
    re.compile(r"^[A-Z]{1,4}\d{2,7}[A-Z]?$"),
    re.compile(r"^\d{1,4}-\d{2,6}$"),
    re.compile(r"^\d{4,8}$"),
)

# ---------------------------------------------------------------------------
# Model singletons — populated by load_models()
# ---------------------------------------------------------------------------
_detect_model = None
_seg_model    = None
_ocr_reader   = None

DETECT_CLASSES: dict[int, str] = {}
SEG_CLASSES:    dict[int, str] = {}
POLE_CLASS_IDS: set[int]       = set()


def load_models() -> None:
    """
    Load YOLO detection, YOLO segmentation, and EasyOCR models.
    Must be called once before run_ocr() or run_encroachment().
    Raises RuntimeError (with full traceback printed) if any model fails to load.
    """
    global _detect_model, _seg_model, _ocr_reader
    global DETECT_CLASSES, SEG_CLASSES, POLE_CLASS_IDS

    try:
        print("[inference] Importing ultralytics...")
        from ultralytics import YOLO  # noqa: PLC0415
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Failed to import ultralytics — check your environment.")

    try:
        print("[inference] Loading detection model...")
        p = DETECT_MODEL_PATH
        if p.exists():
            print(f"  -> fine-tuned weights: {p}")
            _detect_model = YOLO(str(p))
        else:
            print(f"  -> {p} not found, using base: {DETECT_FALLBACK}")
            _detect_model = YOLO(DETECT_FALLBACK)
        DETECT_CLASSES = _detect_model.names
        POLE_CLASS_IDS = {cid for cid, name in DETECT_CLASSES.items() if name.lower() in POLE_NAMES}
        print(f"  -> classes: {DETECT_CLASSES}")
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Failed to load detection model.")

    try:
        print("[inference] Loading segmentation model...")
        p = SEG_MODEL_PATH
        if p.exists():
            print(f"  -> fine-tuned weights: {p}")
            _seg_model = YOLO(str(p))
        else:
            print(f"  -> {p} not found, using base: {SEG_FALLBACK}")
            _seg_model = YOLO(SEG_FALLBACK)
        SEG_CLASSES = _seg_model.names
        print(f"  -> classes: {SEG_CLASSES}")
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Failed to load segmentation model.")

    try:
        print("[inference] Loading EasyOCR reader...")
        import easyocr  # noqa: PLC0415
        EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["EASYOCR_MODULE_PATH"] = str(EASYOCR_MODEL_DIR)
        _ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=str(EASYOCR_MODEL_DIR),
            download_enabled=True,
            verbose=False,
        )
        print("  -> EasyOCR ready.")
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Failed to load EasyOCR reader.")

    print("[inference] All models loaded successfully.")


def _assert_loaded() -> None:
    if _detect_model is None or _seg_model is None or _ocr_reader is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes — ensure PNG/JPEG data is valid.")
    return img


def _parse_detections(results: Any, class_map: dict[int, str]) -> list[dict]:
    items: list[dict] = []
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return items
    for i, (xyxy, cls, conf) in enumerate(zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy(),
    )):
        x1, y1, x2, y2 = xyxy.tolist()
        items.append({
            "id":         i,
            "class":      class_map.get(int(cls), str(cls)),
            "class_id":   int(cls),
            "confidence": round(float(conf), 4),
            "bbox":       {"x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)},
        })
    return items


def _parse_segments(results: Any, H: int, W: int, class_map: dict[int, str]) -> list[dict]:
    segs: list[dict] = []
    boxes = results.boxes
    masks = results.masks
    if boxes is None or masks is None:
        return segs
    for i, (xyxy, cls, conf, xy) in enumerate(zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy(),
        masks.xy,
    )):
        x1, y1, x2, y2 = xyxy.tolist()
        poly = xy.astype(np.int32)
        binary = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(binary, [poly], 1)
        segs.append({
            "id":          i,
            "class":       class_map.get(int(cls), str(cls)),
            "class_id":    int(cls),
            "confidence":  round(float(conf), 4),
            "bbox":        {"x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)},
            "polygon":     [[int(pt[0]), int(pt[1])] for pt in poly.tolist()],
            "binary_mask": binary,
        })
    return segs


def _compute_encroachment(poles: list[dict], segments: list[dict]) -> None:
    for pole in poles:
        b  = pole["bbox"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
        bbox_area = max((bx2 - bx1) * (by2 - by1), 1)
        pole["encroachment_detected"]  = False
        pole["encroaching_segments"]   = []
        for seg in segments:
            mask_crop     = seg["binary_mask"][by1:by2, bx1:bx2]
            overlap_px    = int(mask_crop.sum())
            overlap_ratio = round(overlap_px / bbox_area, 4)
            if overlap_ratio >= ENCROACHMENT_THRESHOLD:
                pole["encroachment_detected"] = True
                pole["encroaching_segments"].append({
                    "segment_id":     seg["id"],
                    "class":          seg["class"],
                    "confidence":     seg["confidence"],
                    "overlap_pixels": overlap_px,
                    "overlap_ratio":  overlap_ratio,
                })


def _encroachment_severity(poles: list[dict]) -> int:
    """0 = None, 1 = Low (<20%), 2 = Medium (20-50%), 3 = High (>50%)."""
    max_ratio = 0.0
    for pole in poles:
        for seg in pole.get("encroaching_segments", []):
            max_ratio = max(max_ratio, seg["overlap_ratio"])
    if max_ratio == 0:
        return 0
    if max_ratio < 0.20:
        return 1
    if max_ratio < 0.50:
        return 2
    return 3


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + baseline + 4)
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def _annotate_combined(
    img_bgr: np.ndarray,
    all_detections: list[dict],
    segments: list[dict],
    poles: list[dict],
) -> np.ndarray:
    combined = img_bgr.copy()
    overlay  = img_bgr.copy()

    for seg in segments:
        fill = SEG_FILL.get(seg["class"], (100, 100, 100))
        cv2.fillPoly(overlay, [np.array(seg["polygon"], dtype=np.int32)], fill)
    cv2.addWeighted(overlay, 0.4, combined, 0.6, 0, combined)

    for seg in segments:
        fill = SEG_FILL.get(seg["class"], (100, 100, 100))
        poly = np.array(seg["polygon"], dtype=np.int32)
        cv2.polylines(combined, [poly], True, (255, 255, 255), 3)
        cv2.polylines(combined, [poly], True, fill, 2)
        _draw_label(combined,
                    f"[seg] {seg['class']} {seg['confidence']:.2f}",
                    seg["bbox"]["x1"], seg["bbox"]["y1"],
                    fg=(255, 255, 255), bg=fill)

    for det in all_detections:
        b     = det["bbox"]
        color = DET_COLORS.get(det["class"], (200, 200, 200))
        thick = 3 if det["class_id"] in POLE_CLASS_IDS else 2
        cv2.rectangle(combined, (b["x1"] - 1, b["y1"] - 1), (b["x2"] + 1, b["y2"] + 1),
                      (255, 255, 255), thick + 1)
        cv2.rectangle(combined, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thick)
        _draw_label(combined,
                    f"[det] {det['class']} {det['confidence']:.2f}",
                    b["x1"], b["y1"], fg=(255, 255, 255), bg=color)

    for pole in poles:
        if pole["encroachment_detected"]:
            b = pole["bbox"]
            cv2.rectangle(combined,
                          (b["x1"] - 2, b["y1"] - 2), (b["x2"] + 2, b["y2"] + 2),
                          ENCROACHMENT_COLOR, 4)
            _draw_label(combined, "ENCROACHMENT",
                        b["x1"], b["y2"] + 30, fg=(255, 255, 255), bg=ENCROACHMENT_COLOR)

    return combined


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _normalize_ocr_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = "".join(ch for ch in text if ch.isalnum() or ch == "-")
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def _is_plate_like(text: str) -> bool:
    if len(text) < 4 or len(text) > 12:
        return False
    if not any(ch.isdigit() for ch in text):
        return False
    alnum_ratio = sum(ch.isalnum() for ch in text) / max(1, len(text))
    return alnum_ratio >= 0.75


def _score_candidate(text: str, confidence: float) -> float:
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


def _ocr_variants(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return {"orig_color": image_bgr, "clahe": clahe}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ocr(png_bytes: bytes) -> str:
    """
    Run plate OCR on the given image bytes.
    Returns the best matching plate ID string, or empty string if not found.
    """
    _assert_loaded()
    img_bgr  = _bytes_to_bgr(png_bytes)
    variants = _ocr_variants(img_bgr)

    ocr_kwargs = {
        "detail":          1,
        "paragraph":       False,
        "allowlist":       OCR_ALLOWLIST,
        "min_size":        12,
        "text_threshold":  0.65,
        "low_text":        0.35,
        "link_threshold":  0.35,
        "canvas_size":     1280,
        "mag_ratio":       1.0,
        "decoder":         "greedy",
    }

    best_by_text: dict[str, tuple[float, float]] = {}
    for variant_img in variants.values():
        results = _ocr_reader.readtext(variant_img, **ocr_kwargs)
        for _bbox, raw_text, confidence in results:
            text = _normalize_ocr_text(raw_text)
            if not text or not _is_plate_like(text):
                continue
            score    = _score_candidate(text, float(confidence))
            existing = best_by_text.get(text)
            if existing is None or float(confidence) > existing[0]:
                best_by_text[text] = (float(confidence), score)

    if not best_by_text:
        return ""

    best_text = max(best_by_text, key=lambda t: (best_by_text[t][0], best_by_text[t][1]))
    _conf, score = best_by_text[best_text]
    return best_text if score >= 0.40 else ""


def run_encroachment(png_bytes: bytes) -> dict:
    """
    Run YOLO detection + segmentation on the given image bytes.

    Returns a dict with:
      pole_type, detected_components, vegetation_severity (0-3),
      encroachment (bool), poles, all_detections, segments,
      annotated_image_b64 (base64 JPEG string).
    """
    _assert_loaded()
    img_bgr = _bytes_to_bgr(png_bytes)
    H, W    = img_bgr.shape[:2]

    det_res = _detect_model(img_bgr, conf=CONF, verbose=False)[0]
    seg_res = _seg_model(img_bgr,    conf=CONF, verbose=False)[0]

    all_detections = [
        d for d in _parse_detections(det_res, DETECT_CLASSES)
        if d["class"] not in DET_IGNORE_CLASSES
    ]
    poles    = [d for d in all_detections if d["class_id"] in POLE_CLASS_IDS]
    segments = _parse_segments(seg_res, H, W, SEG_CLASSES)

    _compute_encroachment(poles, segments)

    encroachment_detected = any(p["encroachment_detected"] for p in poles)
    severity              = _encroachment_severity(poles)

    pole_type = ""
    if poles:
        best_pole = max(poles, key=lambda p: p["confidence"])
        pole_type = best_pole["class"]

    detected_classes    = {d["class"] for d in all_detections}
    detected_components = [
        form_key
        for model_class, form_key in COMPONENT_MAP.items()
        if model_class in detected_classes
    ]

    annotated     = _annotate_combined(img_bgr, all_detections, segments, poles)
    _, jpeg_buf   = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    segs_out = [{k: v for k, v in s.items() if k != "binary_mask"} for s in segments]

    return {
        "pole_type":           pole_type,
        "detected_components": detected_components,
        "vegetation_severity": severity,
        "encroachment":        encroachment_detected,
        "poles":               poles,
        "all_detections":      all_detections,
        "segments":            segs_out,
        "annotated_image_b64": annotated_b64,
    }
