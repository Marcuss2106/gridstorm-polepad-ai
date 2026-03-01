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
import tempfile
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
PADDLE_MODEL_DIR = _MODEL_DIR / "paddle_models"

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
    Load YOLO detection, YOLO segmentation, and PaddleOCR models.
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
        print("[inference] Loading PaddleOCR reader...")
        PADDLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_MODEL_DIR))
        os.environ.setdefault("PADDLE_HOME", str(PADDLE_MODEL_DIR))
        os.environ.setdefault("PADDLEOCR_HOME", str(PADDLE_MODEL_DIR))

        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(
                "paddleocr is not installed. Install with: pip install paddleocr paddlepaddle"
            ) from exc

        try:
            import paddle.inference as _paddle_infer  # noqa: PLC0415
            _paddle_infer.Config.enable_mkldnn = lambda self: None  # noqa: E731
        except Exception:
            pass

        _ocr_reader = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
        print("  -> PaddleOCR ready.")
    except Exception:
        traceback.print_exc()
        raise RuntimeError("Failed to load PaddleOCR reader.")

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

def _extract_paddle_candidates(raw_result: object) -> list[tuple]:
    """
    Walk the nested structure returned by PaddleOCR ``predict()`` and collect
    all ``(bbox, text, confidence)`` triples.

    PaddleOCR (PP-OCRv5 / paddlex backend) returns dicts with keys
    ``rec_texts``, ``rec_scores``, and ``dt_polys``.  Older API versions return
    a nested list of ``[[bbox], [text, conf]]``.  Both shapes are handled.
    """
    candidates: list[tuple] = []

    def _walk(node: object) -> None:
        if node is None:
            return

        if isinstance(node, dict):
            texts  = node.get("rec_texts")
            scores = node.get("rec_scores")
            polys  = node.get("dt_polys")
            if isinstance(texts, list):
                score_list = scores if isinstance(scores, list) else []
                poly_list  = polys  if isinstance(polys,  list) else []
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
                    _walk(value)
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
                    _walk(item)

    _walk(raw_result)
    return candidates


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


def _conf_to_bgr(conf: float) -> tuple[int, int, int]:
    """Map confidence 0→1 to BGR: red → yellow → green."""
    conf = max(0.0, min(1.0, conf))
    r = int(255 * (1.0 - conf))
    g = int(255 * conf)
    return (0, g, r)


def _split_bbox_horizontal(raw_bbox: list, n: int) -> list[list[tuple[int, int]]]:
    """
    Divide a 4-point bbox into *n* equal-width sub-polygons by interpolating
    along the top edge (pts[0]→pts[1]) and bottom edge (pts[3]→pts[2]).
    Returns a list of n sub-boxes, each as [tl, tr, br, bl] integer tuples.
    """
    pts = [tuple(map(float, p)) for p in raw_bbox]
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
    sub_boxes = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        s_tl = (tl[0] + t0 * (tr[0] - tl[0]), tl[1] + t0 * (tr[1] - tl[1]))
        s_tr = (tl[0] + t1 * (tr[0] - tl[0]), tl[1] + t1 * (tr[1] - tl[1]))
        s_br = (bl[0] + t1 * (br[0] - bl[0]), bl[1] + t1 * (br[1] - bl[1]))
        s_bl = (bl[0] + t0 * (br[0] - bl[0]), bl[1] + t0 * (br[1] - bl[1]))
        sub_boxes.append([
            (int(s_tl[0]), int(s_tl[1])),
            (int(s_tr[0]), int(s_tr[1])),
            (int(s_br[0]), int(s_br[1])),
            (int(s_bl[0]), int(s_bl[1])),
        ])
    return sub_boxes


def _per_char_confidences(text: str, all_preds: list[tuple[str, float]]) -> list[float]:
    """
    Compute a per-character confidence for *text* using all predictions from
    every OCR variant as a weighted vote.

    For character position n the confidence is:
        sum(conf_i  for preds where pred_text[n] == text[n]  and len > n)
        ─────────────────────────────────────────────────────────────────
        sum(conf_i  for all preds where len(pred_text) > n)

    A prediction that agrees with *text* at position n contributes its full
    weight; one that disagrees contributes nothing.  The ratio is the
    weighted fraction of agreement — i.e. how certain the ensemble is about
    that specific character.
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


def _draw_ocr_overlay(
    img_bgr: np.ndarray,
    raw_results: list[tuple],
    best_text: str,
    best_conf: float,
    all_preds: list[tuple[str, float]],
) -> np.ndarray:
    """
    Draw OCR detection bounding boxes on a copy of the plate image.

    Each bounding box is split into equal-width sub-boxes — one per character.
    Each sub-box is coloured by per-character confidence (green = high,
    red = low) and labelled with the character and its confidence score.

    - Thick outer box: best plate-ID match (green) / other plate-like (orange)
                       / non-plate text (grey)
    - Green banner at the bottom: final read result
    """
    out = img_bgr.copy()
    H, W = out.shape[:2]

    COLOR_BEST  = (  0, 210,   0)
    COLOR_PLATE = (  0, 165, 255)
    COLOR_OTHER = (100, 100, 100)

    for raw_bbox, raw_text, confidence in raw_results:
        if raw_bbox is None:
            continue
        text     = _normalize_ocr_text(raw_text)
        is_best  = bool(best_text) and text == best_text
        is_plate = bool(text) and _is_plate_like(text)

        # ── outer bounding box ──────────────────────────────────────────────
        outer_pts = np.array(raw_bbox, dtype=np.int32).reshape((-1, 1, 2))
        if is_best:
            outer_color, thick = COLOR_BEST, 3
        elif is_plate:
            outer_color, thick = COLOR_PLATE, 2
        else:
            outer_color, thick = COLOR_OTHER, 1

        cv2.polylines(out, [outer_pts], True, (255, 255, 255), thick + 1)
        cv2.polylines(out, [outer_pts], True, outer_color, thick)

        # ── per-character sub-boxes ─────────────────────────────────────────
        if text:
            n_chars  = len(text)
            char_confs = _per_char_confidences(text, all_preds)
            sub_boxes  = _split_bbox_horizontal(raw_bbox, n_chars)

            for idx, (ch, ch_conf, sub) in enumerate(
                zip(text, char_confs, sub_boxes)
            ):
                sub_color = _conf_to_bgr(ch_conf)
                sub_pts   = np.array(sub, dtype=np.int32).reshape((-1, 1, 2))

                # thin white outline then coloured inner line
                cv2.polylines(out, [sub_pts], True, (255, 255, 255), 2)
                cv2.polylines(out, [sub_pts], True, sub_color, 1)

                # character label centred inside sub-box
                cx = int((sub[0][0] + sub[1][0]) / 2)
                cy = int((sub[0][1] + sub[3][1]) / 2)
                label = f"{ch}\n{ch_conf:.0%}"
                # draw two lines: character on top, confidence below
                (cw, ch_h), base = cv2.getTextSize(
                    ch, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                cv2.putText(out, ch,
                            (cx - cw // 2, cy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, ch,
                            (cx - cw // 2, cy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            sub_color, 1, cv2.LINE_AA)

                conf_str = f"{ch_conf:.0%}"
                (fw, _), _ = cv2.getTextSize(
                    conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1
                )
                cv2.putText(out, conf_str,
                            (cx - fw // 2, cy + ch_h + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (200, 200, 200), 1, cv2.LINE_AA)
        else:
            # Non-text region: just show the raw read + confidence
            x = int(min(p[0] for p in raw_bbox))
            y = int(min(p[1] for p in raw_bbox))
            _draw_label(out, f"{raw_text}  {confidence:.2f}", x, y,
                        fg=(255, 255, 255), bg=outer_color)

    # ── result banner ───────────────────────────────────────────────────────
    banner_h = 36
    cv2.rectangle(out, (0, H - banner_h), (W, H),
                  (0, 120, 0) if best_text else (60, 60, 60), -1)
    banner_text = (f"ID: {best_text}  ({best_conf:.2%})" if best_text
                   else "No plate ID detected")
    cv2.putText(out, banner_text, (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ocr(png_bytes: bytes) -> tuple[str, float, str, list[dict]]:
    """
    Run plate OCR on the given image bytes.

    Returns ``(plate_id, confidence, overlay_b64, detections)`` where:
      - plate_id         – best plate-like string found (empty string if none)
      - confidence       – PaddleOCR score for the winning detection (0.0 if none)
      - overlay_b64      – base64 JPEG of the plate image annotated with all
                           detected text bounding boxes and a result banner
      - detections       – list of dicts, one per detected text region:
          {
            "bbox":                [[x,y]×4]  (4 corner points, floats),
            "text":                str        (raw PaddleOCR read),
            "normalized_text":     str        (uppercased, stripped),
            "confidence":          float      (PaddleOCR region confidence),
            "is_best_match":       bool,
            "characters":          [str, …],
            "per_char_confidences":[float, …] (weighted-vote per position),
          }
    """
    _assert_loaded()
    img_bgr  = _bytes_to_bgr(png_bytes)
    variants = _ocr_variants(img_bgr)

    # orig_raw_results: bboxes from the original-colour image — for overlay + detections
    # all_preds: every (normalised_text, conf) from every variant — for per-char weights
    orig_raw_results: list[tuple] = []
    all_preds:        list[tuple[str, float]] = []
    best_by_text:     dict[str, tuple[float, float]] = {}

    with tempfile.TemporaryDirectory(prefix="polepad_ocr_") as _tmp:
        tmp_dir = Path(_tmp)
        for variant_name, variant_img in variants.items():
            tmp_path = tmp_dir / f"{variant_name}.png"
            cv2.imwrite(str(tmp_path), variant_img)
            try:
                raw_result = _ocr_reader.predict(str(tmp_path))
                results    = _extract_paddle_candidates(raw_result)
            except Exception:
                results = []
            for raw_bbox, raw_text, confidence in results:
                norm = _normalize_ocr_text(raw_text)
                conf = float(confidence)
                if variant_name == "orig_color":
                    orig_raw_results.append((raw_bbox, raw_text, conf))
                if norm:
                    all_preds.append((norm, conf))
                if not norm or not _is_plate_like(norm):
                    continue
                score    = _score_candidate(norm, conf)
                existing = best_by_text.get(norm)
                if existing is None or conf > existing[0]:
                    best_by_text[norm] = (conf, score)

    best_text, raw_conf = "", 0.0
    if best_by_text:
        candidate = max(best_by_text, key=lambda t: (best_by_text[t][0], best_by_text[t][1]))
        cand_conf, cand_score = best_by_text[candidate]
        if cand_score >= 0.40:
            best_text = candidate
            raw_conf  = round(cand_conf, 4)

    # Build structured detection list
    detections: list[dict] = []
    for raw_bbox, raw_text, conf in orig_raw_results:
        norm = _normalize_ocr_text(raw_text)
        char_confs = _per_char_confidences(norm, all_preds) if norm else []
        bbox_pts: list[list[float]] = (
            [[float(p[0]), float(p[1])] for p in raw_bbox]
            if raw_bbox is not None else []
        )
        detections.append({
            "bbox":                 bbox_pts,
            "text":                 raw_text,
            "normalized_text":      norm,
            "confidence":           round(conf, 4),
            "is_best_match":        bool(best_text) and norm == best_text,
            "characters":           list(norm),
            "per_char_confidences": char_confs,
        })

    overlay_img = _draw_ocr_overlay(
        img_bgr, orig_raw_results, best_text, raw_conf, all_preds
    )
    _, jpeg_buf = cv2.imencode(".jpg", overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    overlay_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    return (best_text, raw_conf, overlay_b64, detections)


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

    pole_type           = ""
    pole_type_conf      = 0.0
    if poles:
        best_pole       = max(poles, key=lambda p: p["confidence"])
        pole_type       = best_pole["class"]
        pole_type_conf  = round(best_pole["confidence"], 4)

    # Per-component max confidence (0.0 when a component is absent)
    component_conf: dict[str, float] = {form_key: 0.0 for form_key in COMPONENT_MAP.values()}
    detected_components: list[str] = []
    for det in all_detections:
        form_key = COMPONENT_MAP.get(det["class"])
        if form_key:
            if det["confidence"] > component_conf[form_key]:
                component_conf[form_key] = round(det["confidence"], 4)
            if form_key not in detected_components:
                detected_components.append(form_key)

    # Vegetation severity confidence: max confidence among vegetation segments
    veg_conf = 0.0
    for seg in segments:
        if seg["class"] == "vegetation" and seg["confidence"] > veg_conf:
            veg_conf = round(seg["confidence"], 4)

    # Encroachment confidence: max confidence among encroaching segments
    enc_conf = 0.0
    for pole in poles:
        for enc_seg in pole.get("encroaching_segments", []):
            if enc_seg["confidence"] > enc_conf:
                enc_conf = round(enc_seg["confidence"], 4)

    confidences: dict[str, float] = {
        "pole_type":           pole_type_conf,
        "transformer":         component_conf.get("transformer",  0.0),
        "insulator":           component_conf.get("insulator",    0.0),
        "streetlight":         component_conf.get("streetlight",  0.0),
        "vegetation_severity": veg_conf,
        "encroachment":        enc_conf,
    }

    annotated     = _annotate_combined(img_bgr, all_detections, segments, poles)
    _, jpeg_buf   = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    segs_out = [{k: v for k, v in s.items() if k != "binary_mask"} for s in segments]

    return {
        "pole_type":           pole_type,
        "detected_components": detected_components,
        "vegetation_severity": severity,
        "encroachment":        encroachment_detected,
        "confidences":         confidences,
        "poles":               poles,
        "all_detections":      all_detections,
        "segments":            segs_out,
        "annotated_image_b64": annotated_b64,
    }
