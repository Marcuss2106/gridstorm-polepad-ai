#!/usr/bin/env python3
"""
Test the /analyze endpoint without a physical device or Godot.

Replicates exactly what the Android app does:
  - Loads a pole image and a plate image from disk
  - Resizes both to 720×720, encodes as JPEG (quality 0.9), base64-encodes
  - POSTs { plate_image_b64, pole_image_b64 } to the running API server
  - Prints the full JSON response
  - Saves the annotated image to output/test_annotated.jpg

Usage:
  # Random pole image from the test set, first plate image from images/
  python test_api.py

  # Explicit paths
  python test_api.py --pole "Utility-Poles-5/test/images/some.jpg" --plate "images/plate.png"

  # Different server (e.g. phone on same WiFi)
  python test_api.py --host http://192.168.1.100:8000
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

_HERE = Path(__file__).resolve().parent

DEFAULT_POLE_DIR  = _HERE / "Utility-Poles-5" / "test" / "images"
DEFAULT_PLATE_DIR = _HERE / "images"
OUTPUT_DIR        = _HERE / "output"
DEFAULT_HOST      = "http://localhost:8000"
CAPTURE_SIZE      = 720
JPEG_QUALITY      = 90   # 0-100


def _pick_image(directory: Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Path:
    """Return a random image file from *directory*."""
    candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    if not candidates:
        raise FileNotFoundError(f"No images found in {directory}")
    return random.choice(candidates)


def _encode_image(path: Path) -> str:
    """Load image, resize to 720×720, JPEG-encode at quality 0.9, return base64 string."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (CAPTURE_SIZE, CAPTURE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"JPEG encoding failed for {path}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _save_annotated(b64_str: str, out_path: Path) -> None:
    jpeg_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("  [warn] Could not decode annotated image.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"  Annotated image saved → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pole",  type=Path, default=None,
                   help="Path to the full-pole image. Default: random from Utility-Poles-5/test/images/")
    p.add_argument("--plate", type=Path, default=None,
                   help="Path to the plate/tag image. Default: random from images/ (falls back to pole image)")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"API server base URL. Default: {DEFAULT_HOST}")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save the annotated output image.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Resolve image paths ───────────────────────────────────────────────────
    pole_path = args.pole
    if pole_path is None:
        if not DEFAULT_POLE_DIR.exists():
            print(f"[error] Default pole image directory not found: {DEFAULT_POLE_DIR}")
            print("        Pass --pole <path> to specify an image manually.")
            return 1
        pole_path = _pick_image(DEFAULT_POLE_DIR)

    plate_path = args.plate
    if plate_path is None:
        if DEFAULT_PLATE_DIR.exists():
            try:
                plate_path = _pick_image(DEFAULT_PLATE_DIR)
            except FileNotFoundError:
                plate_path = pole_path  # fallback: use the pole image for OCR too
        else:
            plate_path = pole_path  # fallback

    print(f"Pole image :  {pole_path.name}")
    print(f"Plate image:  {plate_path.name}")
    print()

    # ── Encode ────────────────────────────────────────────────────────────────
    print("Encoding images as 720×720 JPEG...")
    try:
        pole_b64  = _encode_image(pole_path)
        plate_b64 = _encode_image(plate_path)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    # ── POST to API ───────────────────────────────────────────────────────────
    url = args.host.rstrip("/") + "/analyze"
    print(f"Sending POST {url} ...")
    try:
        resp = requests.post(
            url,
            json={"plate_image_b64": plate_b64, "pole_image_b64": pole_b64},
            timeout=120,
        )
    except requests.ConnectionError:
        print(f"\n[error] Could not connect to {url}")
        print("        Make sure the server is running:")
        print("          uvicorn api:app --host 0.0.0.0 --port 8000")
        return 1
    except requests.Timeout:
        print("\n[error] Request timed out (120 s). The server may still be loading models.")
        return 1

    if resp.status_code != 200:
        print(f"\n[error] Server returned HTTP {resp.status_code}")
        print(resp.text)
        return 1

    data = resp.json()

    # ── Print results ─────────────────────────────────────────────────────────
    SEVERITY_NAMES = ["None", "Low", "Medium", "High"]
    severity_idx   = max(0, min(data.get("vegetation_severity", 0), len(SEVERITY_NAMES) - 1))

    print("\n" + "=" * 50)
    print("  ANALYSIS RESULTS")
    print("=" * 50)
    print(f"  Pole ID          : {data.get('pole_id') or '(not detected)'}")
    print(f"  Pole Type        : {data.get('pole_type') or '(unknown)'}")
    comps = data.get("detected_components", [])
    print(f"  Components       : {', '.join(comps) if comps else '(none detected)'}")
    enc = data.get("encroachment", False)
    print(f"  Encroachment     : {'YES' if enc else 'No'}")
    print(f"  Veg. Severity    : {SEVERITY_NAMES[severity_idx]} ({severity_idx}/3)")
    print("=" * 50)

    # Full JSON (omit the long base64 image field for readability)
    display = {k: v for k, v in data.items() if k != "annotated_image_b64"}
    print("\nFull response (image omitted):")
    print(json.dumps(display, indent=2))

    # ── Save annotated image ──────────────────────────────────────────────────
    if not args.no_save:
        b64_img = data.get("annotated_image_b64", "")
        if b64_img:
            _save_annotated(b64_img, OUTPUT_DIR / "test_annotated.jpg")
        else:
            print("  [warn] No annotated image in response.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
