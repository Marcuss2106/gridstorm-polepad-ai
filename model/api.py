"""
FastAPI server for utility pole AI analysis.

Startup:
    uvicorn api:app --host 0.0.0.0 --port 8000

The Android Godot client sends POST /analyze with two base64-encoded PNG images.
The server returns a structured JSON with OCR pole ID, detected components,
vegetation encroachment severity, and an annotated image.
"""

from __future__ import annotations

import base64
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load models on startup so any crash is visible in the terminal
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("=== Loading AI models (this may take 30-60 s on first run) ===")
    try:
        from inference import load_models  # noqa: PLC0415
        load_models()
        logger.info("=== Models ready — server accepting requests ===")
    except Exception:
        traceback.print_exc()
        logger.error("=== Model loading FAILED — /analyze will return 503 ===")
    yield


app = FastAPI(title="Polepad AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    Both images must be base64-encoded PNG or JPEG bytes.
    plate_image_b64: close-up of the pole tag/plate (used for OCR)
    pole_image_b64:  full-pole shot (used for YOLO detection + segmentation)
    """
    plate_image_b64: str
    pole_image_b64: str


class AnalyzeResponse(BaseModel):
    pole_id:             str
    pole_type:           str
    detected_components: list[str]
    vegetation_severity: int
    encroachment:        bool
    annotated_image_b64: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a utility pole image pair.

    Runs EasyOCR on the plate image to extract the pole ID, and runs
    YOLO detection + segmentation on the full-pole image to identify
    components and vegetation encroachment.
    """
    from inference import run_encroachment, run_ocr  # noqa: PLC0415

    # Decode base64 → raw bytes
    try:
        plate_bytes = base64.b64decode(body.plate_image_b64)
        pole_bytes  = base64.b64decode(body.pole_image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {exc}") from exc

    # OCR pass on the plate image
    try:
        pole_id = run_ocr(plate_bytes)
        logger.info("OCR result: %r", pole_id)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR error: {exc}") from exc

    # YOLO inference on the full-pole image
    try:
        enc_result = run_encroachment(pole_bytes)
        logger.info(
            "Encroachment: %s | severity: %d | components: %s",
            enc_result["encroachment"],
            enc_result["vegetation_severity"],
            enc_result["detected_components"],
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}") from exc

    return AnalyzeResponse(
        pole_id=pole_id,
        pole_type=enc_result["pole_type"],
        detected_components=enc_result["detected_components"],
        vegetation_severity=enc_result["vegetation_severity"],
        encroachment=enc_result["encroachment"],
        annotated_image_b64=enc_result["annotated_image_b64"],
    )
