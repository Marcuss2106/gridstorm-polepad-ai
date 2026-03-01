# Gridstorm: Polepad AI

## Mobile Application
* ### Features Implemented:
    * Camera Capture
    * Gallery of Previously Scanned Images
    * Tag Verification Form Utilizing Human and AI read inputs

* ### Iterative Process
    * Started implementation in Android Studio
    * Realized through trial and error that the camera function was not working in Android Studio, so we shifted our process over to Godot

## AI Pipeline

The AI backend is served via a FastAPI server (`model/api.py`) backed by an inference module (`model/inference.py`). When the Godot client submits a pole inspection, it sends two images — a close-up of the pole tag and a full-pole photo. Three distinct AI models process them in parallel:

### 1. Pole Plate OCR — PaddleOCR (PP-OCRv5)
  * Accepts a close-up image of the pole's ID tag/plate
  * Runs PP-OCRv5 (server-grade English OCR model) across two image variants: original color and CLAHE-enhanced grayscale
  * Candidate text regions are normalized (uppercased, stripped of whitespace/special chars) and scored against known plate-ID patterns (e.g. `AB1234`, `1234-56`, `12345678`)
  * A weighted ensemble vote across all variants produces **per-character confidence scores**, so the UI can highlight uncertain individual characters
  * The best plate-like match is selected and returned as the `pole_id` with an overall confidence score
  * Returns an annotated overlay image with color-coded bounding boxes: green = best match, orange = other plate-like text, grey = non-plate text, plus a result banner at the bottom

### 2. Component Detection — YOLOv26 Detection Model
  * Accepts the full-pole image
  * Fine-tuned YOLO detection model identifies pole components including:
    * Pole type: `wood`, `composite`
    * Attached hardware: `transformer`, `insulator`, `street light`
  * Each detected object returns a bounding box and a confidence score
  * Pole type and component presence are returned as structured fields with per-class confidence values

### 3. Vegetation Encroachment — YOLOv26 Segmentation Model
  * Runs on the same full-pole image as the detection model
  * Produces pixel-level segmentation masks for `vegetation` and `guy_guard` classes
  * Encroachment is computed by measuring the overlap ratio of each vegetation mask with each detected pole's bounding box
  * Severity is classified on a 0–3 scale:
    * `0` — No encroachment
    * `1` — Low (< 20% overlap)
    * `2` — Medium (20–50% overlap)
    * `3` — High (> 50% overlap)
  * Returns an annotated full-pole image with segmentation fills, detection boxes, and encroachment highlights

### 4. YOLO Model Ensembling
The detection and segmentation models are deliberately kept as two separate fine-tuned YOLO networks that run on the same full-pole image, and their outputs are fused at inference time. This is a **heterogeneous ensemble** — each model is specialized for a different task and contributes unique information that the other cannot provide:

| Model | Task | Output used downstream |
|-------|------|------------------------|
| Detection (`yolo26m-detect`) | Bounding boxes for poles and hardware | Pole bounding boxes that define the region-of-interest for encroachment; component labels and per-class confidence scores |
| Segmentation (`yolo26m-seg-veg-guy`) | Pixel-level masks for vegetation and guy guards | Binary masks that are cropped to each pole's bounding box to measure vegetation overlap ratio |

**Why two models instead of one?**
A single YOLO model trained to do both bounding-box detection and instance segmentation on all classes simultaneously would require a much larger, harder-to-train dataset. By splitting responsibilities, each model can be fine-tuned on a smaller, focused dataset and retrained independently without disrupting the other. The bounding boxes from the detection model act as **spatial anchors** that give the segmentation masks a precise context window for computing encroachment — neither model could produce the final severity score alone.

**Fusion step:**
After both models run, `_compute_encroachment()` crops each segmentation mask to each detected pole's bounding box and calculates the overlap ratio. The maximum ratio across all encroaching segments drives the severity classification. Confidence values from both models are merged into a single `confidences` dictionary returned to the client, so the UI can surface uncertainty for every field regardless of which model produced it.

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/analyze` | Analyze a pole image pair; returns structured JSON + annotated images |
| `POST` | `/submit` | Accept reviewer-corrected pole data after AI pre-fill |

Start the server with:
```
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Integration
  * Godot AI integration using FastAPI
  * AI-generated tags and attribute suggestions are pre-filled, reducing human cognitive load
  * Users can confirm, dispute, or supplement data for each asset
  * System computes consensus-based confidence scores, which improve data reliability over time

## Learning Process
  * Adaptability throughout FrontEnd development
  * Data integration through the front and back end
  * API and AI integration

## Deliverables Met
  * Image ingestion + AI analysis
  * Asset tag detection + extraction
  * Structured output generation
  * Confidence scoring and uncertainty highlighting

## Impact
  * Reduced manual inspection load
  * Enabled distributing verification of pole assets
  * Built a foundation for a functioning product
