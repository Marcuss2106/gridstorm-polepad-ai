# Pole Plate OCR (Multi-Engine)

This folder contains a multi-engine OCR pipeline for extracting pole plate text
from images in `model/images`.

## What it does

- Runs multiple OCR engines over the same preprocessed image variants:
  - `easyocr`
  - `tesseract`
  - `paddleocr`
  - `kraken` (CLI + model file)
  - `gocr` (CLI)
- Keeps the existing preprocessing pipeline (brightness/contrast/saturation + grayscale/CLAHE/etc).
- Keeps ROI crop detection, confusion normalization, template-aware scoring, and consensus voting.
- Tries multiple image preprocessing variants for each image, including
  brightness/contrast/saturation adjustments.
- Applies confusion-aware canonicalization (e.g. `IPD` -> `PD`, `6-####` -> `C-####`).
- Uses template-aware scoring for likely plate formats (`PD#####`, `X-####`, numeric tags).
- Uses consensus voting across all OCR variants to rank final candidates.
- Filters out low-digit/noise tokens to reduce non-plate words winning selection.
- Adds contour-based ROI plate crops (rectangular/circular) before OCR to improve
  weak full-frame reads.
- Writes results to:
  - `model/output/plate_predictions.csv`
  - `model/output/plate_predictions.json`
- Stores EasyOCR model weights inside `model/easyocr_models`.

## Setup (inside `model/`)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run OCR

```bash
. .venv/bin/activate
python plate_ocr.py
```

Optional flags:

```bash
python plate_ocr.py --top-k 5 --min-score 0.45
python plate_ocr.py --images-dir ./images --output-dir ./output --model-dir ./easyocr_models
python plate_ocr.py --engines easyocr,tesseract,paddleocr
python plate_ocr.py --engines easyocr,kraken --kraken-model /path/to/model.mlmodel
```

## Labeling GUI

Use this to label each image with the plate text for model fine-tuning:

```bash
python label_plate_dataset.py
```

Optional flags:

```bash
python label_plate_dataset.py --images-dir ./images --output-csv ./dataset/plate_labels.csv
python label_plate_dataset.py --unlabeled-only
```

Behavior:

- Shows images in sorted order (top to bottom by filename).
- Saves the CSV after every label entry (immediate persistence).
- By default, re-running the app shows all images with existing labels pre-filled.
  Typing a different value updates that row in the spreadsheet.
- Use `--unlabeled-only` for strict no-replacement across sessions.
- Enter saves the typed label and moves to next image.
- Back button lets you return to the previous image and fix a mistake.
- Top-right corner shows labeling progress as `x/total`.
- Saves dataset rows as `filename, plate_label, image_path, labeled_at_utc`.

## EasyOCR Fine-Tuning (No Cropping)

This training architecture is built under `model/training` and keeps the
existing preprocessing/postprocessing logic from `plate_ocr.py`.
Detailed script roles are listed in `model/training/README.md`.

Important:

- Fine-tuning uses full images only.
- It does **not** perform ROI/image cropping.

### 1) Prepare manifests from labels

```bash
python training/prepare_training_data.py \
  --labels-csv ./dataset/plate_labels.csv \
  --images-dir ./images \
  --output-dir ./dataset/prepared
```

Outputs:

- `dataset/prepared/train_manifest.csv`
- `dataset/prepared/val_manifest.csv`
- `dataset/prepared/test_manifest.csv`
- `dataset/prepared/charset.txt`
- `dataset/prepared/stats.json`

### 2) Fine-tune recognizer

```bash
python training/train_easyocr_recognizer.py \
  --train-manifest ./dataset/prepared/train_manifest.csv \
  --val-manifest ./dataset/prepared/val_manifest.csv \
  --charset-file ./dataset/prepared/charset.txt \
  --output-dir ./training/runs/easyocr_plate_ft \
  --model-dir ./easyocr_models \
  --recog-network generation2 \
  --epochs 40 \
  --batch-size 16
```

Optional examples:

```bash
python training/train_easyocr_recognizer.py --freeze-feature-extractor
python training/train_easyocr_recognizer.py --resume ./training/runs/easyocr_plate_ft/last.pt
python training/train_easyocr_recognizer.py --base-model ./easyocr_models/english_g2.pth
```

### 3) Evaluate checkpoint

```bash
python training/evaluate_easyocr_recognizer.py \
  --checkpoint ./training/runs/easyocr_plate_ft/best.pt \
  --manifest ./dataset/prepared/test_manifest.csv \
  --output-dir ./training/eval \
  --save-predictions
```

### 4) Run inference with fine-tuned model

```bash
python training/infer_with_finetuned_model.py \
  --checkpoint ./training/runs/easyocr_plate_ft/best.pt \
  --images-dir ./images \
  --output-dir ./output \
  --top-k 4 \
  --min-score 0.35
```

## Notes

- The first run downloads EasyOCR model files.
- `kraken` and `gocr` are CLI binaries and must be installed in `PATH`.
- `kraken` requires `--kraken-model` to run.
- If an engine is unavailable, the script logs a warning and continues with available engines.
- The script sets certificate bundle env vars via `certifi` to avoid SSL issues.
