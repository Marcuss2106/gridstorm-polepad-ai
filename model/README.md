# Pole Plate OCR (EasyOCR)

This folder contains an out-of-the-box EasyOCR pipeline for extracting pole
plate text from images in `model/images`.

## What it does

- Uses EasyOCR English model with plate-safe character allowlist.
- Tries multiple image preprocessing variants for each image, including
  brightness/contrast/saturation adjustments.
- Selects the final plate by highest OCR confidence across variants (score still
  used for filtering/tie-breaks).
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
```

## Notes

- The first run downloads EasyOCR model files.
- The script sets certificate bundle env vars via `certifi` to avoid SSL issues.
- If OCR misses some plates, gather labeled crops and train a custom recognizer
  next; this repo currently uses out-of-the-box EasyOCR as requested.
