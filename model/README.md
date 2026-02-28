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

## Notes

- The first run downloads EasyOCR model files.
- `kraken` and `gocr` are CLI binaries and must be installed in `PATH`.
- `kraken` requires `--kraken-model` to run.
- If an engine is unavailable, the script logs a warning and continues with available engines.
- The script sets certificate bundle env vars via `certifi` to avoid SSL issues.
