# EasyOCR Fine-Tuning Pipeline

This folder contains the training architecture for fine-tuning an EasyOCR
recognizer on your labeled pole plate dataset.

Scope:

- Uses full images only.
- Reuses preprocessing and postprocessing logic from `model/plate_ocr.py`.
- Does not perform image/ROI cropping.

## Scripts

- `prepare_training_data.py`
  - Builds train/val/test manifests from `dataset/plate_labels.csv`
  - Writes charset file and split stats
- `train_easyocr_recognizer.py`
  - Fine-tunes EasyOCR recognizer (`generation1` or `generation2`)
  - Supports base-weight transfer loading, resume checkpoints, and best/last outputs
- `evaluate_easyocr_recognizer.py`
  - Computes loss, exact match, CER on a manifest
  - Optionally saves per-image predictions CSV
- `infer_with_finetuned_model.py`
  - Runs inference over `images/` with plate preprocessing variants
  - Applies plate postprocessing ranking and writes output CSV/JSON
- `common.py`
  - Shared data/model/decoding/preprocessing helpers

## Typical order

1. `prepare_training_data.py`
2. `train_easyocr_recognizer.py`
3. `evaluate_easyocr_recognizer.py`
4. `infer_with_finetuned_model.py`
