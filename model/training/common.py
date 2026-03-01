#!/usr/bin/env python3
"""Shared utilities for EasyOCR plate recognizer fine-tuning."""

from __future__ import annotations

import csv
import importlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Ensure `plate_ocr.py` can be imported when scripts are run from model/training.
MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from easyocr.recognition import AlignCollate
from easyocr.utils import CTCLabelConverter

from plate_ocr import (
    Candidate,
    candidate_forms,
    image_variants,
    is_plate_like,
    normalize_text,
    score_candidate,
    selection_key,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestRow:
    """Single sample row used in training/eval manifests."""

    filename: str
    image_path: Path
    label: str


def natural_sort_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda p: natural_sort_key(p.name),
    )


def resolve_image_path(raw_image_path: str, filename: str, images_dir: Path) -> Path:
    """Resolve image path from CSV row, preferring explicit path when valid."""
    if raw_image_path:
        candidate = Path(raw_image_path)
        if not candidate.is_absolute():
            candidate = (images_dir / candidate.name).resolve()
        if candidate.exists():
            return candidate
    return (images_dir / filename).resolve()


def load_labeled_samples(
    labels_csv: Path,
    images_dir: Path,
) -> tuple[list[ManifestRow], list[str]]:
    """Load latest label for each filename from label CSV."""
    if not labels_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {labels_csv}")

    rows_by_filename: dict[str, ManifestRow] = {}
    missing_files: list[str] = []

    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            filename = (raw_row.get("filename") or "").strip()
            raw_label = (raw_row.get("plate_label") or "").strip()
            raw_image_path = (raw_row.get("image_path") or "").strip()

            if not filename or not raw_label:
                continue

            label = normalize_text(raw_label)
            if not label:
                continue

            image_path = resolve_image_path(raw_image_path, filename, images_dir)
            if not image_path.exists():
                missing_files.append(str(image_path))
                continue

            rows_by_filename[filename] = ManifestRow(
                filename=filename,
                image_path=image_path,
                label=label,
            )

    rows = sorted(rows_by_filename.values(), key=lambda r: natural_sort_key(r.filename))
    return rows, missing_files


def filter_rows(
    rows: list[ManifestRow],
    min_label_length: int = 1,
    max_label_length: int = 32,
    allow_chars: str | None = None,
) -> list[ManifestRow]:
    """Filter rows by label length and optional allowlist."""
    allowed = set(allow_chars) if allow_chars else None
    filtered: list[ManifestRow] = []
    for row in rows:
        if len(row.label) < min_label_length or len(row.label) > max_label_length:
            continue
        if allowed is not None and any(ch not in allowed for ch in row.label):
            continue
        filtered.append(row)
    return filtered


def split_rows(
    rows: list[ManifestRow],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[ManifestRow], list[ManifestRow], list[ManifestRow]]:
    """Create deterministic train/val/test split."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    shuffled = rows[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total == 0:
        return [], [], []

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    if n_train == 0 and total > 0:
        n_train = 1
    if n_train + n_val > total:
        n_val = max(0, total - n_train)

    n_test = total - n_train - n_val
    if n_test == 0 and total >= 3:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    train_rows = shuffled[:n_train]
    val_rows = shuffled[n_train : n_train + n_val]
    test_rows = shuffled[n_train + n_val :]
    return train_rows, val_rows, test_rows


def save_manifest(rows: list[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "image_path", "label"])
        writer.writeheader()
        for row in sorted(rows, key=lambda r: natural_sort_key(r.filename)):
            writer.writerow(
                {
                    "filename": row.filename,
                    "image_path": str(row.image_path),
                    "label": row.label,
                }
            )


def load_manifest(manifest_path: Path) -> list[ManifestRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[ManifestRow] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            filename = (raw_row.get("filename") or "").strip()
            label = normalize_text((raw_row.get("label") or "").strip())
            image_path_raw = (raw_row.get("image_path") or "").strip()

            if not filename or not label or not image_path_raw:
                continue

            image_path = Path(image_path_raw)
            rows.append(ManifestRow(filename=filename, image_path=image_path, label=label))
    return rows


def derive_charset(rows: list[ManifestRow]) -> str:
    chars = {ch for row in rows for ch in row.label}
    if not chars:
        return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

    preferred_order = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
    ordered = [ch for ch in preferred_order if ch in chars]
    extras = sorted(chars - set(preferred_order))
    ordered.extend(extras)
    return "".join(ordered)


def save_charset(charset: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(charset, encoding="utf-8")


def load_charset(charset_path: Path) -> str:
    if not charset_path.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_path}")
    charset = charset_path.read_text(encoding="utf-8").strip("\n")
    if not charset:
        raise ValueError(f"Charset file is empty: {charset_path}")
    return charset


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    recog_network: str,
    input_channel: int,
    output_channel: int,
    hidden_size: int,
    num_class: int,
) -> torch.nn.Module:
    """Instantiate EasyOCR recognition model by network alias."""
    if recog_network == "generation1":
        module_name = "easyocr.model.model"
    elif recog_network == "generation2":
        module_name = "easyocr.model.vgg_model"
    else:
        module_name = recog_network

    module = importlib.import_module(module_name)
    return module.Model(
        input_channel=input_channel,
        output_channel=output_channel,
        hidden_size=hidden_size,
        num_class=num_class,
    )


def extract_state_dict(checkpoint: dict | torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract state dict from raw checkpoint payloads."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find model state dict in checkpoint payload")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[7:]] = value
        else:
            stripped[key] = value
    return stripped


def load_state_partial(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, int]:
    """Load matching tensors only. Useful when class count changed."""
    current = model.state_dict()
    incoming = strip_module_prefix(state_dict)

    matched: dict[str, torch.Tensor] = {}
    for key, value in incoming.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value

    merged = current.copy()
    merged.update(matched)
    model.load_state_dict(merged)

    loaded = len(matched)
    total_current = len(current)
    total_incoming = len(incoming)
    return loaded, total_current, total_incoming


def find_default_base_model(model_dir: Path, recog_network: str) -> Path | None:
    model_dir = model_dir.resolve()
    if recog_network == "generation2":
        candidates = ["english_g2.pth", "latin_g2.pth"]
    else:
        candidates = ["english.pth", "latin.pth"]

    for name in candidates:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


class PlateTrainDataset(Dataset):
    """Dataset using full images with optional preprocessing variants (no cropping)."""

    def __init__(
        self,
        rows: list[ManifestRow],
        mode: str,
        eval_variant: str,
        seed: int,
    ) -> None:
        self.rows = rows
        self.mode = mode
        self.eval_variant = eval_variant
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = cv2.imread(str(row.image_path))
        if image is None:
            return None

        variants = image_variants(image)

        if self.mode == "train":
            variant_name = self.rng.choice(list(variants.keys()))
        else:
            variant_name = self.eval_variant if self.eval_variant in variants else "orig_color"

        selected = variants[variant_name]
        if selected.ndim == 3:
            selected = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY)

        pil_image = Image.fromarray(selected).convert("L")
        return pil_image, row.label, row.filename


def make_collate_fn(
    img_h: int,
    img_w: int,
    adjust_contrast: float,
):
    align_collate = AlignCollate(
        imgH=img_h,
        imgW=img_w,
        keep_ratio_with_pad=True,
        adjust_contrast=adjust_contrast,
    )

    def _collate(batch):
        valid = [sample for sample in batch if sample is not None]
        if not valid:
            return torch.empty(0), [], []

        images, labels, filenames = zip(*valid)
        image_tensors = align_collate(list(images))
        return image_tensors, list(labels), list(filenames)

    return _collate


def build_converter(charset: str) -> CTCLabelConverter:
    return CTCLabelConverter(charset, separator_list={}, dict_pathlist={})


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    clipped = np.clip(np.array(values, dtype=np.float64), 1e-8, 1.0)
    return float(np.exp(np.mean(np.log(clipped))))


def decode_batch_predictions(
    logits: torch.Tensor,
    converter: CTCLabelConverter,
) -> tuple[list[str], list[float]]:
    """Decode CTC predictions and estimate confidence per sample."""
    probs = torch.softmax(logits, dim=2)
    pred_indices = probs.argmax(2)

    batch_size, seq_len = pred_indices.shape
    lengths = torch.IntTensor([seq_len] * batch_size)
    flat_indices = pred_indices.contiguous().view(-1).detach().cpu().numpy()

    texts = converter.decode_greedy(flat_indices, lengths)

    probs_np = probs.detach().cpu().numpy()
    idx_np = pred_indices.detach().cpu().numpy()
    confidences: list[float] = []

    for row_probs, row_idx in zip(probs_np, idx_np):
        chosen_probs: list[float] = []
        prev = -1
        for t, idx in enumerate(row_idx):
            idx_int = int(idx)
            if idx_int != 0 and idx_int != prev:
                chosen_probs.append(float(row_probs[t, idx_int]))
            prev = idx_int
        confidences.append(geometric_mean(chosen_probs))

    return texts, confidences


def edit_distance(a: str, b: str) -> int:
    """Classic Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_ctc_inputs(labels: list[str], converter: CTCLabelConverter) -> tuple[torch.IntTensor, torch.IntTensor]:
    encoded, lengths = converter.encode(labels)
    return encoded, lengths


def rank_candidate_strings(
    raw_predictions: list[tuple[str, float, str]],
) -> list[Candidate]:
    """Apply existing postprocessing from plate_ocr to ranked recognizer outputs."""
    aggregates: dict[str, Candidate] = {}

    for raw_text, confidence, variant in raw_predictions:
        normalized = normalize_text(raw_text)
        if not normalized:
            continue

        for text, penalty in candidate_forms(normalized):
            if not is_plate_like(text):
                continue
            adjusted_conf = max(0.0, min(1.0, float(confidence) - penalty))
            score = score_candidate(text, adjusted_conf)
            candidate = Candidate(
                text=text,
                confidence=adjusted_conf,
                score=score,
                variant=variant,
                bbox=[],
            )
            prev = aggregates.get(text)
            if prev is None or (candidate.score, candidate.confidence) > (
                prev.score,
                prev.confidence,
            ):
                aggregates[text] = candidate

    return sorted(aggregates.values(), key=selection_key, reverse=True)


def prepare_tensor_from_gray(
    gray_image: np.ndarray,
    align_collate: AlignCollate,
    device: torch.device,
) -> torch.Tensor:
    pil_image = Image.fromarray(gray_image).convert("L")
    batch_tensor = align_collate([pil_image])
    return batch_tensor.to(device)


def gather_variant_predictions(
    model: torch.nn.Module,
    converter: CTCLabelConverter,
    image_bgr: np.ndarray,
    align_collate: AlignCollate,
    device: torch.device,
    max_label_length: int,
    variant_filter: set[str] | None,
) -> list[tuple[str, float, str]]:
    """Run recognizer across preprocessing variants for one image (no ROI cropping)."""
    raw_predictions: list[tuple[str, float, str]] = []
    variants = image_variants(image_bgr)

    for variant_name, variant_img in variants.items():
        if variant_filter and variant_name not in variant_filter:
            continue

        if variant_img.ndim == 3:
            gray = cv2.cvtColor(variant_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = variant_img

        image_tensor = prepare_tensor_from_gray(gray, align_collate, device)
        with torch.no_grad():
            text_for_pred = torch.zeros(
                (1, max_label_length + 1),
                dtype=torch.long,
                device=device,
            )
            logits = model(image_tensor, text_for_pred)
            texts, confidences = decode_batch_predictions(logits, converter)

        raw_text = texts[0] if texts else ""
        confidence = confidences[0] if confidences else 0.0
        raw_predictions.append((raw_text, confidence, f"finetuned:{variant_name}"))

    return raw_predictions


def iter_existing_paths(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    return [row for row in rows if row.image_path.exists()]
