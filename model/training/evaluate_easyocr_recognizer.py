#!/usr/bin/env python3
"""Evaluate a fine-tuned EasyOCR recognizer checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.common import (
    PlateTrainDataset,
    build_converter,
    build_ctc_inputs,
    build_model,
    choose_device,
    decode_batch_predictions,
    edit_distance,
    extract_state_dict,
    iter_existing_paths,
    load_charset,
    load_manifest,
    make_collate_fn,
    normalize_text,
    save_json,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_dir / "training" / "runs" / "easyocr_plate_ft" / "best.pt",
        help="Fine-tuned checkpoint path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=model_dir / "dataset" / "prepared" / "test_manifest.csv",
        help="Manifest to evaluate.",
    )
    parser.add_argument(
        "--charset-file",
        type=Path,
        default=None,
        help="Optional charset file override. Defaults to charset in checkpoint config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_dir / "training" / "eval",
        help="Directory for metrics and prediction CSV output.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-variant",
        default="clahe",
        help="Preprocessing variant to use for evaluation.",
    )
    parser.add_argument(
        "--adjust-contrast",
        type=float,
        default=0.0,
        help="Contrast adjustment for AlignCollate.",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=None,
        help="Override max label length from checkpoint config.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample prediction CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_config = payload.get("config", {}) if isinstance(payload, dict) else {}

    if args.charset_file:
        charset = load_charset(args.charset_file.resolve())
    else:
        charset = payload.get("charset") or checkpoint_config.get("charset")
        if not charset:
            charset_file = checkpoint_config.get("charset_file")
            if charset_file:
                charset = load_charset(Path(charset_file).resolve())
            else:
                raise RuntimeError(
                    "Charset missing from checkpoint. Provide --charset-file explicitly."
                )

    recog_network = checkpoint_config.get("recog_network", "generation2")
    input_channel = int(checkpoint_config.get("input_channel", 1))
    output_channel = int(checkpoint_config.get("output_channel", 256))
    hidden_size = int(checkpoint_config.get("hidden_size", 256))
    img_h = int(checkpoint_config.get("img_h", 64))
    img_w = int(checkpoint_config.get("img_w", 256))
    max_label_length = int(
        args.max_label_length
        if args.max_label_length is not None
        else checkpoint_config.get("max_label_length", 16)
    )

    converter = build_converter(charset)
    model = build_model(
        recog_network=recog_network,
        input_channel=input_channel,
        output_channel=output_channel,
        hidden_size=hidden_size,
        num_class=len(converter.character),
    ).to(device)

    state_dict = strip_module_prefix(extract_state_dict(payload))
    model.load_state_dict(state_dict, strict=True)

    rows = iter_existing_paths(load_manifest(manifest_path))
    if not rows:
        raise RuntimeError(f"No usable samples in manifest: {manifest_path}")

    dataset = PlateTrainDataset(
        rows=rows,
        mode="eval",
        eval_variant=args.eval_variant,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate_fn(
            img_h=img_h,
            img_w=img_w,
            adjust_contrast=args.adjust_contrast,
        ),
    )

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    exact_matches = 0
    cer_numerator = 0
    cer_denominator = 0
    predictions_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for image_tensors, labels, filenames in loader:
            if image_tensors.numel() == 0:
                continue

            image_tensors = image_tensors.to(device)
            batch_size = image_tensors.size(0)

            text_for_pred = torch.zeros(
                (batch_size, max_label_length + 1),
                dtype=torch.long,
                device=device,
            )
            logits = model(image_tensors, text_for_pred)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)

            encoded, lengths = build_ctc_inputs(labels, converter)
            preds_size = torch.full(
                (batch_size,),
                fill_value=logits.size(1),
                dtype=torch.int32,
                device=device,
            )

            loss = ctc_loss(
                log_probs,
                encoded.to(device),
                preds_size,
                lengths.to(device),
            )

            total_loss += float(loss.item())
            total_batches += 1

            pred_texts, pred_confidences = decode_batch_predictions(logits, converter)

            for filename, pred_text, confidence, true_text in zip(
                filenames, pred_texts, pred_confidences, labels
            ):
                pred_norm = normalize_text(pred_text)
                true_norm = normalize_text(true_text)
                is_correct = pred_norm == true_norm

                total_samples += 1
                if is_correct:
                    exact_matches += 1

                distance = edit_distance(pred_norm, true_norm)
                cer_numerator += distance
                cer_denominator += max(1, len(true_norm))

                predictions_rows.append(
                    {
                        "filename": filename,
                        "label": true_norm,
                        "prediction": pred_norm,
                        "raw_prediction": pred_text,
                        "confidence": round(float(confidence), 6),
                        "is_correct": int(is_correct),
                        "edit_distance": distance,
                    }
                )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "samples": total_samples,
        "loss": total_loss / max(1, total_batches),
        "exact_match": exact_matches / max(1, total_samples),
        "cer": cer_numerator / max(1, cer_denominator),
        "recog_network": recog_network,
        "img_h": img_h,
        "img_w": img_w,
        "max_label_length": max_label_length,
        "eval_variant": args.eval_variant,
        "charset_size": len(charset),
    }

    metrics_path = output_dir / "metrics.json"
    save_json(metrics, metrics_path)

    if args.save_predictions:
        predictions_path = output_dir / "predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filename",
                    "label",
                    "prediction",
                    "raw_prediction",
                    "confidence",
                    "is_correct",
                    "edit_distance",
                ],
            )
            writer.writeheader()
            for row in predictions_rows:
                writer.writerow(row)
        print(f"Saved predictions: {predictions_path}")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
