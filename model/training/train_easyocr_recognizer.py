#!/usr/bin/env python3
"""Fine-tune EasyOCR recognizer on pole plate labels.

This training pipeline uses full images only and applies preprocessing variants from
`plate_ocr.py` during data loading. It does not perform ROI cropping.
"""

from __future__ import annotations

import argparse
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
    find_default_base_model,
    iter_existing_paths,
    load_charset,
    load_manifest,
    load_state_partial,
    make_collate_fn,
    normalize_text,
    save_json,
    set_seed,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=model_dir / "dataset" / "prepared" / "train_manifest.csv",
        help="Training manifest CSV.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=model_dir / "dataset" / "prepared" / "val_manifest.csv",
        help="Validation manifest CSV.",
    )
    parser.add_argument(
        "--charset-file",
        type=Path,
        default=model_dir / "dataset" / "prepared" / "charset.txt",
        help="Charset file used by CTC converter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_dir / "training" / "runs" / "easyocr_plate_ft",
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=model_dir / "easyocr_models",
        help="EasyOCR model storage dir (for default base checkpoints).",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Optional base model checkpoint path for transfer learning.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from an existing training checkpoint.",
    )
    parser.add_argument(
        "--no-base-weights",
        action="store_true",
        help="Disable loading EasyOCR base weights.",
    )
    parser.add_argument(
        "--recog-network",
        default="generation2",
        help="Recognizer network alias: generation1 or generation2.",
    )
    parser.add_argument("--input-channel", type=int, default=1)
    parser.add_argument("--output-channel", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=256)
    parser.add_argument("--max-label-length", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--eval-variant",
        default="clahe",
        help="Preprocessing variant to use for val set.",
    )
    parser.add_argument(
        "--train-adjust-contrast",
        type=float,
        default=0.0,
        help="Contrast adjustment value passed to EasyOCR AlignCollate for training.",
    )
    parser.add_argument(
        "--val-adjust-contrast",
        type=float,
        default=0.0,
        help="Contrast adjustment value passed to EasyOCR AlignCollate for validation.",
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze FeatureExtraction layers during fine-tuning.",
    )
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    converter,
    ctc_loss,
    device: torch.device,
    max_label_length: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    exact_matches = 0
    cer_numerator = 0
    cer_denominator = 0

    with torch.no_grad():
        for image_tensors, labels, _ in loader:
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

            pred_texts, _ = decode_batch_predictions(logits, converter)
            for pred_text, true_text in zip(pred_texts, labels):
                pred_norm = normalize_text(pred_text)
                true_norm = normalize_text(true_text)
                total_samples += 1
                if pred_norm == true_norm:
                    exact_matches += 1
                cer_numerator += edit_distance(pred_norm, true_norm)
                cer_denominator += max(1, len(true_norm))

    avg_loss = total_loss / max(1, total_batches)
    exact = exact_matches / max(1, total_samples)
    cer = cer_numerator / max(1, cer_denominator)
    return {
        "loss": avg_loss,
        "exact_match": exact,
        "cer": cer,
        "samples": float(total_samples),
    }


def maybe_freeze_feature_extractor(model: torch.nn.Module) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("FeatureExtraction"):
            param.requires_grad = False
            frozen += 1
    return frozen


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_exact: float,
    best_cer: float,
    charset: str,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_exact": best_exact,
        "best_cer": best_cer,
        "charset": charset,
        "config": config,
    }
    torch.save(checkpoint, path)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    train_manifest = args.train_manifest.resolve()
    val_manifest = args.val_manifest.resolve()
    charset_file = args.charset_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = iter_existing_paths(load_manifest(train_manifest))
    val_rows = iter_existing_paths(load_manifest(val_manifest))
    charset = load_charset(charset_file)

    if not train_rows:
        raise RuntimeError(f"No training samples found in {train_manifest}")
    if not val_rows:
        raise RuntimeError(f"No validation samples found in {val_manifest}")

    converter = build_converter(charset)
    num_class = len(converter.character)

    device = choose_device(args.device)

    model = build_model(
        recog_network=args.recog_network,
        input_channel=args.input_channel,
        output_channel=args.output_channel,
        hidden_size=args.hidden_size,
        num_class=num_class,
    ).to(device)

    train_dataset = PlateTrainDataset(
        rows=train_rows,
        mode="train",
        eval_variant=args.eval_variant,
        seed=args.seed,
    )
    val_dataset = PlateTrainDataset(
        rows=val_rows,
        mode="eval",
        eval_variant=args.eval_variant,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate_fn(
            img_h=args.img_h,
            img_w=args.img_w,
            adjust_contrast=args.train_adjust_contrast,
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate_fn(
            img_h=args.img_h,
            img_w=args.img_w,
            adjust_contrast=args.val_adjust_contrast,
        ),
    )

    start_epoch = 1
    best_exact = -1.0
    best_cer = 1e9

    if args.resume:
        resume_path = args.resume.resolve()
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = strip_module_prefix(extract_state_dict(payload))
        model.load_state_dict(state_dict, strict=True)
        optimizer_state = payload.get("optimizer")
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_exact = float(payload.get("best_exact", best_exact))
        best_cer = float(payload.get("best_cer", best_cer))
        resume_message = f"Resumed from {resume_path} at epoch {start_epoch}."
    else:
        resume_message = ""

        if not args.no_base_weights:
            if args.base_model:
                base_path = args.base_model.resolve()
            else:
                base_path = find_default_base_model(args.model_dir.resolve(), args.recog_network)

            if base_path and base_path.exists():
                payload = torch.load(base_path, map_location=device, weights_only=False)
                state_dict = extract_state_dict(payload)
                loaded, total_current, total_incoming = load_state_partial(model, state_dict)
                resume_message = (
                    f"Loaded partial base weights from {base_path} "
                    f"({loaded}/{total_current} tensors matched; source tensors={total_incoming})."
                )
            else:
                resume_message = "No base model found; training from random initialization."

    if args.freeze_feature_extractor:
        frozen = maybe_freeze_feature_extractor(model)
        print(f"Frozen feature extractor tensors: {frozen}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.resume:
        payload = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        optimizer_state = payload.get("optimizer")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    config = {
        "recog_network": args.recog_network,
        "input_channel": args.input_channel,
        "output_channel": args.output_channel,
        "hidden_size": args.hidden_size,
        "img_h": args.img_h,
        "img_w": args.img_w,
        "max_label_length": args.max_label_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "charset_file": str(charset_file),
        "charset": charset,
    }

    metrics_history: list[dict] = []

    if resume_message:
        print(resume_message)
    print(f"Training samples: {len(train_rows)}")
    print(f"Validation samples: {len(val_rows)}")
    print(f"Charset size: {len(charset)}")
    print(f"Device: {device}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_batches = 0
        running_samples = 0

        for image_tensors, labels, _ in train_loader:
            if image_tensors.numel() == 0:
                continue

            image_tensors = image_tensors.to(device)
            batch_size = image_tensors.size(0)
            text_for_pred = torch.zeros(
                (batch_size, args.max_label_length + 1),
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            running_batches += 1
            running_samples += batch_size

        train_loss = running_loss / max(1, running_batches)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            converter=converter,
            ctc_loss=ctc_loss,
            device=device,
            max_label_length=args.max_label_length,
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_exact_match": float(val_metrics["exact_match"]),
            "val_cer": float(val_metrics["cer"]),
            "train_samples": running_samples,
            "val_samples": int(val_metrics["samples"]),
        }
        metrics_history.append(metrics)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_exact={metrics['val_exact_match']:.4f} "
            f"val_cer={metrics['val_cer']:.4f}"
        )

        is_better = False
        if metrics["val_exact_match"] > best_exact:
            is_better = True
        elif metrics["val_exact_match"] == best_exact and metrics["val_cer"] < best_cer:
            is_better = True

        if is_better:
            best_exact = metrics["val_exact_match"]
            best_cer = metrics["val_cer"]
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_exact=best_exact,
                best_cer=best_cer,
                charset=charset,
                config=config,
            )

        if epoch % args.save_every == 0:
            save_checkpoint(
                path=output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_exact=best_exact,
                best_cer=best_cer,
                charset=charset,
                config=config,
            )

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_exact=best_exact,
            best_cer=best_cer,
            charset=charset,
            config=config,
        )

        save_json(
            {
                "config": config,
                "best_exact": best_exact,
                "best_cer": best_cer,
                "history": metrics_history,
            },
            output_dir / "metrics.json",
        )

    summary = {
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_exact_match": best_exact,
        "best_cer": best_cer,
        "epochs_trained": len(metrics_history),
        "metrics_file": str(output_dir / "metrics.json"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Last checkpoint: {output_dir / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
