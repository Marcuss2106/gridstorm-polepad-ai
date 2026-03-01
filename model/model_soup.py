"""
Model Soup — Checkpoint Weight Averaging
=========================================
Averages the weights of multiple YOLO checkpoints from the same training run
into a single .pt file. The output is identical in size and architecture to any
individual checkpoint but typically generalises better than any single one.

Reference: Wortsman et al. (2022) "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference time."

Usage:
    python model_soup.py
"""

from __future__ import annotations

import torch
from copy import deepcopy
from pathlib import Path


def soup_checkpoints(paths: list[str | Path], output_path: str | Path) -> None:
    """Average weights from multiple Ultralytics YOLO checkpoints.

    Parameters
    ----------
    paths:
        Paths to the .pt checkpoint files to merge. All must share the same
        model architecture.
    output_path:
        Where to save the merged checkpoint.
    """
    paths = [Path(p) for p in paths]
    output_path = Path(output_path)

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoints not found: {missing}")

    print(f"Loading {len(paths)} checkpoints …")
    ckpts = [
        torch.load(p, map_location="cpu", weights_only=False) for p in paths
    ]

    def _extract_model(ckpt):
        """Return the model object from an Ultralytics checkpoint dict.

        Ultralytics saves checkpoints as dicts with a 'model' key and an 'ema'
        key. When EMA training is enabled (the default), 'model' is set to None
        and the actual weights live under 'ema'. Fall back in order:
        ema → model → raw object.
        """
        if not isinstance(ckpt, dict):
            return ckpt
        for key in ("ema", "model"):
            obj = ckpt.get(key)
            if obj is not None:
                return obj
        raise ValueError(f"Checkpoint has no model weights. Keys: {list(ckpt.keys())}")

    def _state_dict(ckpt):
        return _extract_model(ckpt).state_dict()

    state_dicts = [_state_dict(c) for c in ckpts]
    print(f"  source: {'ema' if ckpts[0].get('ema') is not None else 'model'} weights")

    print("Averaging weights …")
    avg_sd = {}
    for key in state_dicts[0]:
        original_dtype = state_dicts[0][key].dtype
        # Accumulate in float32 to avoid precision loss, then restore dtype
        tensors = torch.stack([sd[key].float() for sd in state_dicts])
        avg_sd[key] = tensors.mean(dim=0).to(original_dtype)

    # Use first checkpoint as the output shell so class names and architecture
    # metadata are preserved, then overwrite with the averaged weights.
    out_ckpt = deepcopy(ckpts[0])
    target_model = _extract_model(out_ckpt)
    target_model.load_state_dict(avg_sd)

    if isinstance(out_ckpt, dict):
        # If weights came from ema, promote them to the 'model' key so the
        # output checkpoint can be loaded without EMA-aware code.
        out_ckpt["model"] = target_model
        # Drop training-only fields — reduces file size by roughly half
        for field in ("optimizer", "ema", "updates"):
            out_ckpt.pop(field, None)
    else:
        out_ckpt = target_model

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, output_path)
    print(f"Saved soup → {output_path}")


if __name__ == "__main__":
    BASE = Path("trained_weights")

    # ── Detection model ───────────────────────────────────────────────────────
    detect_ckpts = [
        BASE / "yolo26m-detect/weights/epoch40.pt",
        BASE / "yolo26m-detect/weights/epoch50.pt",
        BASE / "yolo26m-detect/weights/epoch60.pt",
        BASE / "yolo26m-detect/weights/best.pt",
    ]
    soup_checkpoints(
        paths=detect_ckpts,
        output_path=BASE / "yolo26m-detect/weights/soup.pt",
    )

    # ── Segmentation model ────────────────────────────────────────────────────
    seg_ckpts = [
        BASE / "yolo26m-seg-veg-guy/weights/epoch30.pt",
        BASE / "yolo26m-seg-veg-guy/weights/epoch40.pt",
        BASE / "yolo26m-seg-veg-guy/weights/epoch50.pt",
        BASE / "yolo26m-seg-veg-guy/weights/best.pt",
    ]
    soup_checkpoints(
        paths=seg_ckpts,
        output_path=BASE / "yolo26m-seg-veg-guy/weights/soup.pt",
    )

    print("\nDone. Update your notebook config:")
    print('  DETECT_MODEL_PATH = "trained_weights/yolo26m-detect/weights/soup.pt"')
    print('  SEG_MODEL_PATH    = "trained_weights/yolo26m-seg-veg-guy/weights/soup.pt"')
