#!/usr/bin/env python3
"""Simple GUI tool to label pole plate text image-by-image."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def normalize_label(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"\s+", "", text)
    return text


def list_images(images_dir: Path) -> list[Path]:
    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(images, key=lambda p: natural_sort_key(p.name))


@dataclass
class LabelRow:
    filename: str
    plate_label: str
    image_path: str
    labeled_at_utc: str


def load_labels(csv_path: Path) -> dict[str, LabelRow]:
    labels: dict[str, LabelRow] = {}
    if not csv_path.exists():
        return labels

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            labels[filename] = LabelRow(
                filename=filename,
                plate_label=(row.get("plate_label") or "").strip(),
                image_path=(row.get("image_path") or "").strip(),
                labeled_at_utc=(row.get("labeled_at_utc") or "").strip(),
            )
    return labels


def write_labels(csv_path: Path, labels: dict[str, LabelRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(labels.values(), key=lambda r: natural_sort_key(r.filename))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "plate_label", "image_path", "labeled_at_utc"],
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "filename": row.filename,
                    "plate_label": row.plate_label,
                    "image_path": row.image_path,
                    "labeled_at_utc": row.labeled_at_utc,
                }
            )


class PlateLabelerApp:
    def __init__(
        self,
        root: tk.Tk,
        all_images: list[Path],
        queue_images: list[Path],
        labels: dict[str, LabelRow],
        output_csv: Path,
    ) -> None:
        self.root = root
        self.all_images = all_images
        self.queue_images = queue_images
        self.labels = labels
        self.output_csv = output_csv
        self.index = 0
        self.finished = False
        self.current_photo: ImageTk.PhotoImage | None = None

        self.root.title("Pole Plate Labeler")
        self.root.geometry("1280x920")

        container = ttk.Frame(root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container)
        header.pack(fill=tk.X)

        self.title_label = ttk.Label(
            header,
            text="Plate Labeler",
            font=("Helvetica", 20, "bold"),
        )
        self.title_label.pack(side=tk.LEFT, anchor=tk.W)

        self.counter_label = ttk.Label(
            header,
            text="0/0",
            font=("Helvetica", 16, "bold"),
        )
        self.counter_label.pack(side=tk.RIGHT, anchor=tk.E)

        self.progress_label = ttk.Label(container, text="", font=("Helvetica", 12))
        self.progress_label.pack(anchor=tk.W, pady=(4, 8))

        self.progress_bar = ttk.Progressbar(container, mode="determinate", maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        self.image_label = ttk.Label(container)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.filename_label = ttk.Label(container, text="", font=("Helvetica", 12, "bold"))
        self.filename_label.pack(anchor=tk.W, pady=(8, 2))

        hint_text = "Type the plate text (big input), Enter saves, Back goes to previous image."
        self.hint_label = ttk.Label(container, text=hint_text, font=("Helvetica", 11))
        self.hint_label.pack(anchor=tk.W, pady=(0, 6))

        self.entry = tk.Entry(container, font=("Helvetica", 34, "bold"))
        self.entry.pack(fill=tk.X, pady=(0, 8))
        self.entry.focus_set()

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X)

        self.back_button = ttk.Button(buttons, text="Back", command=self.go_back)
        self.back_button.pack(side=tk.LEFT, padx=(0, 8))

        self.save_button = ttk.Button(buttons, text="Save + Next (Enter)", command=self.save_and_next)
        self.save_button.pack(side=tk.LEFT, padx=(0, 8))

        self.skip_button = ttk.Button(buttons, text="Skip", command=self.skip_image)
        self.skip_button.pack(side=tk.LEFT, padx=(0, 8))

        self.quit_button = ttk.Button(buttons, text="Quit", command=self.quit_app)
        self.quit_button.pack(side=tk.RIGHT)

        self.root.bind("<Return>", self.save_and_next)
        self.root.bind("<Escape>", self.quit_app)
        self.root.bind("<Control-Left>", self.go_back)

        self.load_current_image()

    def labeled_count(self) -> int:
        labeled_names = {row.filename for row in self.labels.values() if row.plate_label}
        all_names = {p.name for p in self.all_images}
        return len(labeled_names & all_names)

    def load_current_image(self) -> None:
        self.finished = False
        if self.index >= len(self.queue_images):
            self.finish()
            return

        image_path = self.queue_images[self.index]
        completed = self.labeled_count()
        total = len(self.all_images)
        in_queue_total = len(self.queue_images)
        in_queue_pos = self.index + 1

        progress_pct = 100.0 if total == 0 else (completed / total) * 100.0
        self.progress_bar["value"] = progress_pct
        self.counter_label.config(text=f"{completed}/{total}")

        self.progress_label.config(
            text=(
                f"Remaining Queue: {in_queue_pos}/{in_queue_total}   "
                f"Labeled Total: {completed}/{total} ({progress_pct:.1f}%)"
            )
        )
        self.filename_label.config(text=f"File: {image_path.name}")

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((1240, 700), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.current_photo)
        except Exception as exc:
            self.image_label.config(image="")
            self.current_photo = None
            messagebox.showerror("Image Load Error", f"Failed to load {image_path.name}\n{exc}")

        self.entry.config(state="normal")
        self.entry.delete(0, tk.END)
        existing = self.labels.get(image_path.name)
        if existing and existing.plate_label:
            self.entry.insert(0, existing.plate_label)
        self.save_button.config(state="normal")
        self.skip_button.config(state="normal")
        self.entry.focus_set()
        self.back_button.config(state="normal" if self.index > 0 else "disabled")

    def save_and_next(self, event: object | None = None) -> None:
        if self.index >= len(self.queue_images):
            self.finish()
            return

        typed = normalize_label(self.entry.get())
        if not typed:
            messagebox.showwarning("Missing Label", "Type a plate label before saving.")
            return

        image_path = self.queue_images[self.index]
        self.labels[image_path.name] = LabelRow(
            filename=image_path.name,
            plate_label=typed,
            image_path=str(image_path),
            labeled_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_labels(self.output_csv, self.labels)

        self.index += 1
        self.load_current_image()

    def go_back(self, event: object | None = None) -> None:
        if self.finished and self.queue_images:
            self.index = len(self.queue_images) - 1
            self.load_current_image()
            return
        if self.index <= 0:
            return
        self.index -= 1
        self.load_current_image()

    def skip_image(self) -> None:
        if self.index >= len(self.queue_images):
            self.finish()
            return
        self.index += 1
        self.load_current_image()

    def finish(self) -> None:
        self.finished = True
        completed = self.labeled_count()
        total = len(self.all_images)
        self.progress_bar["value"] = 100.0
        self.counter_label.config(text=f"{completed}/{total}")
        self.progress_label.config(
            text=f"Done. Labeled {completed}/{total} images."
        )
        self.filename_label.config(text="No remaining unlabeled images.")
        self.entry.delete(0, tk.END)
        self.entry.config(state="disabled")
        self.save_button.config(state="disabled")
        self.skip_button.config(state="disabled")
        self.back_button.config(state="normal" if len(self.queue_images) > 0 else "disabled")
        messagebox.showinfo(
            "Finished",
            f"Saved labels to:\n{self.output_csv}\n\nUse Back if you need to correct the last labels.",
        )

    def quit_app(self, event: object | None = None) -> None:
        self.root.quit()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=base_dir / "images",
        help="Directory containing images to label.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=base_dir / "dataset" / "plate_labels.csv",
        help="CSV output path for filename/label dataset.",
    )
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="Deprecated alias for editing all images; same as default behavior.",
    )
    parser.add_argument(
        "--unlabeled-only",
        action="store_true",
        help="Only show images not yet labeled in the CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_csv = args.output_csv.resolve()

    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}")
        return 1

    all_images = list_images(images_dir)
    if not all_images:
        print(f"No images found in: {images_dir}")
        return 1

    labels = load_labels(output_csv)
    if args.unlabeled_only and args.relabel_all:
        print("Use either --unlabeled-only or --relabel-all, not both.")
        return 1

    # Default behavior: show all images and prefill existing labels so reruns can edit.
    # Use --unlabeled-only to keep strictly no-replacement across sessions.
    if args.unlabeled_only:
        queue_images = [img for img in all_images if img.name not in labels]
    else:
        queue_images = all_images

    if not queue_images:
        print(f"All images already labeled in: {output_csv}")
        return 0

    root = tk.Tk()
    PlateLabelerApp(
        root=root,
        all_images=all_images,
        queue_images=queue_images,
        labels=labels,
        output_csv=output_csv,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
