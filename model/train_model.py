from ultralytics import YOLO
from roboflow import Roboflow
import shutil
from pathlib import Path

# Detection model — all 7 pole component classes (bounding boxes)
detect_model = YOLO("yolo26m.pt")

# Segmentation model — vegetation + guy_guard only (polygon masks)
seg_model = YOLO("yolo26m-seg.pt")

# --- SEGMENTATION DATASET DOWNLOAD ---
# Downloads the same project in yolov11 polygon format into a separate directory
# so the bbox labels in Utility-Poles-4/ are not overwritten.
# NOTE: Before running, verify in Roboflow that guy_guard and vegetation
# are annotated as drawn polygons (not just bounding rectangles).

rf = Roboflow(api_key="038INb0Av0p6eo4CVxxx")
project = rf.workspace("marcuss-workspace").project("utility-poles-kcumt-nt0d2")
version = project.version(5)
seg_dataset = version.download("yolov11", location="Utility-Poles-5-seg")  # polygon format — compatible with YOLO26-seg

# --- DETECTION MODEL ---
# Trains on all 7 classes: composite, guy_guard, insulator, street light,
# transformer, vegetation, wood — outputs bounding boxes.
detect_results = detect_model.train(
    data="Utility-Poles-5/data.yaml",
    epochs=100,
    imgsz=720,
    batch=16,
    device=[0, 1, 2, 3],          # set to "cpu" if no GPU
    optimizer="auto",  # uses MuSGD (YOLO26's new optimizer)
    lr0=0.001,
    patience=20,
    pretrained=True,
    cache=True,
    project="utility-pole-training",
    name="yolo26m-detect",
    save_period=10,
)

# --- FILTER TO 2-CLASS SEGMENTATION DATASET ---
# Keeps only guy_guard (class 1) and vegetation (class 5) from the full 7-class
# seg download and remaps them to class 0 and class 1 respectively.


# Map original class index -> new class index (drop everything else)
KEEP = {1: 0, 5: 1}   # guy_guard -> 0, vegetation -> 1

src_root = Path(seg_dataset.location)
dst_root = Path("Utility-Poles-5-seg-2cls")

for split in ["train", "valid", "test"]:
    (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
    (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)
    lbl_dir = src_root / split / "labels"
    img_dir = src_root / split / "images"
    if not lbl_dir.exists():
        continue
    for lbl_path in lbl_dir.glob("*.txt"):
        new_lines = []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls in KEEP:
                new_lines.append(f"{KEEP[cls]} " + " ".join(parts[1:]))
        if new_lines:
            (dst_root / split / "labels" / lbl_path.name).write_text("\n".join(new_lines))
            for ext in [".jpg", ".jpeg", ".png"]:
                img_src = img_dir / lbl_path.with_suffix(ext).name
                if img_src.exists():
                    shutil.copy(img_src, dst_root / split / "images" / img_src.name)
                    break

(dst_root / "data.yaml").write_text(
    f"train: ../{dst_root}/train/images\n"
    f"val: ../{dst_root}/valid/images\n"
    f"test: ../{dst_root}/test/images\n"
    f"nc: 2\n"
    f"names: ['guy_guard', 'vegetation']\n"
)
print(f"2-class seg dataset written to: {dst_root.resolve()}")

# --- SEGMENTATION MODEL ---
# Trains on 2 classes only: guy_guard + vegetation — outputs polygon masks.
seg_results = seg_model.train(
    data="Utility-Poles-5-seg-2cls/data.yaml",
    epochs=100,
    imgsz=720,
    batch=16,
    device=[0, 1, 2, 3],          # set to "cpu" if no GPU
    optimizer="auto",
    lr0=0.001,
    patience=20,
    pretrained=True,
    cache=True,
    project="utility-pole-training",
    name="yolo26m-seg-veg-guy",
    save_period=10,
)

# --- EXPORT BOTH MODELS FOR MOBILE ---
best_detect = YOLO("utility-pole-training/yolo26m-detect/weights/best.pt")
best_seg = YOLO("utility-pole-training/yolo26m-seg-veg-guy/weights/best.pt")

# Android (TFLite)
best_detect.export(format="tflite")
best_seg.export(format="tflite")