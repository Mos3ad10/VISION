from pathlib import Path
import json

import torch
from ultralytics import YOLO

# =========================================================
# Paths
# =========================================================
DATASET_ROOT = Path(r"D:\DATASET")

TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"

VAL_IMAGES = DATASET_ROOT / "valid" / "images"
VAL_LABELS = DATASET_ROOT / "valid" / "labels"

TEST_IMAGES = DATASET_ROOT / "test" / "images"
TEST_LABELS = DATASET_ROOT / "test" / "labels"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "dataset.yaml"
TASK_INFO_JSON = SCRIPT_DIR / "task_info.json"
RUNS_DIR = SCRIPT_DIR / "runs"

# =========================================================
# Settings
# =========================================================
CLASS_ID = 0
CLASS_NAME = "Waled"

EPOCHS = 50
IMAGE_SIZE = 640

DETECT_MODEL = "yolov8n.pt"
SEG_MODEL = "yolov8n-seg.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_exists(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def write_dataset_yaml():
    yaml_text = f"""path: {DATASET_ROOT.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  {CLASS_ID}: {CLASS_NAME}
"""
    DATA_YAML.write_text(yaml_text, encoding="utf-8")


def infer_line_type(parts, file_path: Path, line_no: int):
    """
    Detect label type from a single line.

    Detection:
        class x_center y_center width height
        => 5 tokens

    Segmentation:
        class x1 y1 x2 y2 x3 y3 ...
        => odd number of tokens, at least 7
    """
    try:
        values = [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric label value in {file_path} line {line_no}: {' '.join(parts)}"
        ) from exc

    if len(parts) == 5:
        # Detection
        x, y, w, h = values[1:]
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            raise ValueError(
                f"Detection label out of range in {file_path} line {line_no}: {' '.join(parts)}"
            )
        return "detect"

    if len(parts) >= 7 and len(parts) % 2 == 1:
        # Segmentation
        coords = values[1:]
        if not all(0.0 <= v <= 1.0 for v in coords):
            raise ValueError(
                f"Segmentation label out of range in {file_path} line {line_no}: {' '.join(parts)}"
            )
        return "segment"

    raise ValueError(
        f"Unsupported label format in {file_path} line {line_no}:\n"
        f"{' '.join(parts)}\n\n"
        f"Detection expects: class x_center y_center width height\n"
        f"Segmentation expects: class x1 y1 x2 y2 x3 y3 ..."
    )


def detect_dataset_task(label_dirs):
    """
    Scan all label files across train/valid/test and infer one unified task:
    - 'detect'
    - 'segment'
    """
    found_types = set()
    non_empty_lines = 0

    for label_dir in label_dirs:
        txt_files = sorted(label_dir.glob("*.txt"))
        for txt_file in txt_files:
            for line_no, raw_line in enumerate(
                txt_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                task_type = infer_line_type(parts, txt_file, line_no)
                found_types.add(task_type)
                non_empty_lines += 1

    if non_empty_lines == 0:
        raise ValueError("No non-empty label lines were found in train/valid/test labels.")

    if len(found_types) > 1:
        raise ValueError(
            f"Mixed label formats detected: {found_types}. "
            f"Use only one format across train/valid/test."
        )

    return found_types.pop()


def remap_all_labels_to_single_class(label_dir: Path):
    """
    Force every annotation to class 0 (Waled).
    """
    txt_files = sorted(label_dir.glob("*.txt"))

    for txt_file in txt_files:
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        new_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            parts[0] = str(CLASS_ID)
            new_lines.append(" ".join(parts))

        txt_file.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8"
        )


def save_task_info(task: str, run_name: str):
    info = {
        "task": task,
        "run_name": run_name,
        "class_id": CLASS_ID,
        "class_name": CLASS_NAME,
        "weights_name": "best.pt"
    }
    TASK_INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    ensure_exists([
        TRAIN_IMAGES, TRAIN_LABELS,
        VAL_IMAGES, VAL_LABELS,
        TEST_IMAGES, TEST_LABELS
    ])

    train_imgs = list_images(TRAIN_IMAGES)
    val_imgs = list_images(VAL_IMAGES)
    test_imgs = list_images(TEST_IMAGES)

    if not train_imgs:
        raise FileNotFoundError(f"No training images found in: {TRAIN_IMAGES}")
    if not val_imgs:
        raise FileNotFoundError(f"No validation images found in: {VAL_IMAGES}")
    if not test_imgs:
        raise FileNotFoundError(f"No test images found in: {TEST_IMAGES}")

    task = detect_dataset_task([TRAIN_LABELS, VAL_LABELS, TEST_LABELS])

    # Force all labels to single class 0: Waled
    remap_all_labels_to_single_class(TRAIN_LABELS)
    remap_all_labels_to_single_class(VAL_LABELS)
    remap_all_labels_to_single_class(TEST_LABELS)

    write_dataset_yaml()

    if task == "segment":
        model_path = SEG_MODEL
        run_name = "waled_seg"
    else:
        model_path = DETECT_MODEL
        run_name = "waled_detect"

    save_task_info(task, run_name)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"[INFO] Dataset task : {task}")
    print(f"[INFO] Model        : {model_path}")
    print(f"[INFO] Train images : {len(train_imgs)}")
    print(f"[INFO] Val images   : {len(val_imgs)}")
    print(f"[INFO] Test images  : {len(test_imgs)}")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] dataset.yaml : {DATA_YAML}")
    print("=" * 60)

    model = YOLO(model_path)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        single_cls=True,
        cache=True,
        amp=True,
        device=device,
        workers=4,
        patience=15,
        close_mosaic=10,
        seed=42,
        deterministic=True,
        verbose=True,
        plots=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"

    print("\n[DONE] Training finished.")
    print(f"[DONE] Best weights saved to: {best_weights}")


if __name__ == "__main__":
    main()