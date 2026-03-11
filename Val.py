from pathlib import Path
import json
import csv

from ultralytics import YOLO

# =========================================================
# Paths
# =========================================================
DATASET_ROOT = Path(r"D:\DATASET")
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_YAML = SCRIPT_DIR / "dataset.yaml"
TASK_INFO_JSON = SCRIPT_DIR / "task_info.json"
RUNS_DIR = SCRIPT_DIR / "runs"

CLASS_ID = 0
CLASS_NAME = "Waled"
IMAGE_SIZE = 640


def write_dataset_yaml_if_missing():
    if DATA_YAML.exists():
        return

    yaml_text = f"""path: {DATASET_ROOT.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  {CLASS_ID}: {CLASS_NAME}
"""
    DATA_YAML.write_text(yaml_text, encoding="utf-8")


def load_task_info():
    if not TASK_INFO_JSON.exists():
        raise FileNotFoundError("task_info.json not found. Run Train.py first.")
    return json.loads(TASK_INFO_JSON.read_text(encoding="utf-8"))


def save_confusion_matrix_summary(summary, save_dir: Path):
    json_path = save_dir / "confusion_matrix_summary.json"
    csv_path = save_dir / "confusion_matrix_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if summary:
        headers = list(summary[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(summary)

    print(f"[INFO] Saved confusion summary JSON: {json_path}")
    print(f"[INFO] Saved confusion summary CSV : {csv_path}")


def main():
    write_dataset_yaml_if_missing()
    info = load_task_info()

    task = info["task"]
    run_name = info["run_name"]

    best_weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt not found: {best_weights}")

    model = YOLO(str(best_weights))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",              # final evaluation on test split
        imgsz=IMAGE_SIZE,
        conf=0.25,
        iou=0.60,
        plots=True,
        project=str(RUNS_DIR),
        name=f"{run_name}_test_eval",
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(metrics.save_dir)
    print(f"\n[INFO] Validation outputs saved in: {save_dir}")

    # Confusion Matrix
    cm = metrics.confusion_matrix
    try:
        cm.plot(save_dir=save_dir)
    except Exception as e:
        print(f"[WARN] Could not plot confusion matrix: {e}")

    try:
        summary = cm.summary(normalize=True, decimals=5)
    except TypeError:
        summary = cm.summary()

    if summary:
        save_confusion_matrix_summary(summary, save_dir)

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)
    print(f"Task         : {task}")
    print(f"Box mAP50-95 : {metrics.box.map:.6f}")
    print(f"Box mAP50    : {metrics.box.map50:.6f}")
    print(f"Box mAP75    : {metrics.box.map75:.6f}")

    if task == "segment" and hasattr(metrics, "seg") and metrics.seg is not None:
        print(f"Mask mAP50-95: {metrics.seg.map:.6f}")
        print(f"Mask mAP50   : {metrics.seg.map50:.6f}")
        print(f"Mask mAP75   : {metrics.seg.map75:.6f}")

    print("\nCONFUSION MATRIX:")
    print(cm.matrix)

    print("\n[DONE] Final test evaluation complete.")


if __name__ == "__main__":
    main()