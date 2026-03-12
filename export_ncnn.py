from pathlib import Path
import json

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_INFO_JSON = SCRIPT_DIR / "task_info.json"
RUNS_DIR = SCRIPT_DIR / "runs"

# Recommended first Pi size
IMAGE_SIZE = 416

def load_task_info():
    if not TASK_INFO_JSON.exists():
        raise FileNotFoundError("task_info.json not found. Run Train.py first.")
    return json.loads(TASK_INFO_JSON.read_text(encoding="utf-8"))

def main():
    info = load_task_info()
    run_name = info["run_name"]

    best_weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt not found: {best_weights}")

    print(f"[INFO] Exporting: {best_weights}")
    print(f"[INFO] imgsz: {IMAGE_SIZE}")

    model = YOLO(str(best_weights))

    exported_path = model.export(
        format="ncnn",
        imgsz=IMAGE_SIZE,
    )

    print(f"[DONE] NCNN export saved to: {exported_path}")

if __name__ == "__main__":
    main()