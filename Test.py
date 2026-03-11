from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================
# Paths
# =========================================================
TEST_IMAGES = Path(r"D:\DATASET\test\images")

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_INFO_JSON = SCRIPT_DIR / "task_info.json"
RUNS_DIR = SCRIPT_DIR / "runs"

# =========================================================
# Display settings
# =========================================================
WINDOW_NAME = "Waled Inference"
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.50
MAX_DET = 1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Brighter and more visible styling
MASK_COLOR = (0, 255, 0)      # bright green (BGR)
BOX_COLOR = (0, 255, 0)       # bright green (BGR)
TEXT_COLOR = (255, 255, 255)  # white
ALPHA = 0.45                  # stronger mask visibility


def load_task_info():
    if not TASK_INFO_JSON.exists():
        raise FileNotFoundError("task_info.json not found. Run Train.py first.")
    return json.loads(TASK_INFO_JSON.read_text(encoding="utf-8"))


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def resize_for_display(image, max_width=1280, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def draw_corner_box(image, box, color, thickness=4, corner_len=32):
    x1, y1, x2, y2 = map(int, box)

    # Top-left
    cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)

    # Top-right
    cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)

    # Bottom-left
    cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)

    # Bottom-right
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)


def draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(y, th + 14)

    # Green label background
    cv2.rectangle(image, (x, y - th - 14), (x + tw + 16, y + baseline - 6), color, -1)

    # Black outline for better visibility
    cv2.putText(image, text, (x + 8, y - 8), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    # White main text
    cv2.putText(image, text, (x + 8, y - 8), font, font_scale, TEXT_COLOR, 2, cv2.LINE_AA)


def overlay_polygon_mask(image, polygon, color, alpha=0.45):
    if polygon is None or len(polygon) < 3:
        return

    poly = polygon.astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Thicker outline around mask
    cv2.polylines(image, [poly], True, color, 4, cv2.LINE_AA)


def main():
    info = load_task_info()
    task = info["task"]
    run_name = info["run_name"]

    best_weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"best.pt not found: {best_weights}")

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(f"Test images folder not found: {TEST_IMAGES}")

    image_paths = list_images(TEST_IMAGES)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {TEST_IMAGES}")

    model = YOLO(str(best_weights))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    predict_kwargs = dict(
        source=[str(p) for p in image_paths],
        stream=True,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],
        max_det=MAX_DET,
        agnostic_nms=True,
        verbose=False,
    )

    if task == "segment":
        predict_kwargs["retina_masks"] = True

    results = model.predict(**predict_kwargs)

    for result in results:
        frame = result.orig_img.copy()
        image_name = Path(result.path).name

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            if task == "segment" and result.masks is not None:
                polygons = result.masks.xy
            else:
                polygons = [None] * len(boxes)

            order = np.argsort(-confs)
            boxes = boxes[order]
            confs = confs[order]
            clss = clss[order]
            polygons = [polygons[i] if i < len(polygons) else None for i in order]

            for box, conf, cls_id, polygon in zip(boxes, confs, clss, polygons):
                if task == "segment":
                    overlay_polygon_mask(frame, polygon, MASK_COLOR, ALPHA)

                draw_corner_box(frame, box, BOX_COLOR, thickness=4, corner_len=32)

                label_name = result.names.get(int(cls_id), "Waled")
                draw_label(frame, f"{label_name} {conf:.2f}", int(box[0]), int(box[1]) - 6, BOX_COLOR)
        else:
            cv2.putText(
                frame,
                "No Waled detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Image: {image_name} | Task: {task}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        display = resize_for_display(frame)
        cv2.imshow(WINDOW_NAME, display)

        # Any key = next image
        # ESC or q = quit
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()