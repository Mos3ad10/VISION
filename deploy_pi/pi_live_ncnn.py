from pathlib import Path
import time

import cv2
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
MODEL_DIR = Path("/home/pi/waled/best_ncnn_model")  # change if needed
SOURCE = 0                                          # USB webcam / Pi camera index
WINDOW_NAME = "Waled Pi 5 NCNN Live"

IMAGE_SIZE = 416
CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.50
MAX_DET = 1
FPS_SMOOTHING = 0.90

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def draw_box(img, box, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    y = max(y1 - 10, th + 10)

    cv2.rectangle(img, (x1, y - th - 10), (x1 + tw + 12, y + baseline - 4), BOX_COLOR, -1)
    cv2.putText(img, label, (x1 + 6, y - 6), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def draw_info(img, fps):
    lines = [
        f"FPS: {fps:.2f}",
        f"imgsz: {IMAGE_SIZE}",
        "Runtime: NCNN",
        "Press Q or ESC to quit",
    ]

    x, y = 20, 35
    line_h = 28

    for i, text in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"NCNN model folder not found: {MODEL_DIR}")

    model = YOLO(str(MODEL_DIR))

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {SOURCE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps = 0.0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to read frame from camera.")
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[0],
            max_det=MAX_DET,
            agnostic_nms=True,
            verbose=False,
        )

        result = results[0]
        display = frame.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                name = result.names.get(int(cls_id), "Waled")
                draw_box(display, box, f"{name} {conf:.2f}")
        else:
            cv2.putText(
                display,
                "No Waled detected",
                (20, display.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        now = time.perf_counter()
        current_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps = FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * current_fps if fps > 0 else current_fps

        draw_info(display, fps)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()