from pathlib import Path
import time
import threading

import cv2
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
MODEL_DIR = Path("/home/pi/waled/best_ncnn_model")
SOURCE = 0
WINDOW_NAME = "Waled Pi 5 NCNN Max FPS"

# Lighter camera settings
CAM_WIDTH = 480
CAM_HEIGHT = 270
CAM_FPS = 30

# Lighter inference settings
IMAGE_SIZE = 320
CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.50
MAX_DET = 1

# 1 = infer every frame
# 2 = infer every second frame
# 3 = infer every third frame
INFER_EVERY_N_FRAMES = 2

FPS_SMOOTHING = 0.90

# Optional display scaling
SHOW_WINDOW = True
DISPLAY_SCALE = 1.0   # 1.0 = original, 1.5 = bigger window

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


class CameraStream:
    """
    Background camera reader.
    Keeps only the newest frame.
    """

    def __init__(self, source=0, width=480, height=270, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed or self.frame is None:
            raise RuntimeError("Could not read initial frame from camera.")

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed or frame is None:
                continue

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def draw_light_box(img, box, label):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

    cv2.putText(
        img,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_info(img, ai_fps):
    cv2.putText(
        img,
        f"FPS: {ai_fps:.2f}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def run_inference(model, frame):
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
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, clss):
            name = result.names.get(int(cls_id), "Waled")
            detections.append((box, conf, name))

    return detections


def maybe_resize(img, scale):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"NCNN model folder not found: {MODEL_DIR}")

    model = YOLO(str(MODEL_DIR))

    stream = CameraStream(
        source=SOURCE,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        fps=CAM_FPS
    ).start()

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    ai_fps = 0.0
    prev_ai_time = time.perf_counter()

    frame_count = 0
    last_detections = []

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            frame_count += 1

            if frame_count % INFER_EVERY_N_FRAMES == 0:
                last_detections = run_inference(model, frame)

                now_ai = time.perf_counter()
                current_ai_fps = 1.0 / max(now_ai - prev_ai_time, 1e-6)
                prev_ai_time = now_ai
                ai_fps = FPS_SMOOTHING * ai_fps + (1.0 - FPS_SMOOTHING) * current_ai_fps if ai_fps > 0 else current_ai_fps

            display = frame.copy()

            if last_detections:
                for box, conf, name in last_detections:
                    draw_light_box(display, box, f"{name} {conf:.2f}")

            draw_info(display, ai_fps)

            if SHOW_WINDOW:
                display = maybe_resize(display, DISPLAY_SCALE)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        stream.stop()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()