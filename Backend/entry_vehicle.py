# entry_vehicle.py
import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime
from ultralytics import YOLO

# ---------- CONFIG ----------
CAMERA_NAME = "Entry"
RTSP_URL = "rtsp://admin:admin123@10.8.21.48:554/cam/realmonitor?channel=1&subtype=0"

FRAME_INTERVAL = 1                  # seconds between processed frames
MOTION_THRESHOLD = 4000             # pixels of change to qualify as "moving"
OBJECT_COOLDOWN = 30                # seconds before re-saving same object
HEADLESS = True                     # True = no video window
SAVE_OBJECTS = True
DEBUG = False

# ---------- LOGGING ----------
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("entry_vehicle.log", mode="a")]
)

def safe_makedirs(path):
    """Create directories safely, avoiding file conflicts."""
    try:
        if os.path.isfile(path):
            os.remove(path)
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logging.warning(f"⚠️ Could not create folder {path}: {e} - entry_vehicle.py:36")

def bbox_iou(box1, box2):
    """Compute IoU (Intersection over Union) for cooldown tracking."""
    x1,y1,x2,y2 = box1
    x1b,y1b,x2b,y2b = box2
    inter_x1, inter_y1 = max(x1,x1b), max(y1,y1b)
    inter_x2, inter_y2 = min(x2,x2b), min(y2,y2b)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = max(0, x2-x1) * max(0, y2-y1)
    area2 = max(0, x2b-x1b) * max(0, y2b-y1b)
    denom = area1 + area2 - inter_area
    return inter_area / denom if denom > 0 else 0.0

# ---------- INIT YOLO ----------
logging.info("Initializing YOLOv8 for Entry vehicle detection... - entry_vehicle.py:52")
yolo = YOLO("yolov8n.pt")  # lightweight, fast model
logging.info("✅ YOLO model loaded successfully. - entry_vehicle.py:54")

# ---------- CAMERA ----------
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    logging.error(f"❌ Cannot open Entry camera stream: {RTSP_URL} - entry_vehicle.py:59")
    raise SystemExit(1)

# Motion detector
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Cooldown tracker
object_cooldowns = {}
last_frame_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        now = datetime.now()
        ts = time.time()

        if not ret:
            logging.warning("⚠️ Frame fetch failed; reconnecting Entry camera... - entry_vehicle.py:76")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        # Frame rate limit
        if ts - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = ts

        # Motion mask
        mask = fgbg.apply(frame)

        # Remove old cooldowns
        object_cooldowns = {k: v for k, v in object_cooldowns.items() if ts - v < OBJECT_COOLDOWN}

        # YOLO Detection
        try:
            for result in yolo(frame, stream=True):
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    label = yolo.names[int(cls)]
                    if label not in ["car", "truck", "bus", "motorbike"]:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    mask_crop = mask[y1:y2, x1:x2]
                    motion_pixels = int(np.sum(mask_crop) / 255) if mask_crop.size > 0 else 0

                    # Only capture moving vehicles
                    if motion_pixels < MOTION_THRESHOLD:
                        if DEBUG:
                            logging.debug(f"Skipping parked {label} (motion={motion_pixels}) - entry_vehicle.py:108")
                        continue

                    detected_box = (x1, y1, x2, y2)
                    # Avoid duplicates (IoU overlap)
                    if any(k[0] == label and bbox_iou(k[1], detected_box) > 0.5 for k in object_cooldowns):
                        continue

                    # Register cooldown
                    object_cooldowns[(label, detected_box)] = ts

                    # Save photo
                    if SAVE_OBJECTS:
                        save_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
                        safe_makedirs(save_dir)
                        file_name = f"{label}_{now.strftime('%H-%M-%S')}.jpg"
                        cv2.imwrite(os.path.join(save_dir, file_name), frame)
                        logging.info(f"🚗 Moving {label} saved to {save_dir}/{file_name} - entry_vehicle.py:125")

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(
                        frame, f"{label} {float(conf):.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )

        except Exception as e:
            logging.error(f"YOLO error: {e} - entry_vehicle.py:135")

        # Overlay info
        cv2.putText(frame, f"{CAMERA_NAME} Vehicle Detection", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Optional Display
        if not HEADLESS:
            cv2.imshow("Entry-Vehicles", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
