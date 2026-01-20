# exit_vehicle.py
import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime
from ultralytics import YOLO

# ---------- CONFIG ----------
CAMERA_NAME = "Exit"
RTSP_URL = "rtsp://admin:admin123@10.8.21.47:554/cam/realmonitor?channel=1&subtype=0"

FRAME_INTERVAL = 1
MOTION_THRESHOLD = 4000     # tune for your camera/resolution
OBJECT_COOLDOWN = 10
HEADLESS = True
SAVE_OBJECTS = True
DEBUG = False

# ---------- LOGGING ----------
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("exit_vehicle.log", mode="a")]
)

def safe_makedirs(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logging.warning(f"Could not create folder {path}: {e} - exit_vehicle.py:35")

def bbox_iou(box1, box2):
    x1,y1,x2,y2 = box1
    x1b,y1b,x2b,y2b = box2
    inter_x1, inter_y1 = max(x1,x1b), max(y1,y1b)
    inter_x2, inter_y2 = min(x2,x2b), min(y2,y2b)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area1 = max(0, x2-x1) * max(0, y2-y1)
    area2 = max(0, x2b-x1b) * max(0, y2b-y1b)
    denom = area1 + area2 - inter_area
    return inter_area/denom if denom>0 else 0.0

# ---------- INIT YOLO ----------
logging.info("Loading YOLO model for vehicles... - exit_vehicle.py:51")
yolo = YOLO("yolov8n.pt")  # ensure ultralytics installed
logging.info("YOLO loaded. - exit_vehicle.py:53")

# ---------- CAMERA ----------
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    logging.error("Cannot open Exit camera stream. - exit_vehicle.py:58")
    raise SystemExit(1)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
object_cooldowns = {}
last_frame_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        now = datetime.now()
        ts = time.time()
        if not ret:
            logging.warning("Frame fetch failed; reconnecting... - exit_vehicle.py:71")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        if ts - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = ts

        # motion mask
        mask = fgbg.apply(frame)

        # YOLO detections
        try:
            # prune old cooldowns
            object_cooldowns = {k:v for k,v in object_cooldowns.items() if ts - v < OBJECT_COOLDOWN}

            for res in yolo(frame, stream=True):
                for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    label = yolo.names[int(cls)]
                    if label not in ["car", "truck", "bus", "motorbike"]:
                        continue
                    x1,y1,x2,y2 = map(int, box)
                    mask_crop = mask[y1:y2, x1:x2]
                    motion_pixels = int(np.sum(mask_crop) / 255) if mask_crop.size>0 else 0
                    if motion_pixels < MOTION_THRESHOLD:
                        if DEBUG:
                            logging.debug(f"Skipping parked {label} motion={motion_pixels} - exit_vehicle.py:99")
                        continue
                    detected_box = (x1,y1,x2,y2)
                    # check cooldown by IoU
                    recent = any(k[0]==label and bbox_iou(k[1], detected_box) > 0.5 for k in object_cooldowns)
                    if recent:
                        continue
                    object_cooldowns[(label, detected_box)] = ts

                    # save snapshot
                    if SAVE_OBJECTS:
                        save_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
                        safe_makedirs(save_dir)
                        fname = f"{label}_{now.strftime('%H-%M-%S')}.jpg"
                        cv2.imwrite(os.path.join(save_dir, fname), frame)
                        logging.info(f"🚗 Moving {label} saved: {fname} - exit_vehicle.py:114")

                    # draw
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(255,255,0),2)
                    cv2.putText(frame, f"{label} {float(conf):.2f}", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        except Exception as e:
            logging.error(f"YOLO error: {e} - exit_vehicle.py:122")

        # overlay + display
        cv2.putText(frame, f"{CAMERA_NAME} Vehicles", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Exit-Vehicles", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
