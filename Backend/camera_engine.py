"""
camera_engine.py — Unified Batch Camera Processor
--------------------------------------------------
- Model sirf ONCE load hota hai (entry + exit dono ke liye)
- Dono cameras ke frames ek saath batch mein GPU pe jaate hain
- Threading se dono cameras simultaneously read hoti hain
- Resolution 640×480 fixed for both cameras
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
import os
from datetime import datetime, timedelta
from collections import deque
from insightface.app import FaceAnalysis
from attendance_db_postgres import init_db, init_summary_table, log_attendance
from triger import trigger_notification

# ============================================================
# CONFIGURATION
# ============================================================
HEADLESS    = True
THRESHOLD   = 0.5
FRAME_W     = 640
FRAME_H     = 480
TARGET_FPS  = 15

ENTRY_COOLDOWN       = 60        # seconds
EXIT_COOLDOWN        = 60
UNKNOWN_COOLDOWN     = 10
UNKNOWN_FACE_MIN_AREA = 2000
KNOWN_FACES_DIR      = "known_faces"
BATCH_COLLECT_MS     = 0.04      # 40ms — collect frames from all cameras before GPU call
MOTION_THRESHOLD     = 2000      # ✅ Higher threshold = less false triggers (was 800)
MOTION_PREV_FRAMES   = {}        # {camera_name: prev_gray_frame}
MAX_FACES_PER_FRAME  = 4         # ✅ Max faces to process per frame — limits CPU spike
FRAME_PROCESS_INTERVAL = 2.0     # ✅ Process 1 frame every 2 seconds per camera (was 0.5s)

CAMERAS = [
    {
        "name":     "Entry",
        "rtsp_url": "rtsp://admin:admin123@10.8.21.48:554/cam/realmonitor?channel=1&subtype=0",
        "type":     "entry",
        "cooldown": ENTRY_COOLDOWN,
    },
    {
        "name":     "Exit",
        "rtsp_url": "rtsp://moogle:Admin_123@10.8.21.47:554/video/live?channel=1&subtype=0",
        "type":     "exit",
        "cooldown": EXIT_COOLDOWN,
    },
]

# ============================================================
# LOGGING
# ============================================================
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("camera_engine.log", mode="a"),
    ],
)


# ============================================================
# SHARED MODEL — Loaded ONCE for all cameras
# ============================================================
logging.info("🔄 Loading face model (shared for all cameras)...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # GPU first, CPU fallback
)
face_app.prepare(ctx_id=0, det_size=(FRAME_W, FRAME_H))
logging.info("✅ Face model loaded (shared instance — saved RAM)")


# ============================================================
# LOAD KNOWN FACES
# ============================================================
def load_known_faces(known_faces_dir=KNOWN_FACES_DIR):
    known_faces      = {}
    known_embeddings = []

    logging.info("📂 Loading known faces...")
    if not os.path.exists(known_faces_dir):
        logging.error(f"❌ Directory not found: {known_faces_dir}")
        return known_faces, np.array([])

    for folder in os.listdir(known_faces_dir):
        if "_" not in folder:
            continue
        try:
            name, emp_id = folder.split("_", 1)
        except ValueError:
            continue

        folder_path = os.path.join(known_faces_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        count = 0
        for file in os.listdir(folder_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder_path, file))
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                known_faces[len(known_embeddings)] = (name, emp_id)
                known_embeddings.append(faces[0].embedding)
                count += 1

        if count:
            logging.info(f"  ✅ Loaded {count} embeddings for {name} ({emp_id})")

    total = len(known_embeddings)
    logging.info(f"✅ Total: {total} embeddings loaded")
    return known_faces, (np.array(known_embeddings, dtype=np.float32) if known_embeddings else np.array([]))


# ============================================================
# CAMERA READER THREAD
# ============================================================
class CameraReader(threading.Thread):
    """
    Each camera runs in its own thread.
    Reads frames and puts them into a shared queue.
    """

    def __init__(self, camera_cfg, frame_queue):
        super().__init__(daemon=True, name=camera_cfg["name"])
        self.cfg         = camera_cfg
        self.frame_queue = frame_queue
        self.cap         = None
        self.running     = True
        self.frame_count = 0
        self.last_frame_time = 0

    def connect(self):
        logging.info(f"🔗 Connecting to {self.cfg['name']} camera...")
        cap = cv2.VideoCapture(self.cfg["rtsp_url"], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ✅ Resolution fix — both cameras 640×480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                logging.info(f"✅ {self.cfg['name']} camera connected")
                return cap
        cap.release()
        return None

    def run(self):
        interval = FRAME_PROCESS_INTERVAL  # ✅ 1 frame per 2 seconds
        self.last_frame_time = time.time() - interval  # process first frame immediately

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self.connect()
                if self.cap is None:
                    logging.warning(f"⚠️ {self.cfg['name']} reconnecting in 3s...")
                    time.sleep(3)
                    continue

            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                logging.warning(f"⚠️ {self.cfg['name']} frame failed, reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(1)
                continue

            # ✅ Throttle to ~2fps
            if now - self.last_frame_time < interval:
                continue
            self.last_frame_time = now

            # ✅ Skip if queue already has this camera's frame (don't pile up)
            try:
                self.frame_queue.put_nowait({
                    "frame":       frame,
                    "camera_name": self.cfg["name"],
                    "camera_type": self.cfg["type"],
                    "cooldown":    self.cfg["cooldown"],
                    "timestamp":   datetime.now(),
                })
            except queue.Full:
                pass  # Drop frame if queue full — better than pile up

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ============================================================
# FACE MATCHING
# ============================================================
def match_face(embedding, known_faces, known_embeddings):
    if len(known_embeddings) == 0:
        return "Unknown", "", 0.0

    sims     = np.dot(known_embeddings, embedding) / (
                   np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
               )
    best_idx  = int(np.argmax(sims))
    best_sim  = float(sims[best_idx])

    if best_sim > (1 - THRESHOLD):
        name, emp_id = known_faces[best_idx]
        return name, emp_id, best_sim

    return "Unknown", "", best_sim


def has_motion(frame, camera_name):
    """
    Lightweight motion check using frame differencing (CPU, ~0.5ms).
    Returns True only if significant motion detected.
    Skip InsightFace entirely if no motion — huge CPU saving!
    """
    global MOTION_PREV_FRAMES
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    prev = MOTION_PREV_FRAMES.get(camera_name)
    MOTION_PREV_FRAMES[camera_name] = gray

    if prev is None:
        return True  # First frame — process it

    diff         = cv2.absdiff(prev, gray)
    _, thresh    = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    motion_score = int(np.sum(thresh) / 255)

    return motion_score > MOTION_THRESHOLD


# ============================================================
# BATCH GPU PROCESSOR (Main Loop)
# ============================================================
def batch_processor(frame_queue, known_faces, known_embeddings):
    """
    Pulls frames from queue → batches them → one GPU call → logs results.
    This is the core batch processing loop.
    """

    # Per-camera state (cooldowns, seen faces)
    camera_state = {}
    for cam in CAMERAS:
        camera_state[cam["name"]] = {
            "seen_known":      {},   # {key: last_seen_time}
            "seen_unknown":    set(),
            "unknown_cooldowns": {},
        }

    logging.info("🚀 Batch GPU processor started")

    while True:
        # ── Collect frames from queue (wait BATCH_COLLECT_MS to gather all cameras) ──
        batch = []
        deadline = time.time() + BATCH_COLLECT_MS

        while time.time() < deadline:
            try:
                item = frame_queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                time.sleep(0.005)

        if not batch:
            time.sleep(0.01)
            continue

        # ── Batch GPU call — only frames WITH motion ──
        frames_with_motion = [(item, item["frame"]) for item in batch if has_motion(item["frame"], item["camera_name"])]

        skipped = len(batch) - len(frames_with_motion)
        if skipped > 0:
            logging.debug(f"⏭️ Skipped {skipped} frames (no motion)")

        if not frames_with_motion:
            time.sleep(0.01)
            continue

        try:
            batch_results = [face_app.get(f) for _, f in frames_with_motion]
        except Exception as e:
            logging.error(f"❌ Batch inference error: {e}")
            continue

        # ── Limit faces per frame to avoid CPU spike ──
        for item, faces in zip([i for i, _ in frames_with_motion], batch_results):
            # Sort by face size (bigger = closer = more important), take top N
            if len(faces) > MAX_FACES_PER_FRAME:
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
                    reverse=True
                )[:MAX_FACES_PER_FRAME]
            camera_name = item["camera_name"]
            camera_type = item["camera_type"]
            cooldown    = item["cooldown"]
            frame       = item["frame"]
            now         = item["timestamp"]
            now_time    = time.time()

            state = camera_state[camera_name]

            for face in faces:
                emb  = face.embedding
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                face_area = (x2 - x1) * (y2 - y1)

                name, emp_id, confidence = match_face(emb, known_faces, known_embeddings)

                if name != "Unknown":
                    # ── Known face — check cooldown ──
                    key       = f"{name}_{emp_id}"
                    last_seen = state["seen_known"].get(key, 0)

                    if now_time - last_seen >= cooldown:
                        # Save photo
                        photo_path = os.path.join(
                            "recognized_photos",
                            now.strftime("%Y-%m-%d"), key,
                            camera_name,
                            f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                        )
                        os.makedirs(os.path.dirname(photo_path), exist_ok=True)
                        cv2.imwrite(photo_path, frame)

                        # Log attendance
                        log_attendance(
                            name, emp_id,
                            now.strftime("%Y-%m-%d"),
                            now.strftime("%H:%M:%S"),
                            camera_name
                        )

                        # Trigger notification
                        trigger_notification(
                            name=name, emp_id=emp_id,
                            date=now.strftime("%Y-%m-%d"),
                            time=now.strftime("%H:%M:%S"),
                            camera=camera_name,
                            event=camera_type.upper()
                        )

                        state["seen_known"][key] = now_time
                        logging.info(f"✅ [{camera_name}] {camera_type.upper()} logged: {name} ({emp_id}) | conf: {confidence:.3f}")

                else:
                    # ── Unknown face ──
                    if face_area < UNKNOWN_FACE_MIN_AREA:
                        continue

                    emb_key   = tuple(np.round(emb[:50], 3))
                    last_time = state["unknown_cooldowns"].get(emb_key, 0)

                    if now_time - last_time >= UNKNOWN_COOLDOWN:
                        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), camera_name)
                        os.makedirs(unknown_dir, exist_ok=True)
                        ms         = int(now.microsecond / 1000)
                        photo_name = f"{now.strftime('%H-%M-%S')}-{ms:03d}.jpg"
                        cv2.imwrite(os.path.join(unknown_dir, photo_name), frame)
                        state["unknown_cooldowns"][emb_key] = now_time
                        logging.info(f"📸 [{camera_name}] Unknown saved: {photo_name}")

        # ── Periodic cleanup of unknown cooldowns ──
        for cam_name, state in camera_state.items():
            cutoff = time.time() - UNKNOWN_COOLDOWN * 3
            state["unknown_cooldowns"] = {
                k: v for k, v in state["unknown_cooldowns"].items() if v > cutoff
            }


# ============================================================
# MAIN
# ============================================================
def main():
    init_db()
    init_summary_table()

    known_faces, known_embeddings = load_known_faces()

    # Shared queue: all camera readers → one batch processor
    frame_queue = queue.Queue(maxsize=len(CAMERAS) * 3)

    # Start one thread per camera
    readers = []
    for cam_cfg in CAMERAS:
        reader = CameraReader(cam_cfg, frame_queue)
        reader.start()
        readers.append(reader)
        logging.info(f"📷 Camera thread started: {cam_cfg['name']}")

    logging.info(f"⚡ Batch processor running — {len(CAMERAS)} cameras, 1 shared model")

    try:
        batch_processor(frame_queue, known_faces, known_embeddings)
    except KeyboardInterrupt:
        logging.info("🛑 Stopping camera engine...")
        for r in readers:
            r.stop()


if __name__ == "__main__":
    main()
