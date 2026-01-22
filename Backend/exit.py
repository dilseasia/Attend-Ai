import cv2
import numpy as np
import os
import time
import logging
from datetime import datetime
from insightface.app import FaceAnalysis
from attendance_db_postgres import (
    init_db,
    init_summary_table,
    log_attendance,
)
from triger import trigger_notification

# === CONFIGURATION ===
CAMERA_NAME = "Exit"
RTSP_URL = "rtsp://moogle:Admin_123@10.8.21.47:554/video/live?channel=1&subtype=1"
THRESHOLD = 0.5
FRAME_INTERVAL = 0.8
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 3000
EXIT_COOLDOWN = 1 * 60   # 1 minutes between logs for same person
HEADLESS = True  # Run headless (no display window)

# === INITIALIZE ===
# Force TCP for RTSP to avoid packet loss
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Suppress FFMPEG warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("exit.log", mode="a")],
)

logging.info(f"Initializing {CAMERA_NAME} camera...")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ✅ Initialize PostgreSQL tables
init_db()
init_summary_table()


# === UTILS ===
def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_known_faces(known_faces_dir="known_faces"):
    """Load all known faces and embeddings into memory"""
    known_faces = {}
    known_embeddings = []

    logging.info("Loading known faces...")
    for folder in os.listdir(known_faces_dir):
        if "_" not in folder:
            continue
        name, emp_id = folder.split("_", 1)
        folder_path = os.path.join(known_faces_dir, folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                known_faces[len(known_embeddings)] = (name, emp_id)
                known_embeddings.append(faces[0].embedding)

    logging.info(f"✅ Loaded {len(known_embeddings)} known embeddings.")
    return known_faces, known_embeddings


def save_hourly_index(unknown_dir, now, photo_name):
    """Append file info into an hourly index (for second layer use)"""
    hour_index_path = os.path.join(unknown_dir, f"hourly_index_{now.strftime('%H')}.txt")
    with open(hour_index_path, "a") as f:
        f.write(photo_name + "\n")


def process_known_face(name, emp_id, frame, now, now_time, seen_known, CAMERA_NAME):
    """Handle logic for recognized faces"""
    key = f"{name}_{emp_id}"
    last_seen = seen_known.get(key, 0)

    if now_time - last_seen >= EXIT_COOLDOWN:
        photo_path = os.path.join(
            "recognized_photos",
            now.strftime("%Y-%m-%d"),
            key,
            CAMERA_NAME,
            f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        )
        os.makedirs(os.path.dirname(photo_path), exist_ok=True)
        cv2.imwrite(photo_path, frame)

        # ✅ Log attendance
        log_attendance(
            name,
            emp_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            CAMERA_NAME
        )
        
        # 🔔 TRIGGER NOTIFICATION
        trigger_notification(
            name=name,
            emp_id=emp_id,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            camera="Exit",
            event="EXIT"
        )

        seen_known[key] = now_time
        logging.info(f"✅ {CAMERA_NAME} logged exit for {key}")
    return seen_known


def process_unknown_face(face, frame, now, now_time, seen_unknown, unknown_cooldowns):
    """Handle logic for unknown faces (anonymous)"""
    emb = face.embedding
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)

    if face_area < UNKNOWN_FACE_MIN_AREA:
        return seen_unknown, unknown_cooldowns

    emb_key = tuple(np.round(emb, 5))
    if emb_key in seen_unknown:
        return seen_unknown, unknown_cooldowns

    if now_time - unknown_cooldowns.get(emb_key, 0) >= UNKNOWN_COOLDOWN:
        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
        os.makedirs(unknown_dir, exist_ok=True)

        milliseconds = int(now.microsecond / 1000)
        photo_name = f"{now.strftime('%H-%M-%S')}-{milliseconds:03d}.jpg"
        photo_path = os.path.join(unknown_dir, photo_name)
        cv2.imwrite(photo_path, frame)

        # ✅ Add to hourly index
        save_hourly_index(unknown_dir, now, photo_name)

        unknown_cooldowns[emb_key] = now_time
        seen_unknown.add(emb_key)
        logging.info(f"📸 Unknown saved: {photo_name}")

    return seen_unknown, unknown_cooldowns


def align_face(frame, face):
    """Return aligned face image using InsightFace landmarks"""
    try:
        aligned = app.face_align(frame, face.landmark_2d_5)
        return aligned
    except:
        return None


def create_video_capture():
    """Create VideoCapture object with optimized settings"""
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
    cap.set(cv2.CAP_PROP_FPS, 15)  # Optimize frame rate
    return cap


def main_loop():
    known_faces, known_embeddings = load_known_faces()

    seen_known = {}
    seen_unknown = set()
    unknown_cooldowns = {}
    last_frame_time = time.time()
    reconnect_attempts = 0
    max_reconnect_attempts = 5

    cap = create_video_capture()
    
    if not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible.")
        return

    logging.info(f"🎥 {CAMERA_NAME} camera running...")

    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()

        if not ret:
            logging.warning(f"⚠️ Reconnecting {CAMERA_NAME}... (Attempt {reconnect_attempts + 1})")
            cap.release()
            time.sleep(2)
            
            cap = create_video_capture()
            reconnect_attempts += 1
            
            if reconnect_attempts >= max_reconnect_attempts:
                logging.error(f"❌ Failed to reconnect after {max_reconnect_attempts} attempts. Resetting counter.")
                reconnect_attempts = 0
                time.sleep(10)
            
            continue
        
        # Reset reconnect counter on successful frame
        if reconnect_attempts > 0:
            logging.info(f"✅ {CAMERA_NAME} reconnected successfully")
            reconnect_attempts = 0

        # Frame interval limiter
        if now_time - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now_time

        faces = app.get(frame)

        # Clean unknown cooldowns
        unknown_cooldowns = {
            k: v for k, v in unknown_cooldowns.items()
            if now_time - v < UNKNOWN_COOLDOWN
        }

        for face in faces:

            # ---------------------------------------------------
            #               FACE ALIGNMENT + EMBEDDING
            # ---------------------------------------------------
            aligned_face = align_face(frame, face)

            if aligned_face is not None:
                aligned_faces = app.get(aligned_face)
                if aligned_faces:
                    emb = aligned_faces[0].embedding
                else:
                    emb = face.embedding
            else:
                emb = face.embedding
            # ---------------------------------------------------

            bbox = face.bbox.astype(int)
            name, emp_id = "Unknown", ""
            color = (0, 0, 255)

            # ----- MATCH KNOWN FACES -----
            for idx, known_emb in enumerate(known_embeddings):
                if cosine_similarity(emb, known_emb) > (1 - THRESHOLD):
                    name, emp_id = known_faces[idx]
                    color = (0, 255, 0)
                    break
            # ------------------------------

            x1, y1, x2, y2 = bbox
            label = f"{name} ({emp_id})" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if name != "Unknown":
                seen_known = process_known_face(
                    name, emp_id, frame, now, now_time,
                    seen_known, CAMERA_NAME
                )
            else:
                seen_unknown, unknown_cooldowns = process_unknown_face(
                    face, frame, now, now_time,
                    seen_unknown, unknown_cooldowns
                )

        # Overlay info
        cv2.putText(frame, f"{CAMERA_NAME} Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not HEADLESS:
            cv2.imshow(CAMERA_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info(f"🛑 {CAMERA_NAME} camera stopped by user")
    except Exception as e:
        logging.error(f"❌ Fatal error in {CAMERA_NAME}: {e}", exc_info=True)