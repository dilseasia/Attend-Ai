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
from datetime import datetime
 
now = datetime.now()
 
# === CONFIGURATION ===
CAMERA_NAME = "Entry"
RTSP_URL = "rtsp://admin:admin123@10.8.21.48:554/cam/realmonitor?channel=1&subtype=0"
THRESHOLD = 0.5
FRAME_INTERVAL = 0.8
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 2500
ENTRY_COOLDOWN = 1 * 60   # 1 minutes between logs for same person
HEADLESS = True             # Run headless (no display window)

# ✅ NEW: RTSP Connection Settings
RTSP_RECONNECT_DELAY = 2      # Seconds to wait before reconnecting
RTSP_TIMEOUT = 10000          # 10 seconds timeout (in milliseconds)
MAX_RECONNECT_ATTEMPTS = 5    # Maximum consecutive reconnection attempts
FRAME_SKIP_ON_ERROR = True    # Skip processing if frame read fails
 
# === INITIALIZE ===
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"

# ✅ Suppress FFmpeg/RTSP warnings (optional)
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Suppress most warnings
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("entry.log", mode="a")],
)
 
logging.info(f"Initializing {CAMERA_NAME} camera... - entry.py:37")
 
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
 
 
# ✅ Initialize PostgreSQL tables
init_db()
init_summary_table()
 
 
# === UTILS ===
def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ✅ NEW: Create RTSP connection with optimized settings
def create_rtsp_connection(rtsp_url, timeout=RTSP_TIMEOUT):
    """Create an optimized RTSP connection with error handling"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Set buffer size to 1 to reduce latency and prevent old frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set timeout for read operations
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout)
    
    # Use TCP instead of UDP to reduce packet loss (more reliable)
    # Note: This requires modifying RTSP URL or camera settings
    # Alternative: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    
    return cap
 
 
def load_known_faces(known_faces_dir="known_faces"):
    """Load all known faces and embeddings into memory"""
    known_faces = {}
    known_embeddings = []
 
    logging.info("Loading known faces... - entry.py:59")
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
 
    logging.info(f"✅ Loaded {len(known_embeddings)} known embeddings. - entry.py:75")
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
 
    if now_time - last_seen >= ENTRY_COOLDOWN:
        photo_path = os.path.join(
            "recognized_photos",
            now.strftime("%Y-%m-%d"),
            key,
            CAMERA_NAME,
            f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        )
        os.makedirs(os.path.dirname(photo_path), exist_ok=True)
        cv2.imwrite(photo_path, frame)
 
        #  Log attendance
        log_attendance(
            name,
            emp_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            CAMERA_NAME
        )
        # ✅ Trigger notification
        from triger import trigger_notification
        trigger_notification(
            name=name,
            emp_id=emp_id,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            camera="Entry",
            event="ENTRY"
        )
 
        seen_known[key] = now_time
        logging.info(f"✅ {CAMERA_NAME} logged entry for {key} - entry.py:121")
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
        logging.info(f"📸 Unknown saved: {photo_name} - entry.py:153")
 
    return seen_unknown, unknown_cooldowns
 
def align_face(frame, face):
    """Return aligned face image using InsightFace landmarks"""
    try:
        aligned = app.face_align(frame, face.landmark_2d_5)
        return aligned
    except:
        return None
 
 
def main_loop():
    known_faces, known_embeddings = load_known_faces()
 
    seen_known = {}
    seen_unknown = set()
    unknown_cooldowns = {}
    last_frame_time = time.time()
    
    # ✅ NEW: Reconnection tracking
    reconnect_attempts = 0
    last_successful_read = time.time()

    # ✅ Use improved RTSP connection
    cap = create_rtsp_connection(RTSP_URL)
    
    if not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible. - entry.py:176")
        return
 
    logging.info(f"🎥 {CAMERA_NAME} camera running... - entry.py:179")
 
    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()
 
        if not ret:
            logging.warning(f"⚠️ Frame read failed for {CAMERA_NAME}... - entry.py:187")
            
            # ✅ NEW: Intelligent reconnection logic
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logging.error(f"❌ Max reconnection attempts reached for {CAMERA_NAME}. Exiting.")
                break
            
            # Check if camera has been unresponsive for too long
            if now_time - last_successful_read > 30:  # 30 seconds timeout
                logging.warning(f"⚠️ Camera unresponsive for 30s. Reconnecting {CAMERA_NAME}...")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = create_rtsp_connection(RTSP_URL)
                reconnect_attempts += 1
                last_successful_read = time.time()
            
            if FRAME_SKIP_ON_ERROR:
                time.sleep(0.1)  # Brief pause before next read attempt
                continue
            else:
                time.sleep(RTSP_RECONNECT_DELAY)
                cap.release()
                cap = create_rtsp_connection(RTSP_URL)
                reconnect_attempts += 1
                continue
        
        # ✅ Reset reconnection counter on successful read
        reconnect_attempts = 0
        last_successful_read = now_time
 
        # Frame interval limiter
        if now_time - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now_time
 
        faces = app.get(frame)
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
    main_loop()