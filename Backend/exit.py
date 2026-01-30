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
RTSP_URL = "rtsp://moogle:Admin_123@10.8.21.47:554/video/live?channel=1&subtype=0"
THRESHOLD = 0.5
FRAME_INTERVAL = 0.8
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 1500  # ✅ REDUCED from 3000 to catch smaller faces
EXIT_COOLDOWN = 1 * 60   # 1 minute between logs for same person
HEADLESS = True

# ✅ IMPROVED: RTSP Connection Settings
RTSP_RECONNECT_DELAY = 2      # Increased to 2 seconds
RTSP_TIMEOUT = 5000           # 5 seconds timeout
MAX_RECONNECT_ATTEMPTS = 5    
FRAME_SKIP_ON_ERROR = True
 
# === INITIALIZE ===
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Suppress FFmpeg warnings
 
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


def create_rtsp_connection(rtsp_url, timeout=RTSP_TIMEOUT):
    """Create an optimized RTSP connection with error handling"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # ✅ CRITICAL: Set buffer to 1 to always get latest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set reasonable timeouts
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout)
    
    return cap
 
 
def load_known_faces(known_faces_dir="known_faces", use_alignment=True):
    """
    ✅ FIX #1: Load known faces with CONSISTENT embedding extraction
    
    Args:
        use_alignment: If True, use aligned faces for embeddings (RECOMMENDED)
    """
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
                face = faces[0]
                
                # ✅ CRITICAL FIX: Use aligned embedding if requested
                if use_alignment:
                    try:
                        aligned = app.face_align(img, face.landmark_2d_106)
                        aligned_faces = app.get(aligned)
                        if aligned_faces:
                            embedding = aligned_faces[0].embedding
                        else:
                            embedding = face.embedding
                    except:
                        embedding = face.embedding
                else:
                    embedding = face.embedding
                
                known_faces[len(known_embeddings)] = (name, emp_id)
                known_embeddings.append(embedding)
 
    logging.info(f"✅ Loaded {len(known_embeddings)} known embeddings.")
    return known_faces, known_embeddings
 
 
def save_hourly_index(unknown_dir, now, photo_name):
    """Append file info into an hourly index"""
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
    """
    ✅ FIX #2: Corrected unknown face cooldown logic
    """
    emb = face.embedding
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
 
    # Skip very small faces
    if face_area < UNKNOWN_FACE_MIN_AREA:
        return seen_unknown, unknown_cooldowns
 
    # Create unique key for this face
    emb_key = tuple(np.round(emb, 5))
    
    # ✅ CRITICAL FIX: Check cooldown FIRST, then update seen_unknown
    last_photo_time = unknown_cooldowns.get(emb_key, 0)
    
    if now_time - last_photo_time >= UNKNOWN_COOLDOWN:
        # Cooldown expired or first time seeing this face - take photo!
        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
        os.makedirs(unknown_dir, exist_ok=True)
 
        milliseconds = int(now.microsecond / 1000)
        photo_name = f"{now.strftime('%H-%M-%S')}-{milliseconds:03d}.jpg"
        photo_path = os.path.join(unknown_dir, photo_name)
        cv2.imwrite(photo_path, frame)
 
        save_hourly_index(unknown_dir, now, photo_name)
 
        # ✅ Update cooldown timer
        unknown_cooldowns[emb_key] = now_time
        logging.info(f"📸 Unknown saved: {photo_name}")
    
    # Mark as seen (but this doesn't prevent future photos after cooldown)
    seen_unknown.add(emb_key)
 
    return seen_unknown, unknown_cooldowns
 

def align_face(frame, face):
    """
    ✅ FIX #3: Improved face alignment with better error handling
    """
    try:
        # Use 106-point landmarks for better alignment (if available)
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            aligned = app.face_align(frame, face.landmark_2d_106)
        elif hasattr(face, 'landmark_2d_5') and face.landmark_2d_5 is not None:
            aligned = app.face_align(frame, face.landmark_2d_5)
        else:
            return None
        return aligned
    except Exception as e:
        logging.debug(f"Face alignment failed: {e}")
        return None
 
 
def main_loop():
    # ✅ Load known faces with alignment matching live detection
    known_faces, known_embeddings = load_known_faces(use_alignment=True)
 
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
        logging.error(f"❌ {CAMERA_NAME} camera not accessible.")
        return
 
    logging.info(f"🎥 {CAMERA_NAME} camera running...")
 
    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()
 
        if not ret:
            logging.warning(f"⚠️ Frame read failed for {CAMERA_NAME}...")
            
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
        
        # ✅ Clean up expired cooldowns to prevent memory growth
        unknown_cooldowns = {
            k: v for k, v in unknown_cooldowns.items()
            if now_time - v < UNKNOWN_COOLDOWN * 2  # Keep some history
        }
 
        for face in faces:
            # ✅ FIX #4: Get aligned embedding consistently
            aligned_face = align_face(frame, face)
 
            if aligned_face is not None:
                aligned_faces = app.get(aligned_face)
                if aligned_faces:
                    emb = aligned_faces[0].embedding
                else:
                    emb = face.embedding
            else:
                emb = face.embedding
 
            bbox = face.bbox.astype(int)
            name, emp_id = "Unknown", ""
            color = (0, 0, 255)
 
            # ----- MATCH KNOWN FACES -----
            best_similarity = 0
            best_match_idx = -1
            
            for idx, known_emb in enumerate(known_embeddings):
                similarity = cosine_similarity(emb, known_emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx
            
            # ✅ Use best match if above threshold
            if best_similarity > (1 - THRESHOLD):
                name, emp_id = known_faces[best_match_idx]
                color = (0, 255, 0)
                logging.debug(f"Match: {name} (similarity: {best_similarity:.3f})")
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