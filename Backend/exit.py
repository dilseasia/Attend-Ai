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
FRAME_INTERVAL = 0.9
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 2000
EXIT_COOLDOWN = 1 * 60
HEADLESS = True

# ✅ OPTIMIZED: Performance settings
MAX_DETECTION_SIZE = 480
SKIP_FRAMES = 2  # Process every 3rd frame
RTSP_RECONNECT_DELAY = 2
RTSP_TIMEOUT = 5000
MAX_RECONNECT_ATTEMPTS = 5
FRAME_SKIP_ON_ERROR = True
CLEANUP_INTERVAL = 300

# === INITIALIZE ===
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("exit.log", mode="a")],
)

logging.info(f"Initializing {CAMERA_NAME} camera...")

# ✅ CRITICAL: Smaller detection size
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(MAX_DETECTION_SIZE, MAX_DETECTION_SIZE))

init_db()
init_summary_table()

# === UTILS ===
def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_rtsp_connection(rtsp_url, timeout=RTSP_TIMEOUT):
    """Create an optimized RTSP connection"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # ✅ CRITICAL: Minimal buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout)
    
    # ✅ Reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap


def load_known_faces(known_faces_dir="known_faces", use_alignment=True):
    """Load known faces with embeddings"""
    known_faces = {}
    known_embeddings = []

    logging.info("Loading known faces...")
    
    if not os.path.exists(known_faces_dir):
        logging.error(f"❌ Directory not found: {known_faces_dir}")
        return known_faces, known_embeddings
    
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
        
        person_count = 0
        
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            try:
                faces = app.get(img)
                if faces:
                    embedding = faces[0].embedding
                    known_faces[len(known_embeddings)] = (name, emp_id)
                    known_embeddings.append(embedding)
                    person_count += 1
            except Exception as e:
                logging.warning(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        if person_count > 0:
            logging.info(f"  ✅ Loaded {person_count} embeddings for {name} ({emp_id})")

    logging.info(f"✅ Total loaded: {len(known_embeddings)} embeddings")
    
    # ✅ CRITICAL: Convert to numpy array
    if known_embeddings:
        known_embeddings = np.array(known_embeddings, dtype=np.float32)
    
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

        log_attendance(
            name,
            emp_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            CAMERA_NAME
        )
        
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
    """Handle unknown faces"""
    emb = face.embedding
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)

    if face_area < UNKNOWN_FACE_MIN_AREA:
        return seen_unknown, unknown_cooldowns

    # ✅ Reduced key size
    emb_key = tuple(np.round(emb[:50], 3))
    
    last_photo_time = unknown_cooldowns.get(emb_key, 0)
    
    if now_time - last_photo_time >= UNKNOWN_COOLDOWN:
        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
        os.makedirs(unknown_dir, exist_ok=True)

        milliseconds = int(now.microsecond / 1000)
        photo_name = f"{now.strftime('%H-%M-%S')}-{milliseconds:03d}.jpg"
        photo_path = os.path.join(unknown_dir, photo_name)
        cv2.imwrite(photo_path, frame)

        save_hourly_index(unknown_dir, now, photo_name)
        unknown_cooldowns[emb_key] = now_time
        logging.info(f"📸 Unknown saved: {photo_name}")
    
    seen_unknown.add(emb_key)
    return seen_unknown, unknown_cooldowns


def cleanup_old_cooldowns(cooldowns, current_time, max_age):
    """Remove old entries from cooldown dict"""
    return {k: v for k, v in cooldowns.items() if current_time - v < max_age}


def main_loop():
    """
    ✅ OPTIMIZED: Main processing loop
    """
    known_faces, known_embeddings = load_known_faces(use_alignment=True)
    
    if len(known_embeddings) == 0:
        logging.error("❌ No known faces loaded!")
        return

    seen_known = {}
    seen_unknown = set()
    unknown_cooldowns = {}
    last_frame_time = time.time()
    last_cleanup_time = time.time()
    
    reconnect_attempts = 0
    last_successful_read = time.time()
    
    # Statistics
    total_frames = 0
    processed_frames = 0
    total_faces_detected = 0
    total_known_matches = 0
    
    # ✅ Frame skip counter
    frame_counter = 0

    cap = create_rtsp_connection(RTSP_URL)
    
    if not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible.")
        return

    logging.info(f"🎥 {CAMERA_NAME} camera running...")

    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()
        
        total_frames += 1

        if not ret:
            logging.warning(f"⚠️ Frame read failed for {CAMERA_NAME}...")
            
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logging.error(f"❌ Max reconnection attempts reached.")
                break
            
            if now_time - last_successful_read > 30:
                logging.warning(f"⚠️ Camera unresponsive. Reconnecting...")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = create_rtsp_connection(RTSP_URL)
                reconnect_attempts += 1
                last_successful_read = time.time()
            
            if FRAME_SKIP_ON_ERROR:
                time.sleep(0.1)
                continue
            else:
                time.sleep(RTSP_RECONNECT_DELAY)
                cap.release()
                cap = create_rtsp_connection(RTSP_URL)
                reconnect_attempts += 1
                continue
        
        reconnect_attempts = 0
        last_successful_read = now_time
        
        # ✅ CRITICAL: Skip frames
        frame_counter += 1
        if frame_counter % (SKIP_FRAMES + 1) != 0:
            continue

        # Frame interval limiter
        if now_time - last_frame_time < FRAME_INTERVAL:
            continue
        
        last_frame_time = now_time
        processed_frames += 1
        
        # ✅ Periodic cleanup
        if now_time - last_cleanup_time > CLEANUP_INTERVAL:
            unknown_cooldowns = cleanup_old_cooldowns(
                unknown_cooldowns, now_time, UNKNOWN_COOLDOWN * 3
            )
            seen_unknown.clear()
            last_cleanup_time = now_time
            logging.info(f"🧹 Cleaned up cooldowns")

        # Detect faces
        try:
            faces = app.get(frame)
        except Exception as e:
            logging.error(f"❌ Face detection error: {e}")
            continue
        
        if faces:
            total_faces_detected += len(faces)

        for face in faces:
            emb = face.embedding
            bbox = face.bbox.astype(int)
            name, emp_id = "Unknown", ""
            color = (0, 0, 255)

            # ✅ CRITICAL: Vectorized similarity
            if len(known_embeddings) > 0:
                similarities = np.dot(known_embeddings, emb) / (
                    np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb)
                )
                best_idx = int(np.argmax(similarities))
                best_similarity = float(similarities[best_idx])
            else:
                best_similarity = 0
                best_idx = -1
            
            if best_similarity > (1 - THRESHOLD):
                name, emp_id = known_faces[best_idx]
                color = (0, 255, 0)
                total_known_matches += 1
                logging.debug(f"Match: {name} (similarity: {best_similarity:.3f})")
            
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
        
        # Statistics
        if processed_frames % 100 == 0:
            match_rate = (total_known_matches / total_faces_detected * 100) if total_faces_detected > 0 else 0
            logging.info(f"📊 Processed: {processed_frames} | Faces: {total_faces_detected} | Matches: {total_known_matches} ({match_rate:.1f}%)")

        if not HEADLESS:
            cv2.imshow(CAMERA_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    
    logging.info(f"🏁 Session ended. Processed: {processed_frames} | Faces: {total_faces_detected}")


if __name__ == "__main__":
    main_loop()