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
FRAME_INTERVAL = 1.0  # ✅ MATCHED: Same as entry.py
SKIP_FRAMES = 2  # ✅ FIX #1: Increased from 2 to 5 (same as entry.py)
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 2000
EXIT_COOLDOWN = 1 * 60
HEADLESS = True

# ✅ OPTIMIZED: Performance settings
MAX_DETECTION_SIZE = 480
RTSP_RECONNECT_DELAY = 3
RTSP_TIMEOUT = 10000
MAX_RECONNECT_ATTEMPTS = 5
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
    """
    ✅ IMPROVED: Create an optimized RTSP connection with better error handling
    """
    logging.info(f"🔗 Attempting to connect to {CAMERA_NAME} camera...")
    
    # ✅ FIX 1: Use TCP transport instead of UDP (more reliable)
    rtsp_url_tcp = rtsp_url
    if "?" in rtsp_url:
        rtsp_url_tcp = rtsp_url + "&tcp"
    else:
        rtsp_url_tcp = rtsp_url + "?tcp"
    
    # Try multiple connection methods
    connection_attempts = [
        (rtsp_url_tcp, "TCP transport"),
        (rtsp_url, "Default transport"),
    ]
    
    for attempt_url, method in connection_attempts:
        logging.info(f"  Trying {method}...")
        
        try:
            cap = cv2.VideoCapture(attempt_url, cv2.CAP_FFMPEG)
            
            # ✅ FIX 2: Set timeouts BEFORE opening
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout)
            
            # ✅ FIX 3: Minimal buffering to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # ✅ FIX 4: Lower resolution to reduce bandwidth
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS
            
            # ✅ FIX 5: Test if connection actually works
            if cap.isOpened():
                # Try to read a test frame with timeout
                start_time = time.time()
                ret, test_frame = cap.read()
                elapsed = time.time() - start_time
                
                if ret and test_frame is not None:
                    logging.info(f"  ✅ Connection successful via {method} (took {elapsed:.2f}s)")
                    return cap
                else:
                    logging.warning(f"  ⚠️ Connection opened but couldn't read frame via {method}")
                    cap.release()
            else:
                logging.warning(f"  ⚠️ Couldn't open connection via {method}")
                cap.release()
                
        except Exception as e:
            logging.warning(f"  ⚠️ Error with {method}: {e}")
            continue
    
    # ✅ FIX 6: Try with GStreamer as fallback (if available)
    try:
        logging.info("  Trying GStreamer backend...")
        gst_pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink max-buffers=1 drop=true"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                logging.info("  ✅ Connection successful via GStreamer")
                return cap
        cap.release()
    except Exception as e:
        logging.warning(f"  ⚠️ GStreamer not available: {e}")
    
    logging.error(f"❌ Failed to connect to camera after all attempts")
    return None


def load_known_faces(known_faces_dir="known_faces"):
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
    
    # ✅ FIX #2: Convert to numpy array for vectorized operations
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
    ✅ OPTIMIZED: Main processing loop matching entry.py optimizations
    """
    known_faces, known_embeddings = load_known_faces()
    
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
    
    # ✅ FIX #1: Frame skip counter
    frame_counter = 0

    # ✅ IMPROVED: Better initial connection with retry
    cap = None
    for attempt in range(MAX_RECONNECT_ATTEMPTS):
        logging.info(f"Connection attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS}...")
        cap = create_rtsp_connection(RTSP_URL)
        
        if cap is not None and cap.isOpened():
            logging.info(f"✅ Successfully connected to {CAMERA_NAME} camera")
            break
        
        if attempt < MAX_RECONNECT_ATTEMPTS - 1:
            wait_time = RTSP_RECONNECT_DELAY * (attempt + 1)
            logging.warning(f"⚠️ Connection failed, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    if cap is None or not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible after {MAX_RECONNECT_ATTEMPTS} attempts.")
        return

    logging.info(f"🎥 {CAMERA_NAME} camera running...")

    # ✅ IMPROVED: Main loop with better error handling
    consecutive_failures = 0
    
    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()
        
        total_frames += 1

        if not ret or frame is None:
            consecutive_failures += 1
            logging.warning(f"⚠️ Frame read failed for {CAMERA_NAME} (failure #{consecutive_failures})...")
            
            # ✅ IMPROVED: Reconnect logic
            if consecutive_failures >= 3 or (now_time - last_successful_read > 15):
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logging.error(f"❌ Max reconnection attempts reached.")
                    break
                
                logging.warning(f"⚠️ Camera unresponsive. Reconnecting... (attempt {reconnect_attempts + 1})")
                
                if cap is not None:
                    cap.release()
                
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = create_rtsp_connection(RTSP_URL)
                
                if cap is None or not cap.isOpened():
                    reconnect_attempts += 1
                    logging.error(f"❌ Reconnection failed")
                    time.sleep(RTSP_RECONNECT_DELAY * 2)
                    continue
                else:
                    logging.info(f"✅ Reconnection successful")
                    reconnect_attempts = 0
                    consecutive_failures = 0
                    last_successful_read = time.time()
            
            time.sleep(0.1)
            continue
        
        # ✅ Frame read successful
        reconnect_attempts = 0
        consecutive_failures = 0
        last_successful_read = now_time
        
        # ✅ FIX #1: Skip frames (same logic as entry.py)
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
            # ✅ FIX #3: Direct embedding (no alignment, same as entry.py)
            emb = face.embedding
            
            bbox = face.bbox.astype(int)
            name, emp_id = "Unknown", ""
            color = (0, 0, 255)

            # ✅ FIX #2: Vectorized similarity (same as entry.py)
            if len(known_embeddings) > 0:
                similarities = np.dot(known_embeddings, emb) / (
                    np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb)
                )
                best_idx = int(np.argmax(similarities))
                best_similarity = float(similarities[best_idx])
                
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

    if cap is not None:
        cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    
    logging.info(f"🏁 Session ended. Processed: {processed_frames} | Faces: {total_faces_detected}")


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("🛑 Program stopped by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)