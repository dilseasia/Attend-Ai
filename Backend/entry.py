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
CAMERA_NAME = "Entry"
RTSP_URL = "rtsp://admin:admin123@10.8.21.48:554/cam/realmonitor?channel=1&subtype=0"

# ✅ CRITICAL FIX: More lenient threshold for better recognition
THRESHOLD = 0.45  # Was 0.5 - lower = less strict, catches more matches

FRAME_INTERVAL = 0.8
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 1200  # ✅ Further reduced to catch distant faces
ENTRY_COOLDOWN = 0.5 * 60   # 30 seconds between logs for same person
HEADLESS = True

# ✅ FIXED: Proper RTSP timeout (was 50ms - too short!)
RTSP_RECONNECT_DELAY = 2
RTSP_TIMEOUT = 5000  # 5 seconds in milliseconds
MAX_RECONNECT_ATTEMPTS = 5    
FRAME_SKIP_ON_ERROR = True

# ✅ NEW: Quality thresholds
MIN_DETECTION_SCORE = 0.3  # Minimum face detection confidence (0-1)
USE_MULTIPLE_PHOTOS_PER_PERSON = True  # Load all photos per person for better matching
 
# === INITIALIZE ===
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("entry.log", mode="a")],
)
 
logging.info(f"Initializing {CAMERA_NAME} camera...")
 
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # ✅ Larger detection size for better accuracy
 
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
    
    # ✅ Try to set resolution (may help with face quality)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap
 
 
def load_known_faces(known_faces_dir="known_faces", use_alignment=False):
    """
    ✅ CRITICAL FIX: DO NOT use alignment during loading to match live detection better
    Load multiple photos per person for more robust matching
    """
    known_faces = {}
    known_embeddings = []
 
    logging.info("Loading known faces...")
    
    if not os.path.exists(known_faces_dir):
        logging.error(f"❌ Known faces directory not found: {known_faces_dir}")
        return known_faces, known_embeddings
    
    for folder in os.listdir(known_faces_dir):
        if "_" not in folder:
            continue
        
        try:
            name, emp_id = folder.split("_", 1)
        except ValueError:
            logging.warning(f"⚠️ Skipping invalid folder name: {folder}")
            continue
            
        folder_path = os.path.join(known_faces_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        person_embeddings = []
        
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            
            if img is None:
                logging.warning(f"⚠️ Could not read image: {img_path}")
                continue
            
            try:
                faces = app.get(img)
                if faces:
                    face = faces[0]
                    
                    # ✅ FIXED: Use ORIGINAL embedding without alignment
                    # This matches the live detection which also struggles with alignment
                    embedding = face.embedding
                    person_embeddings.append(embedding)
                    
                    # Store each embedding with person info
                    known_faces[len(known_embeddings)] = (name, emp_id)
                    known_embeddings.append(embedding)
                    
            except Exception as e:
                logging.warning(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        if person_embeddings:
            logging.info(f"  ✅ Loaded {len(person_embeddings)} embeddings for {name} ({emp_id})")
        else:
            logging.warning(f"  ⚠️ No valid embeddings found for {name} ({emp_id})")
 
    logging.info(f"✅ Total loaded: {len(known_embeddings)} embeddings from {len(set(known_faces.values()))} people")
    return known_faces, known_embeddings
 
 
def save_hourly_index(unknown_dir, now, photo_name):
    """Append file info into an hourly index"""
    hour_index_path = os.path.join(unknown_dir, f"hourly_index_{now.strftime('%H')}.txt")
    with open(hour_index_path, "a") as f:
        f.write(photo_name + "\n")
 
 
def process_known_face(name, emp_id, frame, now, now_time, seen_known, CAMERA_NAME, similarity):
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
            camera="Entry",
            event="ENTRY"
        )
 
        seen_known[key] = now_time
        logging.info(f"✅ {CAMERA_NAME} logged entry for {key} (confidence: {similarity:.3f})")
    else:
        # Already logged recently, but still detected
        logging.debug(f"🔄 {key} seen again (cooldown active, {now_time - last_seen:.0f}s elapsed)")
    
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
        logging.debug(f"⏭️ Skipping small face: {face_area} < {UNKNOWN_FACE_MIN_AREA}")
        return seen_unknown, unknown_cooldowns
 
    # Create unique key for this face
    emb_key = tuple(np.round(emb, 5))
    
    # ✅ CRITICAL FIX: Check cooldown FIRST
    last_photo_time = unknown_cooldowns.get(emb_key, 0)
    
    if now_time - last_photo_time >= UNKNOWN_COOLDOWN:
        # Cooldown expired or first time - take photo!
        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
        os.makedirs(unknown_dir, exist_ok=True)
 
        milliseconds = int(now.microsecond / 1000)
        photo_name = f"{now.strftime('%H-%M-%S')}-{milliseconds:03d}.jpg"
        photo_path = os.path.join(unknown_dir, photo_name)
        cv2.imwrite(photo_path, frame)
 
        save_hourly_index(unknown_dir, now, photo_name)
 
        # ✅ Update cooldown timer
        unknown_cooldowns[emb_key] = now_time
        logging.info(f"📸 Unknown saved: {photo_name} (area: {face_area})")
    
    # Mark as seen
    seen_unknown.add(emb_key)
 
    return seen_unknown, unknown_cooldowns
 

def get_face_embedding(frame, face, use_alignment=False):
    """
    ✅ NEW: Centralized embedding extraction with consistent method
    """
    if not use_alignment:
        # Use original detection embedding (more reliable for our case)
        return face.embedding
    
    # Try alignment (but often fails or gives inconsistent results)
    try:
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            aligned = app.face_align(frame, face.landmark_2d_106)
        elif hasattr(face, 'landmark_2d_5') and face.landmark_2d_5 is not None:
            aligned = app.face_align(frame, face.landmark_2d_5)
        else:
            return face.embedding
        
        aligned_faces = app.get(aligned)
        if aligned_faces:
            return aligned_faces[0].embedding
        else:
            return face.embedding
    except Exception as e:
        logging.debug(f"Alignment failed, using original: {e}")
        return face.embedding
 
 
def main_loop():
    # ✅ CRITICAL: Load without alignment to match live detection
    known_faces, known_embeddings = load_known_faces(use_alignment=False)
    
    if len(known_embeddings) == 0:
        logging.error("❌ No known faces loaded! Check your known_faces directory.")
        return
 
    seen_known = {}
    seen_unknown = set()
    unknown_cooldowns = {}
    last_frame_time = time.time()
    
    reconnect_attempts = 0
    last_successful_read = time.time()
    
    # ✅ Statistics
    total_frames = 0
    total_faces_detected = 0
    total_known_matches = 0

    cap = create_rtsp_connection(RTSP_URL)
    
    if not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible.")
        return
 
    logging.info(f"🎥 {CAMERA_NAME} camera running...")
    logging.info(f"📊 Configuration: THRESHOLD={THRESHOLD}, MIN_AREA={UNKNOWN_FACE_MIN_AREA}, COOLDOWN={ENTRY_COOLDOWN}s")
 
    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()
 
        if not ret:
            logging.warning(f"⚠️ Frame read failed for {CAMERA_NAME}...")
            
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logging.error(f"❌ Max reconnection attempts reached. Exiting.")
                break
            
            if now_time - last_successful_read > 30:
                logging.warning(f"⚠️ Camera unresponsive for 30s. Reconnecting...")
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
 
        # Frame interval limiter
        if now_time - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now_time
        
        total_frames += 1
 
        # ✅ Detect faces
        try:
            faces = app.get(frame)
        except Exception as e:
            logging.error(f"❌ Face detection error: {e}")
            continue
        
        if faces:
            total_faces_detected += len(faces)
        
        # ✅ Clean up expired cooldowns
        unknown_cooldowns = {
            k: v for k, v in unknown_cooldowns.items()
            if now_time - v < UNKNOWN_COOLDOWN * 2
        }
 
        for face in faces:
            # ✅ Check detection quality
            if hasattr(face, 'det_score') and face.det_score < MIN_DETECTION_SCORE:
                logging.debug(f"⏭️ Skipping low-quality detection: {face.det_score:.3f}")
                continue
            
            # ✅ Get embedding WITHOUT alignment (consistent with loading)
            emb = get_face_embedding(frame, face, use_alignment=False)
 
            bbox = face.bbox.astype(int)
            name, emp_id = "Unknown", ""
            color = (0, 0, 255)
 
            # ✅ IMPROVED MATCHING: Find best match across ALL embeddings
            best_similarity = 0
            best_match_idx = -1
            
            for idx, known_emb in enumerate(known_embeddings):
                similarity = cosine_similarity(emb, known_emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx
            
            # ✅ Log all similarities above a lower threshold for debugging
            if best_similarity > 0.3:
                logging.debug(f"🔍 Best match similarity: {best_similarity:.3f} (threshold: {1-THRESHOLD:.3f})")
            
            # ✅ Use best match if above threshold
            if best_similarity > (1 - THRESHOLD):
                name, emp_id = known_faces[best_match_idx]
                color = (0, 255, 0)
                total_known_matches += 1
                logging.info(f"✅ Recognized: {name} ({emp_id}) - confidence: {best_similarity:.3f}")
            else:
                logging.debug(f"❌ No match found. Best similarity: {best_similarity:.3f} < threshold: {1-THRESHOLD:.3f}")
 
            x1, y1, x2, y2 = bbox
            face_area = (x2 - x1) * (y2 - y1)
            
            # ✅ Enhanced label with confidence
            if name != "Unknown":
                label = f"{name} ({emp_id}) [{best_similarity:.2f}]"
            else:
                label = f"Unknown [{best_similarity:.2f}]"
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
            if name != "Unknown":
                seen_known = process_known_face(
                    name, emp_id, frame, now, now_time,
                    seen_known, CAMERA_NAME, best_similarity
                )
            else:
                seen_unknown, unknown_cooldowns = process_unknown_face(
                    face, frame, now, now_time,
                    seen_unknown, unknown_cooldowns
                )
 
        # ✅ Enhanced overlay info
        cv2.putText(frame, f"{CAMERA_NAME} Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ✅ Show statistics every 100 frames
        if total_frames % 100 == 0:
            match_rate = (total_known_matches / total_faces_detected * 100) if total_faces_detected > 0 else 0
            logging.info(f"📊 Stats: {total_frames} frames, {total_faces_detected} faces, {total_known_matches} matches ({match_rate:.1f}%)")
 
        if not HEADLESS:
            cv2.imshow(CAMERA_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    
    logging.info(f"🏁 Session ended. Total frames: {total_frames}, Faces: {total_faces_detected}, Matches: {total_known_matches}")
 
 
if __name__ == "__main__":
    main_loop()