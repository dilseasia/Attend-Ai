import cv2
import numpy as np
import os
import time
import torch
import torchreid
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
ENTRY_COOLDOWN = 1 * 60   # 1 minute between logs for same person
HEADLESS = True            # Run headless (no display window)
BODY_DETECTION_ENABLED = True  # Enable body detection fallback
BODY_MATCH_THRESHOLD = 0.75


# === INITIALIZE ===
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("entry.log", mode="a")],
)

logging.info(f"Initializing {CAMERA_NAME} camera... - entry.py")

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

device = "cpu"


# Initialize body recognition model (optional)
body_model = None
if BODY_DETECTION_ENABLED:
    try:
        body_model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )
        body_model.to(device)
        body_model.eval()
        logging.info("✅ Body recognition model loaded")
    except Exception as e:
        logging.error(f"❌ Failed to load body model: {e}")
        BODY_DETECTION_ENABLED = False

# Initialize PostgreSQL tables
init_db()
init_summary_table()


# === UTILS ===
def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_body_embedding(frame, bbox=None):
    """Extract body embedding from frame region"""
    if not BODY_DETECTION_ENABLED or body_model is None:
        return None
    
    try:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            body = frame[y1:y2, x1:x2]
        else:
            body = frame

        if body.size == 0:
            return None

        body = cv2.resize(body, (256, 128))
        body = body[:, :, ::-1] / 255.0
        body = torch.tensor(body).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            emb = body_model(body).cpu().numpy()[0]

        return emb
    except Exception as e:
        logging.debug(f"Body embedding extraction failed: {e}")
        return None


def load_known_faces_and_bodies(known_faces_dir="known_faces"):
    """Load all known faces and body embeddings from the same directory"""
    known_faces = {}
    known_face_embeddings = []
    known_body_embeddings = []

    if not os.path.exists(known_faces_dir):
        logging.warning(f"⚠️ Known faces directory not found: {known_faces_dir}")
        return known_faces, known_face_embeddings, known_body_embeddings

    logging.info("Loading known faces and bodies...")
    
    for folder in os.listdir(known_faces_dir):
        if "_" not in folder:
            continue
        try:
            name, emp_id = folder.split("_", 1)
        except ValueError:
            logging.warning(f"⚠️ Invalid folder format: {folder}")
            continue
            
        folder_path = os.path.join(known_faces_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Extract face embedding
            faces = app.get(img)
            if faces:
                face_emb = faces[0].embedding
                
                # Extract body embedding from the same image
                body_emb = None
                if BODY_DETECTION_ENABLED and body_model is not None:
                    # Get face bounding box
                    bbox = faces[0].bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Expand bbox to include more body area
                    h, w = img.shape[:2]
                    face_height = y2 - y1
                    face_width = x2 - x1
                    
                    # Expand downward and sideways to capture torso
                    expand_down = int(face_height * 3)  # 3x face height down
                    expand_side = int(face_width * 0.5)  # 0.5x face width on each side
                    
                    body_x1 = max(0, x1 - expand_side)
                    body_y1 = y1
                    body_x2 = min(w, x2 + expand_side)
                    body_y2 = min(h, y2 + expand_down)
                    
                    body_bbox = (body_x1, body_y1, body_x2, body_y2)
                    body_emb = extract_body_embedding(img, body_bbox)
                
                # Store both embeddings with the same index
                idx = len(known_face_embeddings)
                known_faces[idx] = (name, emp_id)
                known_face_embeddings.append(face_emb)
                
                if body_emb is not None:
                    known_body_embeddings.append(body_emb)
                else:
                    # Add None placeholder to keep indices aligned
                    known_body_embeddings.append(None)

    logging.info(f"✅ Loaded {len(known_face_embeddings)} known face embeddings")
    
    valid_body_count = sum(1 for emb in known_body_embeddings if emb is not None)
    if BODY_DETECTION_ENABLED:
        logging.info(f"✅ Loaded {valid_body_count} known body embeddings")
    
    return known_faces, known_face_embeddings, known_body_embeddings


def save_hourly_index(unknown_dir, now, photo_name):
    """Append file info into an hourly index (for second layer use)"""
    hour_index_path = os.path.join(unknown_dir, f"hourly_index_{now.strftime('%H')}.txt")
    try:
        with open(hour_index_path, "a") as f:
            f.write(photo_name + "\n")
    except Exception as e:
        logging.error(f"❌ Failed to write hourly index: {e}")


def match_body(emb, known_embeddings, threshold=None):
    """Match body embedding against known embeddings"""
    if not known_embeddings or emb is None:
        return None, 0
    
    if threshold is None:
        threshold = BODY_MATCH_THRESHOLD
        
    best_score = 0
    best_idx = None

    for i, known_emb in enumerate(known_embeddings):
        if known_emb is None:
            continue
        score = cosine_similarity(emb, known_emb)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score > threshold:
        return best_idx, best_score

    return None, best_score


def process_known_face(name, emp_id, frame, now, now_time, seen_known, camera_name, match_method="face"):
    """Handle logic for recognized faces or body"""
    key = f"{name}_{emp_id}"
    last_seen = seen_known.get(key, 0)

    if now_time - last_seen >= ENTRY_COOLDOWN:
        photo_path = os.path.join(
            "recognized_photos",
            now.strftime("%Y-%m-%d"),
            key,
            camera_name,
            f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        )
        os.makedirs(os.path.dirname(photo_path), exist_ok=True)
        cv2.imwrite(photo_path, frame)

        # Log attendance
        log_attendance(
            name,
            emp_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            camera_name
        )

        # 🔔 TRIGGER NOTIFICATION HERE
        trigger_notification(
            name=name,
            emp_id=emp_id,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            camera_name="Entry",
            event="ENTRY"
        )

        seen_known[key] = now_time

        # ✅ CLEAR LOG MESSAGE
        if match_method == "body":
            logging.info(f"🟡 LOGGED BY BODY DETECTION → {key}")
        else:
            logging.info(f"🟢 LOGGED BY FACE DETECTION → {key}")

    return seen_known


def process_unknown_face(face, frame, now, now_time, seen_unknown, unknown_cooldowns, camera_name):
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
        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), camera_name)
        os.makedirs(unknown_dir, exist_ok=True)

        milliseconds = int(now.microsecond / 1000)
        photo_name = f"{now.strftime('%H-%M-%S')}-{milliseconds:03d}.jpg"
        photo_path = os.path.join(unknown_dir, photo_name)
        cv2.imwrite(photo_path, frame)

        # Add to hourly index
        save_hourly_index(unknown_dir, now, photo_name)

        unknown_cooldowns[emb_key] = now_time
        seen_unknown.add(emb_key)
        logging.info(f"📸 Unknown saved: {photo_name}")

    return seen_unknown, unknown_cooldowns


def align_face_improved(frame, face):
    """
    Align face using affine transformation based on landmarks
    This is a custom implementation since app.face_align doesn't exist
    """
    try:
        if not hasattr(face, 'kps') or face.kps is None:
            return None
            
        # Get 5 keypoints from face
        kps = face.kps
        
        # Reference points for aligned face (standard positions)
        # Based on 112x112 output size
        ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Compute affine transformation matrix
        tform = cv2.estimateAffinePartial2D(kps, ref_pts)[0]
        
        if tform is None:
            return None
            
        # Apply transformation
        aligned = cv2.warpAffine(frame, tform, (112, 112))
        return aligned
        
    except Exception as e:
        logging.debug(f"Face alignment failed: {e}")
        return None


def main_loop():
    """Main processing loop"""
    known_faces, known_face_embeddings, known_body_embeddings = load_known_faces_and_bodies()

    seen_known = {}
    seen_unknown = set()
    unknown_cooldowns = {}
    last_frame_time = time.time()

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible")
        return

    logging.info(f"🎥 {CAMERA_NAME} camera running...")

    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()

        if not ret:
            logging.warning(f"⚠️ Reconnecting {CAMERA_NAME}...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        # Frame interval limiter
        if now_time - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now_time

        # Detect faces
        faces = app.get(frame)
        
        if faces:
            # Clean up old cooldowns
            unknown_cooldowns = {
                k: v for k, v in unknown_cooldowns.items()
                if now_time - v < UNKNOWN_COOLDOWN
            }

            for face in faces:
                # Try to align face for better recognition
                aligned_face = align_face_improved(frame, face)
                
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
                match_method = ""

                # Match against known face embeddings
                best_face_similarity = 0
                best_face_idx = None
                
                for idx, known_emb in enumerate(known_face_embeddings):
                    similarity = cosine_similarity(emb, known_emb)
                    if similarity > best_face_similarity:
                        best_face_similarity = similarity
                        best_face_idx = idx

                if best_face_similarity > (1 - THRESHOLD):
                    name, emp_id = known_faces[best_face_idx]
                    color = (0, 255, 0)
                    match_method = "face"
                else:
                    # Face match failed, try body matching if enabled
                    if BODY_DETECTION_ENABLED and body_model is not None:
                        x1, y1, x2, y2 = bbox
                        h, w = frame.shape[:2]
                        face_height = y2 - y1
                        face_width = x2 - x1
                        
                        # Expand bbox to include body
                        expand_down = int(face_height * 3)
                        expand_side = int(face_width * 0.5)
                        
                        body_x1 = max(0, x1 - expand_side)
                        body_y1 = y1
                        body_x2 = min(w, x2 + expand_side)
                        body_y2 = min(h, y2 + expand_down)
                        
                        body_bbox = (body_x1, body_y1, body_x2, body_y2)
                        body_emb = extract_body_embedding(frame, body_bbox)
                        
                        if body_emb is not None:
                            body_idx, body_score = match_body(body_emb, known_body_embeddings)
                            if body_idx is not None:
                                name, emp_id = known_faces[body_idx]
                                color = (0, 255, 255)  # Yellow for body match
                                match_method = "body"
                                logging.info(f"🔍 Body match: {name} ({emp_id}) - score: {body_score:.3f}")

                # Draw bounding box and label
                x1, y1, x2, y2 = bbox
                if name != "Unknown":
                    label = f"{name} ({emp_id}) [{match_method}]"
                else:
                    label = "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Process based on recognition result
                if name != "Unknown":
                    seen_known = process_known_face(
                        name, emp_id, frame, now, now_time,
                        seen_known, CAMERA_NAME, match_method
                    )
                else:
                    seen_unknown, unknown_cooldowns = process_unknown_face(
                        face, frame, now, now_time,
                        seen_unknown, unknown_cooldowns, CAMERA_NAME
                    )

        # Overlay camera info
        cv2.putText(frame, f"{CAMERA_NAME} Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display frame if not headless
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
        logging.info("⚠️ System stopped by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)