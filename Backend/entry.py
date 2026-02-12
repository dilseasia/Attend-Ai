import cv2
import numpy as np
import os
import time
import logging
import psutil
from datetime import datetime
from collections import deque
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
THRESHOLD = 0.5
FRAME_INTERVAL = 1.0
SKIP_FRAMES = 1
UNKNOWN_COOLDOWN = 10
UNKNOWN_FACE_MIN_AREA = 2500
ENTRY_COOLDOWN = 1 * 60
HEADLESS = True

# ✅ NEW: CPU-AWARE CONFIGURATION
CPU_CHECK_INTERVAL = 2.0  # Check CPU every 2 seconds
CPU_HIGH_THRESHOLD = 800  # Consider CPU high above 800%
CPU_CRITICAL_THRESHOLD = 1000  # Critical above 1000%
MAX_FACES_NORMAL = 15  # Process all faces when CPU is normal
MAX_FACES_HIGH = 8   # Process max 8 faces when CPU is high
MAX_FACES_CRITICAL = 4  # Process max 4 faces when CPU is critical
BATCH_QUEUE_SIZE = 50  # Queue size for deferred processing

# === INITIALIZE ===
os.environ["QT_QPA_PLATFORM"] = "offscreen" if HEADLESS else "xcb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("entry.log", mode="a")],
)

logging.info(f"🚀 Initializing {CAMERA_NAME} camera with CPU-aware processing...")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(480, 480))

init_db()
init_summary_table()


# ✅ NEW: CPU Monitoring Class
class CPUMonitor:
    """Monitor CPU usage and adjust processing mode"""
    
    def __init__(self):
        self.last_check = 0
        self.current_cpu = 0
        self.cpu_history = deque(maxlen=5)  # Keep last 5 readings
        self.mode = "normal"  # normal, high, critical
        
    def get_cpu_usage(self):
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def update(self):
        """Update CPU status if enough time has passed"""
        now = time.time()
        
        if now - self.last_check >= CPU_CHECK_INTERVAL:
            self.current_cpu = self.get_cpu_usage()
            self.cpu_history.append(self.current_cpu)
            self.last_check = now
            
            # Calculate average CPU from history
            avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
            
            # Determine mode based on average
            if avg_cpu >= CPU_CRITICAL_THRESHOLD:
                self.mode = "critical"
            elif avg_cpu >= CPU_HIGH_THRESHOLD:
                self.mode = "high"
            else:
                self.mode = "normal"
        
        return self.mode, self.current_cpu
    
    def get_max_faces(self):
        """Get maximum faces to process based on CPU load"""
        if self.mode == "critical":
            return MAX_FACES_CRITICAL
        elif self.mode == "high":
            return MAX_FACES_HIGH
        else:
            return MAX_FACES_NORMAL
    
    def should_defer_processing(self):
        """Check if we should defer processing to background queue"""
        return self.mode == "critical"


# ✅ NEW: Background Processing Queue
class BackgroundQueue:
    """Queue for deferred face processing when CPU is high"""
    
    def __init__(self, max_size=BATCH_QUEUE_SIZE):
        self.queue = deque(maxlen=max_size)
        self.processed_count = 0
        self.dropped_count = 0
    
    def add(self, face_data):
        """Add face detection task to queue"""
        if len(self.queue) >= BATCH_QUEUE_SIZE:
            self.dropped_count += 1
            return False
        
        self.queue.append(face_data)
        return True
    
    def process_batch(self, known_faces, known_embeddings, cpu_monitor):
        """Process queued faces when CPU is available"""
        
        # Only process queue when CPU is back to normal
        if cpu_monitor.mode != "normal":
            return 0
        
        processed = 0
        max_to_process = 10  # Process max 10 from queue per cycle
        
        while self.queue and processed < max_to_process:
            # Check CPU hasn't spiked again
            if cpu_monitor.get_cpu_usage() > CPU_HIGH_THRESHOLD:
                break
            
            face_data = self.queue.popleft()
            
            # Process this deferred face
            try:
                self._process_deferred_face(face_data, known_faces, known_embeddings)
                processed += 1
                self.processed_count += 1
            except Exception as e:
                logging.error(f"❌ Error processing queued face: {e}")
        
        return processed
    
    def _process_deferred_face(self, face_data, known_faces, known_embeddings):
        """Process a single deferred face detection"""
        emb = face_data['embedding']
        frame = face_data['frame']
        now = face_data['timestamp']
        bbox = face_data['bbox']
        
        # Vectorized matching
        if len(known_embeddings) > 0:
            similarities = np.dot(known_embeddings, emb) / (
                np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb)
            )
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])
            
            if best_similarity > (1 - THRESHOLD):
                name, emp_id = known_faces[best_idx]
                
                # Log attendance from queued detection
                key = f"{name}_{emp_id}"
                
                photo_path = os.path.join(
                    "recognized_photos",
                    now.strftime("%Y-%m-%d"),
                    key,
                    CAMERA_NAME,
                    f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_queued.jpg"
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
                
                logging.info(f"✅ [QUEUED] Logged entry for {key}")


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
        print(f"✅ {CAMERA_NAME} logged entry for {key}")

        seen_known[key] = now_time
        logging.info(f"✅ {CAMERA_NAME} logged entry for {key}")
    return seen_known


def process_unknown_face(face, frame, now, now_time, seen_unknown, unknown_cooldowns):
    """Handle logic for unknown faces"""
    emb = face.embedding
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)

    if face_area < UNKNOWN_FACE_MIN_AREA:
        return seen_unknown, unknown_cooldowns

    emb_key = tuple(np.round(emb[:50], 3))
    
    if emb_key in seen_unknown:
        return seen_unknown, unknown_cooldowns

    if now_time - unknown_cooldowns.get(emb_key, 0) >= UNKNOWN_COOLDOWN:
        unknown_dir = os.path.join("Anonymous", now.strftime("%Y-%m-%d"), CAMERA_NAME)
        os.makedirs(unknown_dir, exist_ok=True)

        milliseconds = int(now.microsecond / 1000)
        photo_name = f"{now.strftime('%H-%M-%S')}-{milliseconds:03d}.jpg"
        photo_path = os.path.join(unknown_dir, photo_name)
        cv2.imwrite(photo_path, frame)

        save_hourly_index(unknown_dir, now, photo_name)

        unknown_cooldowns[emb_key] = now_time
        seen_unknown.add(emb_key)
        logging.info(f"📸 Unknown saved: {photo_name}")

    return seen_unknown, unknown_cooldowns


def main_loop():
    """Main processing loop with CPU-aware batching"""
    
    known_faces, known_embeddings = load_known_faces()

    seen_known = {}
    seen_unknown = set()
    unknown_cooldowns = {}
    last_frame_time = time.time()
    frame_counter = 0
    
    # ✅ NEW: Initialize CPU monitor and background queue
    cpu_monitor = CPUMonitor()
    bg_queue = BackgroundQueue()
    
    # Statistics
    stats = {
        'total_frames': 0,
        'processed_faces': 0,
        'queued_faces': 0,
        'dropped_faces': 0,
        'last_log_time': time.time()
    }

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logging.error(f"❌ {CAMERA_NAME} camera not accessible.")
        return

    logging.info(f"🎥 {CAMERA_NAME} camera running with intelligent CPU management...")

    while True:
        ret, frame = cap.read()
        now = datetime.now()
        now_time = time.time()

        if not ret:
            logging.warning(f"⚠️ Reconnecting {CAMERA_NAME}...")
            time.sleep(1)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        stats['total_frames'] += 1

        # ✅ NEW: Update CPU status
        mode, cpu_usage = cpu_monitor.update()
        max_faces = cpu_monitor.get_max_faces()

        # Frame skipping
        frame_counter += 1
        if frame_counter % (SKIP_FRAMES + 1) != 0:
            continue

        # Frame interval limiter
        if now_time - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now_time

        # Periodic cleanup
        unknown_cooldowns = {
            k: v for k, v in unknown_cooldowns.items()
            if now_time - v < UNKNOWN_COOLDOWN * 3
        }

        # Detect faces
        faces = app.get(frame)
        
        # ✅ NEW: Limit faces based on CPU load
        num_faces = len(faces)
        
        if num_faces > max_faces:
            # Sort faces by size (larger faces = closer/more important)
            faces_with_size = []
            for face in faces:
                bbox = face.bbox.astype(int)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                faces_with_size.append((face, area))
            
            # Sort by area descending and take top N
            faces_with_size.sort(key=lambda x: x[1], reverse=True)
            
            # Process top priority faces immediately
            faces_to_process = [f[0] for f in faces_with_size[:max_faces]]
            
            # Queue remaining faces for later (if in critical mode)
            if mode == "critical":
                faces_to_queue = [f[0] for f in faces_with_size[max_faces:]]
                
                for face in faces_to_queue:
                    queued = bg_queue.add({
                        'embedding': face.embedding,
                        'frame': frame.copy(),
                        'timestamp': now,
                        'bbox': face.bbox.astype(int)
                    })
                    
                    if queued:
                        stats['queued_faces'] += 1
                    else:
                        stats['dropped_faces'] += 1
                
                logging.warning(
                    f"⚠️ CPU {mode.upper()} ({cpu_usage:.0f}%): "
                    f"Processing {len(faces_to_process)}/{num_faces} faces, "
                    f"Queued: {len(faces_to_queue)}, Queue size: {len(bg_queue.queue)}"
                )
            else:
                # In high mode, just skip extra faces
                logging.info(
                    f"ℹ️ CPU {mode.upper()} ({cpu_usage:.0f}%): "
                    f"Processing {len(faces_to_process)}/{num_faces} faces"
                )
            
            faces = faces_to_process
        
        # Process current faces
        for face in faces:
            emb = face.embedding
            bbox = face.bbox.astype(int)
            name, emp_id = "Unknown", ""
            color = (0, 0, 255)

            # Vectorized matching
            if len(known_embeddings) > 0:
                similarities = np.dot(known_embeddings, emb) / (
                    np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb)
                )
                best_idx = int(np.argmax(similarities))
                best_similarity = float(similarities[best_idx])
                
                if best_similarity > (1 - THRESHOLD):
                    name, emp_id = known_faces[best_idx]
                    color = (0, 255, 0)

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
                stats['processed_faces'] += 1
            else:
                seen_unknown, unknown_cooldowns = process_unknown_face(
                    face, frame, now, now_time,
                    seen_unknown, unknown_cooldowns
                )

        # ✅ NEW: Process background queue when CPU is available
        if mode == "normal" and len(bg_queue.queue) > 0:
            processed = bg_queue.process_batch(known_faces, known_embeddings, cpu_monitor)
            if processed > 0:
                logging.info(f"🔄 Processed {processed} queued faces, {len(bg_queue.queue)} remaining")

        # Display CPU mode on frame
        mode_color = {
            'normal': (0, 255, 0),
            'high': (0, 165, 255),
            'critical': (0, 0, 255)
        }
        
        cv2.putText(frame, f"{CAMERA_NAME} - CPU: {mode.upper()} ({cpu_usage:.0f}%)", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color[mode], 2)
        cv2.putText(frame, f"Processing: {len(faces)}/{num_faces} faces | Queue: {len(bg_queue.queue)}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Log statistics every 60 seconds
        if now_time - stats['last_log_time'] >= 60:
            logging.info(
                f"📊 STATS - Frames: {stats['total_frames']} | "
                f"Processed: {stats['processed_faces']} | "
                f"Queued: {stats['queued_faces']} | "
                f"Dropped: {stats['dropped_faces']} | "
                f"Queue: {len(bg_queue.queue)}/{BATCH_QUEUE_SIZE}"
            )
            stats['last_log_time'] = now_time

        if not HEADLESS:
            cv2.imshow(CAMERA_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()