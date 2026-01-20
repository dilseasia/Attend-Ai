import os
import cv2
import re
import time
import logging
import requests
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# === CONFIGURATION ===
CAMERA_NAMES = ["Entry", "Exit"]
THRESHOLD = 0.5
ANONYMOUS_DIR = "Anonymous"
KNOWN_DIR = "known_faces"
API_URL = "http://10.8.21.51:8000/api/convert-anonymous"  # Your FastAPI endpoint
CHECK_INTERVAL = 60  # seconds to check if hour changed (e.g. 60s)
LOG_FILE = "anonymous_recognizer.log"

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)

# === INIT FACE RECOGNITION ===
logging.info("🧠 Initializing FaceAnalysis model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# === LOAD KNOWN FACES ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_known_faces():
    known_faces = {}
    known_embeddings = []

    for folder in os.listdir(KNOWN_DIR):
        if "_" not in folder:
            continue
        name, emp_id = folder.split("_", 1)
        folder_path = os.path.join(KNOWN_DIR, folder)
        for file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, file))
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                known_faces[len(known_embeddings)] = (name, emp_id)
                known_embeddings.append(faces[0].embedding)
    logging.info(f"✅ Loaded {len(known_embeddings)} known embeddings")
    return known_faces, known_embeddings

known_faces, known_embeddings = load_known_faces()

# === PROCESS FUNCTIONS ===
def process_anonymous_image(img_path, camera_name, date_str):
    """Try to recognize an anonymous face and call convert-anonymous API if matched."""
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"⚠️ Cannot read image: {img_path}")
        return

    faces = app.get(img)
    if not faces:
        return

    for face in faces:
        emb = face.embedding
        for idx, known_emb in enumerate(known_embeddings):
            if cosine_similarity(emb, known_emb) > (1 - THRESHOLD):
                name, emp_id = known_faces[idx]
                logging.info(f"✅ Recognized {name}_{emp_id} in {camera_name}: {img_path}")

                relative_path = os.path.relpath(img_path, os.getcwd()).replace("\\", "/")
                payload = {
                    "emp_id": emp_id,
                    "name": name,
                    "anon_path": relative_path,
                    "camera": camera_name,
                }

                try:
                    response = requests.post(API_URL, json=payload, timeout=10)
                    if response.status_code == 200:
                        res_json = response.json()
                        if res_json.get("success"):
                            logging.info(f"🚀 API success for {name}_{emp_id}: {res_json['message']}")
                        else:
                            logging.warning(f"⚠️ API returned error: {res_json}")
                    else:
                        logging.error(f"❌ API HTTP {response.status_code} for {img_path}")
                except Exception as e:
                    logging.error(f"❌ API call failed for {img_path}: {e}")
                return  # Stop after first match


def process_hourly_index(current_hour):
    """Read hourly index and process only images for current hour."""
    today = datetime.now().strftime("%Y-%m-%d")

    for camera_name in CAMERA_NAMES:
        cam_dir = os.path.join(ANONYMOUS_DIR, today, camera_name)
        if not os.path.exists(cam_dir):
            continue

        index_file = os.path.join(cam_dir, f"hourly_index_{current_hour:02d}.txt")
        if not os.path.exists(index_file):
            continue

        with open(index_file, "r") as f:
            photos = [line.strip() for line in f.readlines() if line.strip()]

        for photo_name in photos:
            img_path = os.path.join(cam_dir, photo_name)
            if os.path.exists(img_path):
                process_anonymous_image(img_path, camera_name, today)


# === MAIN LOOP (NO CRON NEEDED) ===
if __name__ == "__main__":
    logging.info("🚀 Starting continuous anonymous recognizer (auto-hourly mode)...")
    last_hour = None

    while True:
        now = datetime.now()
        current_hour = now.hour

        # Run once per hour
        if current_hour != last_hour:
            logging.info(f"🕒 Processing new hour: {current_hour:02d}")
            process_hourly_index(current_hour)
            last_hour = current_hour
            logging.info(f"✅ Hour {current_hour:02d} processing complete.")

        # Sleep before checking again
        time.sleep(CHECK_INTERVAL)
