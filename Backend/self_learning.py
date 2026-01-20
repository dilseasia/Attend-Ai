# ==================================
# 🧠 self_learning.py (No Duplicates, Persistent Cache)
# ==================================
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from insightface.app import FaceAnalysis

# ==================================
# ⚙️ CONFIGURATION
# ==================================
RECOGNIZED_DIR = "recognized_photos"
KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "self_learning.log"

THRESHOLD_SIMILARITY = 0.75   # Minimum similarity to count as same person
DUPLICATE_THRESHOLD = 0.99    # Above this → considered duplicate
MAX_SAMPLES_PER_PERSON = 30   # Max auto images to keep per person
SELF_LEARN_PREFIX = "auto_"   # Prefix for auto-learned samples
EMBEDDING_CACHE_FILE = "embeddings.npy"  # Cache file for faster duplicate check

# ==================================
# 🧩 LOGGING
# ==================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a")
    ]
)

# ==================================
# 🧱 UTILITIES
# ==================================
def safe_makedirs(path: str):
    """Safely create a directory if it doesn’t exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logging.warning(f"⚠️ Could not create folder {path}: {e} - self_learning.py:44")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==================================
# 🚀 INITIALIZE INSIGHTFACE
# ==================================
logging.info("Initializing InsightFace for selflearning... - self_learning.py:53")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)
logging.info("✅ InsightFace initialized successfully. - self_learning.py:56")

# ==================================
# 📂 LOAD EXISTING EMBEDDINGS
# ==================================
def load_existing_embeddings():
    """Load all known face embeddings (from cache if available)."""
    known_embeddings = {}
    for folder in os.listdir(KNOWN_FACES_DIR):
        if "_" not in folder:
            continue
        person_dir = os.path.join(KNOWN_FACES_DIR, folder)
        emb_path = os.path.join(person_dir, EMBEDDING_CACHE_FILE)

        # Try to load from cache first
        if os.path.exists(emb_path):
            try:
                known_embeddings[folder] = np.load(emb_path, allow_pickle=True).tolist()
                logging.info(f"📦 Loaded cached embeddings for {folder} ({len(known_embeddings[folder])} samples) - self_learning.py:74")
                continue
            except Exception as e:
                logging.warning(f"⚠️ Could not load {emb_path}: {e} - self_learning.py:77")

        # If no cache, compute from images
        emb_list = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            emb_list.append(faces[0].embedding)

        if emb_list:
            np.save(emb_path, emb_list)
            known_embeddings[folder] = emb_list
            logging.info(f"📦 Created new embedding cache for {folder} ({len(emb_list)} samples) - self_learning.py:96")

    return known_embeddings

# ==================================
# 🧠 SELF-LEARNING PROCESS
# ==================================
def self_learn():
    """Add new face samples for existing known identities only."""
    logging.info("🚀 Starting selflearning (only known faces)... - self_learning.py:105")
    known_embeddings = load_existing_embeddings()

    for date_folder in sorted(os.listdir(RECOGNIZED_DIR)):
        date_path = os.path.join(RECOGNIZED_DIR, date_folder)
        if not os.path.isdir(date_path):
            continue

        # Loop through recognized people
        for person_folder in os.listdir(date_path):
            if person_folder not in known_embeddings:
                logging.info(f"⏭️ Skipping new identity '{person_folder}' (not in known_faces). - self_learning.py:116")
                continue

            person_dir = os.path.join(KNOWN_FACES_DIR, person_folder)
            safe_makedirs(person_dir)

            emb_cache_path = os.path.join(person_dir, EMBEDDING_CACHE_FILE)
            person_embs = known_embeddings[person_folder]

            person_path = os.path.join(date_path, person_folder)
            for cam_folder in os.listdir(person_path):
                cam_path = os.path.join(person_path, cam_folder)
                for file in os.listdir(cam_path):
                    img_path = os.path.join(cam_path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    faces = app.get(img)
                    if not faces:
                        continue

                    face = faces[0]
                    new_emb = face.embedding
                    face_crop = img[
                        int(face.bbox[1]):int(face.bbox[3]),
                        int(face.bbox[0]):int(face.bbox[2])
                    ]
                    if face_crop.size == 0:
                        continue

                    # Compare with cached embeddings
                    similarities = [cosine_similarity(new_emb, e) for e in person_embs]
                    max_sim = max(similarities) if similarities else 0

                    # Skip if too similar (duplicate)
                    if max_sim > DUPLICATE_THRESHOLD:
                        logging.info(f"⚠️ Skipping duplicate for {person_folder} (sim={max_sim:.2f}) - self_learning.py:153")
                        continue

                    # Add if moderately new
                    if THRESHOLD_SIMILARITY < max_sim < 0.98 or not person_embs:
                        filename = f"{SELF_LEARN_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        save_path = os.path.join(person_dir, filename)
                        cv2.imwrite(save_path, face_crop)
                        person_embs.append(new_emb)
                        np.save(emb_cache_path, person_embs)
                        logging.info(f"🧠 Added new sample for {person_folder} (sim={max_sim:.2f}) - self_learning.py:163")

                        # Keep folder clean
                        auto_files = sorted(
                            [f for f in os.listdir(person_dir) if f.startswith(SELF_LEARN_PREFIX)],
                            key=lambda x: os.path.getmtime(os.path.join(person_dir, x))
                        )
                        if len(auto_files) > MAX_SAMPLES_PER_PERSON:
                            old_file = os.path.join(person_dir, auto_files[0])
                            os.remove(old_file)
                            logging.info(f"🧹 Removed old autosample {old_file} - self_learning.py:173")

    logging.info("✅ Selflearning process finished successfully. - self_learning.py:175")

# ==================================
# 🔁 MAIN
# ==================================
if __name__ == "__main__":
    self_learn()
3