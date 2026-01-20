import cv2
import numpy as np
import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from database_new import init_db, log_attendance

# === Configuration ===
RTSP_URL = "rtsp://admin:admin123@10.8.21.47:554/cam/realmonitor?channel=1&subtype=0"
THRESHOLD = 0.5
FRAME_INTERVAL = 1
COOLDOWN_SECONDS = 10
CAMERA_NAME = "Exit"  # Change to "Exit" for exit cam

# === Init FaceAnalysis and DB ===
os.environ["QT_QPA_PLATFORM"] = "xcb"
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
init_db()

# === Load known faces ===
known_faces = {}  # id: (name, emp_id)
known_embeddings = []
last_logged_time = {}

def load_known_faces():
    for folder in os.listdir("known_faces"):
        folder_path = os.path.join("known_faces", folder)
        if not os.path.isdir(folder_path) or "_" not in folder:
            continue
        name, emp_id = folder.split("_", 1)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                known_faces[len(known_embeddings)] = (name, emp_id)
                known_embeddings.append(faces[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Start video ===
load_known_faces()
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Entry camera not accessible.")
    exit()

print(f"🎥 {CAMERA_NAME} camera stream started. Press 'q' to quit.")
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    now = datetime.now()

    if not ret:
        print("⚠️ Camera disconnected. Reconnecting...")
        time.sleep(1)
        cap = cv2.VideoCapture(RTSP_URL)
        continue

    if time.time() - last_frame_time < FRAME_INTERVAL:
        continue
    last_frame_time = time.time()

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding
        bbox = face.bbox.astype(int)
        name, emp_id = "Unknown", ""
        color = (0, 0, 255)  # Red for unknown

        for idx, known_emb in enumerate(known_embeddings):
            if cosine_similarity(emb, known_emb) > (1 - THRESHOLD):
                name, emp_id = known_faces[idx]
                color = (0, 255, 0)  # Green for known
                break

        x1, y1, x2, y2 = bbox
        label = f"{name} ({emp_id})" if name != "Unknown" else "Unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if name != "Unknown":
            key = f"{name}_{emp_id}"
            last_time = last_logged_time.get(key, 0)
            if time.time() - last_time > COOLDOWN_SECONDS:
                date_today = now.strftime("%Y-%m-%d")
                entry_time = now.strftime("%H:%M:%S")
                timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

                # === Save today's cropped image if not already ===
                folder_path = os.path.join("known_faces", f"{name}_{emp_id}")
                os.makedirs(folder_path, exist_ok=True)
                today_filename = f"{date_today}.jpg"
                save_path = os.path.join(folder_path, today_filename)

                if not os.path.exists(save_path):
                    face_crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(save_path, face_crop)
                    print(f"📸 Saved today's image for {name} ({emp_id})")

                # ✅ Save full image to new path
                folder_path = os.path.join("recognized_photos", date_today, key, CAMERA_NAME)
                os.makedirs(folder_path, exist_ok=True)
                filename = f"{timestamp}.jpg"
                save_path = os.path.join(folder_path, filename)

                cv2.imwrite(save_path, frame)
                print(f"📸 Saved full image for {name} ({emp_id}) at {save_path}")

                log_attendance(name, emp_id, date_today, entry_time, CAMERA_NAME, photo_path=save_path)
                last_logged_time[key] = time.time()

    # === Overlay camera status & timestamp ===
    cv2.putText(frame, f"{CAMERA_NAME} Camera - Connected", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S"), (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(f"{CAMERA_NAME} Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
