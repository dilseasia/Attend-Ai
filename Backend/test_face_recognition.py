import cv2
import numpy as np
import joblib
from insightface.app import FaceAnalysis

# === CONFIGURATION ===
TEST_IMAGE = "2025-10-30_11-10-19.jpg"   # <-- change this to your test image path
MODEL_FILE = "face_classifier.pkl"
ENCODER_FILE = "label_encoder.pkl"

# === LOAD MODEL ===
print("🧠 Loading trained model...")
clf = joblib.load(MODEL_FILE)
le = joblib.load(ENCODER_FILE)

# === INIT INSIGHTFACE ===
print("📸 Initializing InsightFace...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# === LOAD TEST IMAGE ===
img = cv2.imread(TEST_IMAGE)
if img is None:
    print(f"❌ Could not read image: {TEST_IMAGE}")
    exit(1)

faces = app.get(img)
if not faces:
    print("❌ No face detected in image.")
    exit(1)

# === CLASSIFY EACH FACE ===
for face in faces:
    emb = face.embedding.reshape(1, -1)
    probs = clf.predict_proba(emb)[0]
    max_prob = np.max(probs)
    pred_label = le.inverse_transform([np.argmax(probs)])[0]

    if max_prob > 0.6:  # Confidence threshold
        name, emp_id = pred_label.split("_")
        label_text = f"{name} ({emp_id}) {max_prob:.2f}"
        color = (0, 255, 0)
    else:
        label_text = f"Unknown ({max_prob:.2f})"
        color = (0, 0, 255)

    # Draw box and label
    box = face.bbox.astype(int)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    cv2.putText(img, label_text, (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# === SHOW RESULT ===
cv2.imshow("Face Recognition Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
