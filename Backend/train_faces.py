import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# === CONFIGURATION ===
BASE_DIR = "known_faces"
EMBEDDING_FILE = "face_embeddings.npz"
MODEL_FILE = "face_classifier.pkl"
ENCODER_FILE = "label_encoder.pkl"

# === INIT INSIGHTFACE ===
print("🧠 Initializing InsightFace model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

X = []  # embeddings
y = []  # labels

# === EXTRACT EMBEDDINGS ===
print(f"📂 Scanning known faces from: {BASE_DIR}")
for folder in os.listdir(BASE_DIR):
    if "_" not in folder:
        continue
    label = folder
    folder_path = os.path.join(BASE_DIR, folder)

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        # ✅ Read image properly
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        faces = app.get(img)
        if faces:
            X.append(faces[0].embedding)
            y.append(label)
        else:
            print(f"[INFO] No face found in {img_path}")

# === CHECK DATA ===
if len(X) == 0:
    print("❌ No embeddings collected. Please check your known_faces directory.")
    exit(1)

X = np.array(X)
y = np.array(y)

print(f"✅ Collected {len(X)} embeddings for {len(set(y))} people.")

# === SAVE RAW EMBEDDINGS ===
np.savez(EMBEDDING_FILE, X=X, y=y)
print(f"💾 Embeddings saved to {EMBEDDING_FILE}")

# === TRAIN SVM CLASSIFIER ===
print("🎯 Training SVM classifier...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

clf = SVC(kernel='linear', probability=True)
clf.fit(X, y_encoded)

# === SAVE TRAINED MODEL ===
joblib.dump(clf, MODEL_FILE)
joblib.dump(le, ENCODER_FILE)

print(f"✅ Model trained and saved to {MODEL_FILE}")
print(f"✅ Label encoder saved to {ENCODER_FILE}")

# === OPTIONAL: Training report (accuracy check) ===
preds = clf.predict(X)
print("\n📊 Training Performance:")
print(classification_report(y_encoded, preds, target_names=le.classes_))
