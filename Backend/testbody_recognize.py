import os
import cv2
import torch
import torchreid
import pickle
import numpy as np
from PIL import Image

# === Configuration ===
EMBEDDING_PATH = "body_embeddings.pkl"  # Path to saved body embeddings
TEST_IMAGE_PATH = "testbody_images/2025-10-31_16-48-32.jpg"  
THRESHOLD = 0.90  

# === Load Stored Embeddings ===
if not os.path.exists(EMBEDDING_PATH):
    raise FileNotFoundError(f"❌ Embedding file not found: {EMBEDDING_PATH}")

with open(EMBEDDING_PATH, "rb") as f:
    known_embeddings = pickle.load(f)

print(f"✅ Loaded embeddings for {len(known_embeddings)} employees")

# === Initialize Model (OSNet) ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU if GPU not needed
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=1,
    pretrained=True
)
model.to(device)
model.eval()

# === Build Transform for Inference ===
_, transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    normalization=True
)

# === Load Test Image ===
image = cv2.imread(TEST_IMAGE_PATH)
if image is None:
    raise ValueError(f"❌ Could not read image at {TEST_IMAGE_PATH}")



# Convert OpenCV image (BGR) → RGB → PIL
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# Apply TorchReID transform
img_t = transform(img_pil).unsqueeze(0).to(device)

# === Generate Embedding for Test Image ===
with torch.no_grad():
    test_embedding = model(img_t)
    if isinstance(test_embedding, torch.Tensor):
        test_embedding = test_embedding.cpu().numpy().flatten()

# === Compare with Stored Embeddings ===
best_match = None
best_score = -1  # start with lowest possible
for name, emb in known_embeddings.items():
    emb = np.array(emb).flatten()  # ✅ ensure 1D vector
    similarity = np.dot(test_embedding, emb) / (
        np.linalg.norm(test_embedding) * np.linalg.norm(emb)
    )
    if similarity > best_score:
        best_score = similarity
        best_match = name

# === Display Result ===
print("\n🔍 Test Result:")
print(f"Best match: {best_match}")
print(f"Similarity score: {best_score:.4f}")

if best_score >= THRESHOLD:
    print(f"✅ Recognized as {best_match}")
else:
    print("❌ Not recognized (below threshold)")

# === Optional: Show Image with Label ===
cv2.putText(image, f"{best_match} ({best_score:.2f})", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (0, 255, 0) if best_score >= THRESHOLD else (0, 0, 255), 3)

cv2.imshow("Body Recognition Test", image)
print("Press any key to close window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
