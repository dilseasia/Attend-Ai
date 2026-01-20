import cv2
import numpy as np
import os
import logging
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# === CONFIGURATION ===
THRESHOLD = 0.5
KNOWN_FACES_DIR = "known_faces"
TEST_IMAGES_DIR = "test_images"  # Create this folder with test photos

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_known_faces(known_faces_dir=KNOWN_FACES_DIR):
    """Load all known faces and embeddings into memory"""
    known_faces = {}
    known_embeddings = []

    logging.info("Loading known faces for testing...")
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

    logging.info(f"✅ Loaded {len(known_embeddings)} known embeddings from {len(set(known_faces.values()))} people")
    return known_faces, known_embeddings


def predict_face(img_path, known_faces, known_embeddings, threshold=THRESHOLD):
    """
    Predict who is in the image
    Returns: (predicted_name, predicted_id, confidence_score, processing_time)
    """
    start_time = datetime.now()
    
    img = cv2.imread(img_path)
    if img is None:
        return None, None, 0, 0
    
    faces = app.get(img)
    if not faces:
        return "No Face Detected", "", 0, 0
    
    emb = faces[0].embedding
    name, emp_id = "Unknown", ""
    max_similarity = 0
    
    # Match against known faces
    for idx, known_emb in enumerate(known_embeddings):
        similarity = cosine_similarity(emb, known_emb)
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > (1 - threshold):
                name, emp_id = known_faces[idx]
    
    processing_time = (datetime.now() - start_time).total_seconds()
    return name, emp_id, max_similarity, processing_time


def run_evaluation():
    """
    Run evaluation on test images
    
    Test folder structure should be:
    test_images/
        ├── Name1_EmpID1/
        │   ├── photo1.jpg
        │   ├── photo2.jpg
        ├── Name2_EmpID2/
        │   ├── photo1.jpg
        ├── Unknown/  (optional - for testing unknown faces)
        │   ├── stranger1.jpg
    """
    
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Test directory '{TEST_IMAGES_DIR}' not found!")
        print(f"Please create it with structure: {TEST_IMAGES_DIR}/Name_EmployeeID/photo.jpg")
        return
    
    # Load known faces
    known_faces, known_embeddings = load_known_faces()
    
    # Store results
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels
    y_true_ids = []  # Ground truth IDs
    y_pred_ids = []  # Predicted IDs
    confidence_scores = []
    processing_times = []
    detailed_results = []
    
    print("\n" + "="*80)
    print("RUNNING FACE RECOGNITION EVALUATION")
    print("="*80 + "\n")
    
    total_images = 0
    correct_predictions = 0
    
    # Process each test folder
    for folder in os.listdir(TEST_IMAGES_DIR):
        folder_path = os.path.join(TEST_IMAGES_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Parse ground truth from folder name
        if folder.lower() == "unknown":
            true_name, true_id = "Unknown", ""
        elif "_" in folder:
            true_name, true_id = folder.split("_", 1)
        else:
            print(f"⚠️ Skipping folder '{folder}' - invalid format (use Name_EmployeeID)")
            continue
        
        # Process each image in folder
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, img_file)
            total_images += 1
            
            # Get prediction
            pred_name, pred_id, confidence, proc_time = predict_face(
                img_path, known_faces, known_embeddings
            )
            
            # Store results
            y_true.append(true_name)
            y_pred.append(pred_name)
            y_true_ids.append(f"{true_name}_{true_id}")
            y_pred_ids.append(f"{pred_name}_{pred_id}" if pred_name != "Unknown" else "Unknown")
            confidence_scores.append(confidence)
            processing_times.append(proc_time)
            
            is_correct = (pred_name == true_name and pred_id == true_id)
            if is_correct:
                correct_predictions += 1
            
            detailed_results.append({
                'image': img_path,
                'true_label': f"{true_name}_{true_id}",
                'predicted_label': f"{pred_name}_{pred_id}" if pred_name != "Unknown" else "Unknown",
                'confidence': confidence,
                'correct': is_correct,
                'processing_time': proc_time
            })
            
            # Print result
            status = "✅" if is_correct else "❌"
            print(f"{status} {img_file:30s} | True: {true_name:15s} | Pred: {pred_name:15s} | Conf: {confidence:.4f}")
    
    if total_images == 0:
        print("❌ No test images found!")
        return
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall accuracy
    accuracy = correct_predictions / total_images
    print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}% ({correct_predictions}/{total_images})")
    print(f"⏱️  Avg Processing Time: {np.mean(processing_times):.4f}s")
    print(f"📈 Avg Confidence Score: {np.mean(confidence_scores):.4f}")
    
    # Confusion Matrix
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print(f"\n{'':15s}", end='')
    for label in unique_labels:
        print(f"{label:15s}", end='')
    print()
    print("-" * (15 * (len(unique_labels) + 1)))
    
    for i, true_label in enumerate(unique_labels):
        print(f"{true_label:15s}", end='')
        for j, pred_label in enumerate(unique_labels):
            print(f"{cm[i][j]:15d}", end='')
        print()
    
    # Classification Report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Calculate per-class metrics
    print("\n" + "="*80)
    print("PER-PERSON STATISTICS")
    print("="*80)
    
    person_stats = {}
    for result in detailed_results:
        true_label = result['true_label']
        if true_label not in person_stats:
            person_stats[true_label] = {
                'total': 0, 'correct': 0, 'confidences': []
            }
        person_stats[true_label]['total'] += 1
        if result['correct']:
            person_stats[true_label]['correct'] += 1
        person_stats[true_label]['confidences'].append(result['confidence'])
    
    print(f"\n{'Person':<20s} {'Accuracy':<15s} {'Correct/Total':<15s} {'Avg Confidence':<15s}")
    print("-" * 70)
    for person, stats in sorted(person_stats.items()):
        acc = stats['correct'] / stats['total'] * 100
        avg_conf = np.mean(stats['confidences'])
        print(f"{person:<20s} {acc:>6.2f}%{'':<8s} {stats['correct']}/{stats['total']}{'':<10s} {avg_conf:.4f}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Face Recognition Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Confusion matrix saved as: {plot_filename}")
    
    # Save detailed results to JSON
    results_filename = f"evaluation_results_{timestamp}.json"
    
    # Convert all numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    detailed_results_clean = []
    for result in detailed_results:
        detailed_results_clean.append({
            'image': result['image'],
            'true_label': result['true_label'],
            'predicted_label': result['predicted_label'],
            'confidence': float(result['confidence']),
            'correct': bool(result['correct']),
            'processing_time': float(result['processing_time'])
        })
    
    with open(results_filename, 'w') as f:
        json.dump({
            'summary': {
                'total_images': int(total_images),
                'correct_predictions': int(correct_predictions),
                'accuracy': float(accuracy),
                'avg_confidence': float(np.mean(confidence_scores)),
                'avg_processing_time': float(np.mean(processing_times)),
                'threshold': float(THRESHOLD)
            },
            'per_person_stats': {k: {
                'total': int(v['total']),
                'correct': int(v['correct']),
                'confidences': [float(c) for c in v['confidences']]
            } for k, v in person_stats.items()},
            'detailed_results': detailed_results_clean,
            'confusion_matrix': cm.tolist(),
            'labels': unique_labels
        }, f, indent=2)
    print(f"💾 Detailed results saved as: {results_filename}")
    
    print("\n" + "="*80 + "\n")
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    print("\nRaw Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'detailed_results': detailed_results,
        'person_stats': person_stats
    }
    

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         FACE RECOGNITION SYSTEM - EVALUATION TOOL              ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Setup Instructions:
    1. Create a 'test_images' folder in the same directory
    2. Inside 'test_images', create folders for each person: Name_EmployeeID
    3. Put test photos in their respective folders
    4. Optionally create 'Unknown' folder for unknown faces
    
    Example structure:
    test_images/
        ├── John_EMP001/
        │   ├── test1.jpg
        │   ├── test2.jpg
        ├── Sarah_EMP002/
        │   ├── test1.jpg
        └── Unknown/
            └── stranger.jpg
    """)
    
    input("Press Enter to start evaluation...")
    run_evaluation()