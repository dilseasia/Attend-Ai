import os
import shutil

SOURCE_DIR = "recognized_photos"
TARGET_DIR = "known_bodies"
KNOWN_FACES_DIR = "known_faces"

os.makedirs(TARGET_DIR, exist_ok=True)

# Get valid employee folder names from known_faces
valid_employees = set(os.listdir(KNOWN_FACES_DIR))
print(f"📁 Found {len(valid_employees)} valid employees in known_faces")

count_copied = 0

for date_folder in os.listdir(SOURCE_DIR):
    date_path = os.path.join(SOURCE_DIR, date_folder)
    if not os.path.isdir(date_path):
        continue

    for emp_folder in os.listdir(date_path):
        if emp_folder not in valid_employees:
            continue  # Skip if not in known_faces

        emp_path = os.path.join(date_path, emp_folder)
        if not os.path.isdir(emp_path):
            continue

        target_emp_dir = os.path.join(TARGET_DIR, emp_folder)
        os.makedirs(target_emp_dir, exist_ok=True)

        for root, _, files in os.walk(emp_path):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    src = os.path.join(root, file)
                    dst = os.path.join(target_emp_dir, file)
                    shutil.copy2(src, dst)
                    count_copied += 1

print(f"✅ known_bodies folder created with body samples for {len(valid_employees)} employees.")
print(f"🖼️ Total copied images: {count_copied}")
