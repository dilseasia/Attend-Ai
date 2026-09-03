import os
import sys
import csv
import zipfile
import psycopg2
import paramiko
from datetime import datetime, timedelta
from attendance_db_postgres import DB_CONFIG

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
RECOGNIZED_DIR = os.path.join(BASE_DIR, "recognized_photos")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Remote Server Config
REMOTE_HOST = "10.8.14.83"
REMOTE_PORT = 22     # SSH / SFTP Port
REMOTE_USER = "faceattendance"
REMOTE_PASS = r"q6eMX/CZ1{gX?h^E\_jb"
REMOTE_DEST_DIR = "backups"


def get_connection():
    """Return PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def export_date_csv(target_date_str):
    """
    Export attendance_logs for target_date_str into date-specific CSV file.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Fetch attendance logs for specific date
    cur.execute("""
        SELECT id, name, emp_id, date, time, camera, location_type
        FROM attendance_logs
        WHERE date = %s
        ORDER BY emp_id, time
    """, (target_date_str,))
    log_rows = cur.fetchall()

    logs_csv_path = os.path.join(EXPORT_DIR, f"attendance_logs_{target_date_str}.csv")
    log_fields = ["id", "name", "emp_id", "date", "time", "camera", "location_type"]
    with open(logs_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_fields)
        writer.writerows(log_rows)

    print(f"✅ Saved logs CSV: {logs_csv_path} ({len(log_rows)} rows)")

    cur.close()
    conn.close()
    return logs_csv_path


def create_daily_zip(target_date_str, logs_csv):
    """
    Zip the date-wise attendance_logs CSV and the recognized_photos/YYYY-MM-DD directory into exports/backup_YYYY-MM-DD.zip.
    """
    zip_path = os.path.join(EXPORT_DIR, f"backup_{target_date_str}.zip")
    photo_date_dir = os.path.join(RECOGNIZED_DIR, target_date_str)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add attendance_logs CSV file
        if os.path.exists(logs_csv):
            zipf.write(logs_csv, arcname=os.path.basename(logs_csv))

        # Add recognized_photos date folder if it exists
        if os.path.exists(photo_date_dir):
            for root, _, files in os.walk(photo_date_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, BASE_DIR)
                    zipf.write(full_path, arcname=rel_path)
            print(f"📸 Included recognized photos folder: {photo_date_dir}")
        else:
            print(f"⚠️ No recognized photos folder found for date: {target_date_str}")

    print(f"📦 Created Zip Archive: {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.2f} MB)")
    return zip_path


def upload_to_server(file_path):
    """
    Upload file_path to remote server via SFTP (tries port 22 first, then 29898).
    """
    ports = [REMOTE_PORT, 29898]
    filename = os.path.basename(file_path)

    for port in ports:
        print(f"🚀 Attempting connection to {REMOTE_HOST}:{port}...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(REMOTE_HOST, port=port, username=REMOTE_USER, password=REMOTE_PASS, timeout=10)
            sftp = ssh.open_sftp()

            # Ensure remote dest dir exists
            try:
                sftp.mkdir(REMOTE_DEST_DIR)
            except IOError:
                pass  # Directory already exists

            remote_path = f"{REMOTE_DEST_DIR}/{filename}"
            print(f"📤 Uploading {filename} -> {REMOTE_HOST}:{remote_path}...")

            def progress_callback(transferred, total):
                percent = (transferred / total) * 100
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({transferred}/{total} bytes)")
                sys.stdout.flush()

            sftp.put(file_path, remote_path, callback=progress_callback)
            print(f"\n✅ Upload complete: {remote_path}")

            sftp.close()
            ssh.close()
            return True
        except Exception as e:
            print(f"⚠️ Failed on port {port}: {e}")

    print(f"❌ Upload failed on all ports for {filename}")
    return False


def run_backup(target_date_str=None):
    """
    Run backup for a specific date (default: yesterday).
    """
    if not target_date_str:
        today = datetime.now()
        target_date_str = today.strftime("%Y-%m-%d")

    print(f"\n==========================================")
    print(f"📅 Running Daily Backup for Date: {target_date_str}")
    print(f"==========================================")

    # 1. Export CSV
    logs_csv = export_date_csv(target_date_str)

    # 2. Create Zip
    zip_file = create_daily_zip(target_date_str, logs_csv)

    # 3. Upload to server
    upload_success = upload_to_server(zip_file)

    if upload_success:
        print(f"\n🎉 Daily Backup for {target_date_str} completed successfully!")
    else:
        print(f"\n⚠️ Backup created locally at {zip_file}, but remote upload could not complete.")


if __name__ == "__main__":
    # If date passed via argument (e.g. python3 daily_backup.py 2026-09-01)
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_backup(target_date)
