import sqlite3
import os

DB_PATH = "logs/attendance.db"

if not os.path.exists(DB_PATH):
    print(f"⚠️ Database not found at: {DB_PATH}")
else:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check if attendance_logs table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance_logs'")
    table_exists = cur.fetchone()

    if table_exists:
        print("🗑️ Deleting all attendance logs...")

        cur.execute("DELETE FROM attendance_logs")
        conn.commit()

        # Reclaim space
        cur.execute("VACUUM")
        conn.close()

        print("✅ All attendance logs deleted successfully (only once).")
    else:
        print("⚠️ Table 'attendance_logs' not found in database.")
