import psycopg2
import csv
import os
from datetime import datetime
from attendance_db_postgres import DB_CONFIG

# 📁 CSV output directory
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

CSV_FILE = os.path.join(EXPORT_DIR, "daily_summary.csv")
ATTENDANCE_LOGS_CSV = os.path.join(EXPORT_DIR, "attendance_logs.csv")


def get_connection():
    """Return a PostgreSQL connection using DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)


def backfill_daily_summary():
    conn = get_connection()
    cur = conn.cursor()

    # ✅ Ensure daily_summary table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id SERIAL PRIMARY KEY,
            emp_id TEXT,
            name TEXT,
            date DATE,
            working_hours TEXT,
            entry_count INTEGER,
            exit_count INTEGER,
            first_entry TIME,
            last_exit TIME,
            status TEXT,
            UNIQUE(emp_id, date)
        )
    """)

    # ✅ Fetch all distinct employee-date pairs
    cur.execute("SELECT DISTINCT emp_id, name, date FROM attendance_logs")
    pairs = cur.fetchall()

    print(f"📋 Found {len(pairs)} employee-date combinations to process...")

    total_saved = 0
    csv_rows = []

    for emp_id, name, date in pairs:
        # Get all logs for that employee and date
        cur.execute("""
            SELECT time, camera FROM attendance_logs
            WHERE emp_id = %s AND date = %s
            ORDER BY time
        """, (emp_id, date))
        logs = cur.fetchall()

        if not logs:
            continue

        entry_times = [t for t, c in logs if c and c.lower() == "entry"]
        exit_times  = [t for t, c in logs if c and c.lower() == "exit"]

        if not entry_times:
            continue

        first_entry = entry_times[0]
        last_exit   = exit_times[-1] if exit_times else None

        # 🕒 Calculate total working hours
        total_seconds = 0
        last_entry = None

        for time, cam in logs:
            if cam.lower() == "entry":
                last_entry = time
            elif cam.lower() == "exit" and last_entry:
                try:
                    t1   = datetime.strptime(str(last_entry), "%H:%M:%S")
                    t2   = datetime.strptime(str(time), "%H:%M:%S")
                    diff = (t2 - t1).seconds
                    if diff > 0:
                        total_seconds += diff
                except Exception as e:
                    print(f"⚠️ Error calculating duration for {emp_id} {date}: {e}")
                last_entry = None

        # If employee still inside (no exit yet)
        if last_entry:
            try:
                t1   = datetime.strptime(str(last_entry), "%H:%M:%S")
                now  = datetime.now()
                diff = (now - t1).seconds
                if diff > 0:
                    total_seconds += diff
            except Exception as e:
                print(f"⚠️ Error adding ongoing duration for {emp_id}: {e}")

        hours         = total_seconds // 3600
        minutes       = (total_seconds % 3600) // 60
        working_hours = f"{hours}h {minutes}m"
        entry_count   = len(entry_times)
        exit_count    = len(exit_times)
        status        = "Present"

        # 💾 Save / update record in PostgreSQL daily_summary
        cur.execute("""
            INSERT INTO daily_summary (
                emp_id, name, date, working_hours,
                entry_count, exit_count, first_entry, last_exit, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (emp_id, date)
            DO UPDATE SET
                name          = EXCLUDED.name,
                working_hours = EXCLUDED.working_hours,
                entry_count   = EXCLUDED.entry_count,
                exit_count    = EXCLUDED.exit_count,
                first_entry   = EXCLUDED.first_entry,
                last_exit     = EXCLUDED.last_exit,
                status        = EXCLUDED.status;
        """, (
            emp_id, name, date, working_hours,
            entry_count, exit_count, first_entry, last_exit, status
        ))

        total_saved += 1

        # 📝 Collect row for CSV
        csv_rows.append({
            "emp_id":        emp_id,
            "name":          name,
            "date":          str(date),
            "working_hours": working_hours,
            "entry_count":   entry_count,
            "exit_count":    exit_count,
            "first_entry":   str(first_entry) if first_entry else "",
            "last_exit":     str(last_exit)   if last_exit   else "",
            "status":        status,
        })

    conn.commit()

    # ✅ Write daily_summary CSV
    fieldnames = ["emp_id", "name", "date", "working_hours",
                  "entry_count", "exit_count", "first_entry", "last_exit", "status"]

    # Sort by date desc, then name
    csv_rows.sort(key=lambda r: (r["date"], r["name"]), reverse=False)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"✅ daily_summary CSV saved  →  {CSV_FILE}  ({len(csv_rows)} rows)")

    # ✅ Also export raw attendance_logs to CSV
    cur.execute("""
        SELECT id, name, emp_id, date, time, camera, location_type
        FROM attendance_logs
        ORDER BY date, emp_id, time
    """)
    log_rows = cur.fetchall()
    log_fields = ["id", "name", "emp_id", "date", "time", "camera", "location_type"]

    with open(ATTENDANCE_LOGS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_fields)
        writer.writerows(log_rows)

    print(f"✅ attendance_logs CSV saved →  {ATTENDANCE_LOGS_CSV}  ({len(log_rows)} rows)")

    cur.close()
    conn.close()

    print(f"\n🎉 Backfill done! {total_saved} daily summary records added/updated in DB + CSV.")
    print(f"📂 All exports saved in: {EXPORT_DIR}")


if __name__ == "__main__":
    backfill_daily_summary()
