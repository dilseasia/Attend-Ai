import psycopg2
from datetime import datetime
from attendance_db_postgres import DB_CONFIG  # Import your connection config from attendance_db_postgres.py

def get_connection():
    """Return a PostgreSQL connection using DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)

def backfill_daily_summary():
    conn = get_connection()
    cur = conn.cursor()

    # Ensure daily_summary table exists
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
        exit_times = [t for t, c in logs if c and c.lower() == "exit"]

        if not entry_times:
            continue

        first_entry = entry_times[0]
        last_exit = exit_times[-1] if exit_times else None

        # 🕒 Calculate total working hours
        total_seconds = 0
        last_entry = None

        for time, cam in logs:
            if cam.lower() == "entry":
                last_entry = time
            elif cam.lower() == "exit" and last_entry:
                try:
                    t1 = datetime.strptime(str(last_entry), "%H:%M:%S")
                    t2 = datetime.strptime(str(time), "%H:%M:%S")
                    diff = (t2 - t1).seconds
                    if diff > 0:
                        total_seconds += diff
                except Exception as e:
                    print(f"⚠️ Error calculating duration for {emp_id} {date}: {e}")
                last_entry = None

        # If employee still inside (no exit yet)
        if last_entry:
            try:
                t1 = datetime.strptime(str(last_entry), "%H:%M:%S")
                now = datetime.now()
                diff = (now - t1).seconds
                if diff > 0:
                    total_seconds += diff
            except Exception as e:
                print(f"⚠️ Error adding ongoing duration for {emp_id}: {e}")

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        working_hours = f"{hours}h {minutes}m"
        entry_count = len(entry_times)
        exit_count = len(exit_times)
        status = "Present"

        # 💾 Save or update record in daily_summary
        cur.execute("""
            INSERT INTO daily_summary (
                emp_id, name, date, working_hours,
                entry_count, exit_count, first_entry, last_exit, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (emp_id, date)
            DO UPDATE SET
                name = EXCLUDED.name,
                working_hours = EXCLUDED.working_hours,
                entry_count = EXCLUDED.entry_count,
                exit_count = EXCLUDED.exit_count,
                first_entry = EXCLUDED.first_entry,
                last_exit = EXCLUDED.last_exit,
                status = EXCLUDED.status;
        """, (
            emp_id, name, date, working_hours,
            entry_count, exit_count, first_entry, last_exit, status
        ))

        total_saved += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Backfill completed successfully. {total_saved} daily summary records added/updated.")


if __name__ == "__main__":
    backfill_daily_summary()
