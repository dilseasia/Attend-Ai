import sqlite3
from attendance_db_postgres import get_connection

# ✅ Path to your SQLite database
SQLITE_DB = "logs/attendance.db"

# Connect to SQLite
sqlite_conn = sqlite3.connect(SQLITE_DB)
sqlite_cur = sqlite_conn.cursor()

# Connect to PostgreSQL
pg_conn = get_connection()
pg_cur = pg_conn.cursor()

# ==========================
# 🔁 MIGRATE attendance_logs
# ==========================
print("➡️ Migrating attendance_logs...")

sqlite_cur.execute("SELECT name, emp_id, date, time, camera FROM attendance_logs")
rows = sqlite_cur.fetchall()

for row in rows:
    name, emp_id, date, time, camera = row
    try:
        pg_cur.execute("""
            INSERT INTO attendance_logs (name, emp_id, date, time, camera)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, emp_id, date, time, camera))
    except Exception as e:
        print(f"⚠️ Skipped record ({emp_id}, {date}):", e)

pg_conn.commit()
print(f"✅ Migrated {len(rows)} attendance_logs rows.")

# ==========================
# 🔁 MIGRATE daily_summary
# ==========================
print("➡️ Migrating daily_summary...")

sqlite_cur.execute("""
    SELECT emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status
    FROM daily_summary
""")
rows = sqlite_cur.fetchall()

for row in rows:
    emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status = row
    try:
        pg_cur.execute("""
            INSERT INTO daily_summary (
                emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (emp_id, date)
            DO NOTHING;
        """, (emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status))
    except Exception as e:
        print(f"⚠️ Skipped summary ({emp_id}, {date}):", e)

pg_conn.commit()
print(f"✅ Migrated {len(rows)} daily_summary rows.")

# Close connections
sqlite_conn.close()
pg_conn.close()

print("🎉 Migration completed successfully!")
