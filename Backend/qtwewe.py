import sqlite3

conn = sqlite3.connect("logs/attendance.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(attendance_logs)")
for row in cursor.fetchall():
    print(row)

conn.close()
