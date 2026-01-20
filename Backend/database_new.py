import sqlite3
from datetime import datetime

DB_PATH = "logs/attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            emp_id TEXT,
            date TEXT,
            time TEXT,
            camera TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_attendance(name, emp_id, date, time, camera):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attendance_logs (name, emp_id, date, time, camera)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, emp_id, date, time, camera))
    conn.commit()
    conn.close()



def init_summary_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id TEXT,
            name TEXT,
            date TEXT,
            working_hours TEXT,
            entry_count INTEGER,
            exit_count INTEGER,
            first_entry TEXT,
            last_exit TEXT,
            status TEXT,
            UNIQUE(emp_id, date)
        )
    """)
    conn.commit()
    conn.close()


def save_daily_summary(emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO daily_summary
        (emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status))
    conn.commit()
    conn.close()


def get_daily_summary(date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status
        FROM daily_summary
        WHERE date = ?
    """, (date,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "emp_id": r[0],
            "name": r[1],
            "date": r[2],
            "working_hours": r[3],
            "entry_count": r[4],
            "exit_count": r[5],
            "first_entry": r[6],
            "last_exit": r[7],
            "status": r[8]
        } for r in rows
    ]


