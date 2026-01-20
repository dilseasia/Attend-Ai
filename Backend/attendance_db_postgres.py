import psycopg2
from datetime import datetime

# ✅ PostgreSQL Database Configuration
DB_CONFIG = {
    "dbname": "attendanceDB",
    "user": "postgres",
    "password": "daljeet@123",
    "host": "10.8.21.51",
    "port": "5432"      
}



def get_connection():
    """Create and return a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print("❌ Database connection failed: - attendance_db_postgres.py:20", e)
        raise


def init_db():
    """Create the attendance_logs table if it doesn't exist."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_logs (
                        id SERIAL PRIMARY KEY,
                        name TEXT,
                        emp_id TEXT,
                        date DATE,
                        time TIME,
                        camera TEXT
                    )
                ''')
        print("✅ Table 'attendance_logs' ready.")
    except Exception as e:
        print("❌ Failed to create attendance_logs table:", e)


def log_attendance(name, emp_id, date, time, camera):
    """Insert a new attendance log."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO attendance_logs (name, emp_id, date, time, camera)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (name, emp_id, date, time, camera))
        print(f"🕒 Attendance logged for {name} ({emp_id}) on {date} at {time}.")
    except Exception as e:
        print("❌ Failed to log attendance:", e)


def init_summary_table():
    """Create the daily_summary table if it doesn't exist."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
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
                        UNIQUE (emp_id, date)
                    )
                ''')
        print("✅ Table 'daily_summary' ready.")
    except Exception as e:
        print("❌ Failed to create daily_summary table:", e)


def save_daily_summary(emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status):
    """Insert or update a daily summary record."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
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
                ''', (emp_id, name, date, working_hours, entry_count, exit_count, first_entry, last_exit, status))
        print(f"📊 Daily summary saved for {name} ({emp_id}) on {date}.")
    except Exception as e:
        print("❌ Failed to save daily summary:", e)


def get_daily_summary(date):
    """Fetch all daily summaries for a specific date."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT emp_id, name, date, working_hours,
                           entry_count, exit_count, first_entry, last_exit, status
                    FROM daily_summary
                    WHERE date = %s
                ''', (date,))
                rows = cursor.fetchall()
        summaries = [
            {
                "emp_id": r[0],
                "name": r[1],
                "date": str(r[2]),
                "working_hours": r[3],
                "entry_count": r[4],
                "exit_count": r[5],
                "first_entry": str(r[6]) if r[6] else None,
                "last_exit": str(r[7]) if r[7] else None,
                "status": r[8]
            }
            for r in rows
        ]
        print(f"📅 Retrieved {len(summaries)} summaries for {date}.")
        return summaries
    except Exception as e:
        print("❌ Failed to fetch daily summary:", e)
        return []




def init_attendance_requests_table():
    """Create the attendance_requests table if it doesn't exist."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_requests (
                        id SERIAL PRIMARY KEY,
                        emp_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        request_type TEXT NOT NULL CHECK (request_type IN ('wfh', 'manual_capture')),
                        date DATE NOT NULL,
                        in_time TIME,
                        out_time TIME,
                        reason TEXT,
                        status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
                        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        approved_by TEXT,
                        approved_at TIMESTAMP,
                        remarks TEXT,
                        UNIQUE(emp_id, date, request_type)
                    )
                ''')
                
                # Create index for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_attendance_requests_emp_date 
                    ON attendance_requests(emp_id, date)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_attendance_requests_status 
                    ON attendance_requests(status)
                ''')
                
        print("✅ Table 'attendance_requests' ready.")
    except Exception as e:
        print("❌ Failed to create attendance_requests table:", e)


def create_attendance_request(emp_id, name, request_type, date, in_time=None, out_time=None, reason=None):
    """Insert a new attendance request."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO attendance_requests 
                    (emp_id, name, request_type, date, in_time, out_time, reason, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                    ON CONFLICT (emp_id, date, request_type) 
                    DO UPDATE SET
                        in_time = EXCLUDED.in_time,
                        out_time = EXCLUDED.out_time,
                        reason = EXCLUDED.reason,
                        status = 'pending',
                        requested_at = CURRENT_TIMESTAMP
                    RETURNING id
                ''', (emp_id, name, request_type, date, in_time, out_time, reason))
                
                request_id = cursor.fetchone()[0]
        
        print(f"✅ Attendance request created: ID {request_id} for {name} ({emp_id})")
        return request_id
    except Exception as e:
        print(f"❌ Failed to create attendance request:", e)
        raise




def get_attendance_requests(emp_id=None, status=None, limit=100, offset=0):
    """Fetch attendance requests with optional filters."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                # ---------- Build base query ----------
                base_query = """
                    FROM attendance_requests
                    WHERE 1=1
                """
                params = []

                if emp_id:
                    base_query += " AND emp_id = %s"
                    params.append(emp_id)

                if status:
                    base_query += " AND status = %s"
                    params.append(status)

                # ---------- Fetch records ----------
                data_query = f"""
                    SELECT id, emp_id, name, request_type, date,
                           in_time, out_time, reason, status,
                           requested_at, approved_by, approved_at, remarks
                    {base_query}
                    ORDER BY requested_at DESC
                    LIMIT %s OFFSET %s
                """

                cursor.execute(data_query, params + [limit, offset])
                rows = cursor.fetchall()

                # ---------- Fetch total count ----------
                count_query = f"SELECT COUNT(*) {base_query}"
                cursor.execute(count_query, params)
                total = cursor.fetchone()[0]

        requests = [
            {
                "id": r[0],
                "emp_id": r[1],
                "name": r[2],
                "request_type": r[3],
                "date": str(r[4]),
                "in_time": str(r[5]) if r[5] else None,
                "out_time": str(r[6]) if r[6] else None,
                "reason": r[7],
                "status": r[8],
                "requested_at": str(r[9]),
                "approved_by": r[10],
                "approved_at": str(r[11]) if r[11] else None,
                "remarks": r[12]
            }
            for r in rows
        ]

        return {
            "total": total,
            "requests": requests
        }

    except Exception as e:
        print("❌ Failed to fetch attendance requests:", e)
        return {"total": 0, "requests": []}



def update_request_status(request_id, status, approved_by=None, remarks=None):
    """Update the status of an attendance request."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    UPDATE attendance_requests
                    SET status = %s,
                        approved_by = %s,
                        approved_at = CASE WHEN %s IN ('approved', 'rejected') 
                                      THEN CURRENT_TIMESTAMP ELSE approved_at END,
                        remarks = %s
                    WHERE id = %s
                    RETURNING emp_id, name, date, request_type
                ''', (status, approved_by, status, remarks, request_id))
                
                result = cursor.fetchone()
                
                if not result:
                    return None
                
        print(f"✅ Request {request_id} status updated to '{status}'")
        return {
            "emp_id": result[0],
            "name": result[1],
            "date": str(result[2]),
            "request_type": result[3]
        }
    except Exception as e:
        print(f"❌ Failed to update request status:", e)
        raise


def create_device_tokens_table():
    """Create the device_tokens table for storing FCM tokens."""
    query = """
    CREATE TABLE IF NOT EXISTS device_tokens (
        id SERIAL PRIMARY KEY,
        emp_id VARCHAR(20) NOT NULL,
        fcm_token TEXT NOT NULL,
        platform VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        print("✅ Table 'device_tokens' created successfully!")
    except Exception as e:
        print("❌ Failed to create 'device_tokens' table:", e)
    finally:
        if conn:
            conn.close()

from attendance_db_postgres import get_connection

def init_notification_logs_table():
    """Create the notification_logs table if it doesn't exist."""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notification_logs (
                    id SERIAL PRIMARY KEY,
                    emp_id VARCHAR(20),
                    title TEXT,
                    body TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
                );
            """)
        conn.commit()
        print("✅ Table 'notification_logs' is ready.")
    except Exception as e:
        print("❌ Failed to create 'notification_logs' table:", e)
    finally:
        if conn:
            conn.close()



if __name__ == "__main__":
    # print("🔄 Initializing PostgreSQL tables...")
    # init_db()
    # init_summary_table()
    # print("🔄 Initializing attendance_requests table...")
    # init_attendance_requests_table()
    # print("✅ Attendance requests table initialized successfully!")
    # print("✅ Tables initialized successfully!")
    # create_device_tokens_table()
    init_notification_logs_table()

    # # Optional test data
    # log_attendance("Daljeet Singh", "E001", "2025-10-30", "09:00:00", "Front Camera")
    # save_daily_summary("E001", "Daljeet Singh", "2025-10-30", "8h", 1, 1, "09:00:00", "17:00:00", "Present")
    # get_daily_summary("2025-10-30")
