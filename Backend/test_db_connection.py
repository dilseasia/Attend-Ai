import psycopg2

DB_CONFIG = {
    "dbname": "attendanceDB",
    "user": "postgres",
    "password": "DALJEET123",
    "host": "localhost",
    "port": "5432"
}

def test_connection():
    """Check PostgreSQL connection"""
    try:
        print("🔄 Connecting to PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print("✅ Connection successful!")
        print("PostgreSQL version:", version[0])

    except Exception as e:
        print("❌ Connection failed:")
        print(e)

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
            print("🔒 Connection closed.")


if __name__ == "__main__":
    test_connection()
