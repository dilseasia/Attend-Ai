import psycopg2
from datetime import datetime, timedelta, timezone
import requests
import json
import os
from attendance_db_postgres import DB_CONFIG

# ---------------------------
# Install google-auth if not installed:
# pip install google-auth
# ---------------------------
try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
except ImportError:
    print("❌ ERROR: google-auth not installed!")
    print("Run: pip install google-auth")
    exit(1)

# ---------------------------
# FCM v1 CONFIG
# ---------------------------
SERVICE_ACCOUNT_FILE = "firebase-service-account.json"
PROJECT_ID = "faceregister-cc435"


# ---------------------------
# Get OAuth2 Access Token
# ---------------------------
def get_access_token():
    """
    Get OAuth2 access token for FCM HTTP v1 API
    """
    try:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            print(f"\n❌ ERROR: '{SERVICE_ACCOUNT_FILE}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"\n📥 Download it from:")
            print(f"https://console.firebase.google.com/project/{PROJECT_ID}/settings/serviceaccounts/adminsdk")
            print(f"\nSteps:")
            print(f"1. Click 'Generate new private key'")
            print(f"2. Save as '{SERVICE_ACCOUNT_FILE}' in this folder")
            return None
            
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/firebase.messaging"]
        )
        credentials.refresh(Request())
        print(f"✅ OAuth2 token generated")
        return credentials.token
        
    except Exception as e:
        print(f"❌ Error getting access token: {e}")
        return None


# ---------------------------
# Spam protection
# ---------------------------
def can_notify(cursor, emp_id, minutes=2):
    """
    Check if the employee can be notified again (anti-spam).
    Returns True if notification is allowed, False otherwise.
    """
    cursor.execute("""
        SELECT created_at FROM notification_logs
        WHERE emp_id=%s
        ORDER BY created_at DESC
        LIMIT 1
    """, (emp_id,))
    row = cursor.fetchone()
    if not row:
        return True
    
    last_time = row[0]
    now_utc = datetime.now(timezone.utc)
    
    # Ensure timezone-aware comparison
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    
    time_diff = now_utc - last_time
    return time_diff > timedelta(minutes=minutes)


# ---------------------------
# Send FCM Push - HTTP v1 API
# ---------------------------
def send_push_fcm_v1(token, title, body, data=None):
    """
    Send push notification via FCM HTTP v1 API
    Returns True if successful, False otherwise
    """
    access_token = get_access_token()
    if not access_token:
        return False
    
    url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
    
    # All data values must be strings for FCM
    data_payload = {}
    if data:
        data_payload = {k: str(v) for k, v in data.items()}
    
    # Construct FCM message payload
    message = {
        "message": {
            "token": token,
            "notification": {
                "title": title,
                "body": body
            },
            "data": data_payload,
            "android": {
                "priority": "high",
                "notification": {
                    "sound": "default",
                    "click_action": "FLUTTER_NOTIFICATION_CLICK",
                    "channel_id": "attendance_channel"
                }
            },
            "apns": {
                "headers": {
                    "apns-priority": "10"
                },
                "payload": {
                    "aps": {
                        "sound": "default",
                        "badge": 1
                    }
                }
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"📤 Sending to: {token[:40]}...")
        response = requests.post(url, json=message, headers=headers, timeout=10)
        
        print(f"📥 Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Notification sent successfully!")
            return True
        else:
            print(f"⚠️ Failed: {response.status_code}")
            try:
                error_data = response.json()
                error_code = error_data.get('error', {}).get('message', 'Unknown error')
                
                # Handle common FCM errors
                if error_code == "NotRegistered" or "UNREGISTERED" in str(error_data):
                    print(f"❌ Token is invalid/expired. Please generate a new FCM token from your app.")
                elif "PERMISSION_DENIED" in str(error_data):
                    print(f"❌ Permission denied. Enable FCM API at:")
                    print(f"   https://console.cloud.google.com/apis/library/fcm.googleapis.com?project={PROJECT_ID}")
                else:
                    print(f"📋 Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"📋 Response: {response.text[:300]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⚠️ Request timeout - FCM server not responding")
        return False
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return False


# ---------------------------
# Clean up invalid tokens
# ---------------------------
def cleanup_invalid_token(cursor, conn, emp_id, token):
    """
    Remove invalid/expired token from database
    """
    try:
        cursor.execute("""
            DELETE FROM device_tokens 
            WHERE emp_id=%s AND fcm_token=%s
        """, (emp_id, token))
        conn.commit()
        print(f"🗑️ Removed invalid token from database")
    except Exception as e:
        print(f"⚠️ Failed to cleanup token: {e}")


# ---------------------------
# Main trigger
# ---------------------------
def trigger_notification(name, emp_id, date, time, camera, event="ENTRY"):
    """
    Send attendance notification and log it.
    
    Args:
        name: Employee name
        emp_id: Employee ID
        date: Date string (YYYY-MM-DD)
        time: Time string (HH:MM:SS)
        camera: Camera location/name
        event: Event type (ENTRY/EXIT)
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Anti-spam protection
        if not can_notify(cursor, emp_id):
            print(f"⏱️ Notification skipped (spam protection) for {emp_id}")
            return

        # Fetch all device tokens for this employee
        cursor.execute("SELECT fcm_token FROM device_tokens WHERE emp_id=%s", (emp_id,))
        tokens = cursor.fetchall()
        
        if not tokens:
            print(f"⚠️ No device tokens found for employee ID: {emp_id}")
            print(f"💡 User needs to login to the app to register their device")
            return

        # Prepare notification content
        title = f"{event} Recorded 📸"
        body = f"Hi {name}, your {event.lower()} was recorded at {time} ({camera})"
        
        data_payload = {
            "type": "attendance_log",
            "event": event,
            "name": name,
            "emp_id": emp_id,
            "date": date,
            "time": time,
            "camera": camera,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        print(f"\n{'='*60}")
        print(f"📨 Notification for {name} ({emp_id})")
        print(f"📋 {title}")
        print(f"📋 {body}")
        print(f"{'='*60}\n")

        success_count = 0
        failed_tokens = []
        
        # Send to all registered devices
        for idx, t in enumerate(tokens, 1):
            token = t[0]
            print(f"\n--- Token {idx}/{len(tokens)} ---")
            
            if send_push_fcm_v1(token, title, body, data_payload):
                success_count += 1
            else:
                failed_tokens.append(token)

        # Clean up invalid tokens (404 errors)
        if failed_tokens:
            print(f"\n🗑️ Cleaning up {len(failed_tokens)} invalid token(s)...")
            for token in failed_tokens:
                cleanup_invalid_token(cursor, conn, emp_id, token)

        # Log notification in database
        cursor.execute("""
            INSERT INTO notification_logs (emp_id, title, body)
            VALUES (%s, %s, %s)
        """, (emp_id, title, body))
        conn.commit()

        # Summary
        print(f"\n{'='*60}")
        if success_count > 0:
            print(f"✅ SUCCESS: Sent to {success_count}/{len(tokens)} devices")
        else:
            print(f"❌ FAILED: Could not send to any devices")
            print(f"💡 Solution: Generate new FCM tokens from your mobile app")
        print(f"{'='*60}\n")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"🔔 Notification error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()


# ---------------------------
# Test and verify setup
# ---------------------------
if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"🔧 FCM v1 Configuration Check")
    print(f"{'='*60}")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Service Account File: {SERVICE_ACCOUNT_FILE}")
    print(f"File exists: {os.path.exists(SERVICE_ACCOUNT_FILE)}")
    
    # Check if service account file exists
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"\n❌ SETUP REQUIRED:")
        print(f"\n1. Go to: https://console.firebase.google.com/project/{PROJECT_ID}/settings/serviceaccounts/adminsdk")
        print(f"2. Click 'Generate new private key'")
        print(f"3. Save the downloaded JSON as '{SERVICE_ACCOUNT_FILE}'")
        print(f"4. Put it in: {os.getcwd()}")
        print(f"\n5. Install google-auth: pip install google-auth")
        print(f"\nThen run this script again.")
        exit(1)
    
    # Verify service account file
    try:
        with open(SERVICE_ACCOUNT_FILE, 'r') as f:
            sa_data = json.load(f)
            file_project_id = sa_data.get('project_id')
            
            if file_project_id == PROJECT_ID:
                print(f"✅ Service account verified")
                print(f"   Email: {sa_data.get('client_email', 'N/A')}")
            else:
                print(f"⚠️ WARNING: Project ID mismatch!")
                print(f"   File project_id: {file_project_id}")
                print(f"   Expected: {PROJECT_ID}")
                print(f"\n   Update PROJECT_ID in this script to: '{file_project_id}'")
                print(f"   OR download the correct service account for: '{PROJECT_ID}'")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in service account file")
        exit(1)
    except Exception as e:
        print(f"❌ Error reading service account: {e}")
        exit(1)
    
    # Check database connection and tokens
    print(f"\n{'='*60}")
    print(f"📊 Database Tokens Check")
    print(f"{'='*60}")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get tokens for test employee
        cursor.execute("SELECT emp_id, fcm_token FROM device_tokens WHERE emp_id='3561'")
        tokens = cursor.fetchall()
        
        if tokens:
            print(f"✅ Found {len(tokens)} token(s) for emp_id 3561")
            for emp_id, token in tokens:
                print(f"   Token: {token[:50]}... ({len(token)} chars)")
        else:
            print(f"⚠️ No tokens found for emp_id 3561")
            print(f"\n💡 To add a token:")
            print(f"   1. Run your Flutter app and login as employee 3561")
            print(f"   2. Copy the FCM token from app logs")
            print(f"   3. Insert into database:")
            print(f"      INSERT INTO device_tokens (emp_id, fcm_token) VALUES ('3561', 'YOUR_TOKEN');")
        
        conn.close()
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        print(f"\n💡 Check your DB_CONFIG in attendance_db_postgres.py")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)
    
    # Send test notification
    print(f"\n{'='*60}")
    print(f"🧪 SENDING TEST NOTIFICATION")
    print(f"{'='*60}")
    
    test_name = "Anchal"
    test_emp_id = "3561"
    now = datetime.now(timezone.utc)
    
    trigger_notification(
        name=test_name,
        emp_id=test_emp_id,
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"),
        camera="Entry Gate",
        event="ENTRY"
    )
    
    print(f"\n{'='*60}")
    print(f"📱 NEXT STEPS:")
    print(f"{'='*60}")
    print(f"If notification failed with 'NotRegistered' error:")
    print(f"1. Open your Flutter app")
    print(f"2. Login as employee 3561")
    print(f"3. Check app logs for FCM token")
    print(f"4. Update token in database:")
    print(f"   UPDATE device_tokens SET fcm_token='NEW_TOKEN' WHERE emp_id='3561';")
    print(f"\nOr delete old token and app will register new one on next login:")
    print(f"   DELETE FROM device_tokens WHERE emp_id='3561';")
    print(f"{'='*60}\n")