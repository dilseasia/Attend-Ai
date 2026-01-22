from multiprocessing import Process
import subprocess
import time
import sys
import signal
from datetime import datetime, time as dt_time, timedelta
import os
import shutil

# === Automatic cleanup of old anonymous images ===
ANONYMOUS_DIR = "Anonymous"
DAYS_TO_KEEP = 30  # keep last 30 days

def delete_old_anonymous_images():
    """Delete anonymous image folders older than DAYS_TO_KEEP"""
    if not os.path.exists(ANONYMOUS_DIR):
        return
    cutoff = datetime.now() - timedelta(days=DAYS_TO_KEEP)
    for date_folder in os.listdir(ANONYMOUS_DIR):
        date_path = os.path.join(ANONYMOUS_DIR, date_folder)
        if not os.path.isdir(date_path):
            continue
        try:
            folder_date = datetime.strptime(date_folder, "%Y-%m-%d")
        except ValueError:
            continue
        if folder_date < cutoff:
            shutil.rmtree(date_path)
            print(f"🗑️ Deleted old anonymous folder: {date_path} - main.py")

def run_with_restart(script_name, description):
    """Run a script and restart it if it crashes"""
    while True:
        try:
            print(f"🚀 Starting {description}... - main.py:12")
            result = subprocess.run(
                ["python3", script_name],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} crashed: {e} - main.py:20")
            print(f"stderr: {e.stderr} - main.py:21")
            print(f"🔄 Restarting {description} in 3 seconds... - main.py:22")
            time.sleep(3)
        except Exception as e:
            print(f"❌ Unexpected error in {description}: {e} - main.py:25")
            time.sleep(3)

def run_entry_camera():
    run_with_restart("entry.py", "Entry Camera")

def run_exit_camera():
    run_with_restart("exit.py", "Exit Camera")

def run_entry_vehicle():
    run_with_restart("entry_vehicle.py", "Entry Vehicle")

def run_exit_vehicle():
    run_with_restart("exit_vehicle.py", "Exit Vehicle")

def run_backend():
    while True:
        try:
            print("🚀 Starting FastAPI backend... - main.py:43")
            subprocess.run(
                ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"],
                check=True
            )
        except Exception as e:
            print(f"❌ Backend crashed: {e} - main.py:49")
            print("🔄 Restarting backend in 3 seconds... - main.py:50")
            time.sleep(3)

# Global list to track processes
processes = []
processes_running = False
last_cleanup_date = None

def is_within_operating_hours():
    """Check if current time is between 5:00 AM and 11:59 PM"""
    now = datetime.now().time()
    start_time = dt_time(5, 0)  # 5:00 AM
    end_time = dt_time(23, 59)  # 11:59 PM
    return start_time <= now <= end_time

def run_daily_cleanup():
    """Run cleanup once per day"""
    global last_cleanup_date
    today = datetime.now().date()
    
    if last_cleanup_date != today:
        print(f"🧹 Running daily cleanup... - main.py")
        delete_old_anonymous_images()
        last_cleanup_date = today

def stop_all_processes():
    """Stop all running processes"""
    global processes, processes_running
    
    if not processes_running:
        return
    
    print("\n🛑 Stopping all processes... - main.py:71")
    for p in processes:
        if p.is_alive():
            p.terminate()
    
    # Wait for graceful shutdown
    for p in processes:
        p.join(timeout=5)
    
    # Force kill if still alive
    for p in processes:
        if p.is_alive():
            print(f"⚠️ Force killing {p.name}... - main.py:83")
            p.kill()
    
    processes = []
    processes_running = False
    print("✅ All processes stopped. - main.py:88")

def start_all_processes():
    """Start all processes"""
    global processes, processes_running
    
    if processes_running:
        return
    
    print("\n▶️ Starting all processes... - main.py:97")
    
    # Create all processes
    processes = [
        Process(target=run_entry_camera, name="EntryCamera"),
        Process(target=run_exit_camera, name="ExitCamera"),
        # Process(target=run_entry_vehicle, name="EntryVehicle"),
        Process(target=run_exit_vehicle, name="ExitVehicle"),
        Process(target=run_backend, name="Backend")
    ]

    # Start all processes
    for p in processes:
        p.start()
        print(f"✅ Started process: {p.name} - main.py:111")
    
    processes_running = True

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Received shutdown signal, terminating all processes... - main.py:117")
    stop_all_processes()
    print("✅ Shutdown complete. - main.py:119")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🕐 Timebased scheduler initialized - main.py:127")
    print("⏰ Operating hours: 5:00 AM  11:59 PM - main.py:128")
    
    # Run initial cleanup
    delete_old_anonymous_images()
    
    # Main scheduler loop
    try:
        while True:
            current_time = datetime.now().time()
            should_run = is_within_operating_hours()
            
            # Run daily cleanup
            run_daily_cleanup()
            
            if should_run and not processes_running:
                print(f"⏰ [{datetime.now().strftime('%H:%M:%S')}] Within operating hours, starting processes... - main.py:137")
                start_all_processes()
            elif not should_run and processes_running:
                print(f"⏰ [{datetime.now().strftime('%H:%M:%S')}] Outside operating hours, stopping processes... - main.py:140")
                stop_all_processes()
            
            # Monitor processes if running
            if processes_running:
                for p in processes:
                    if not p.is_alive():
                        print(f"⚠️ Process {p.name} died unexpectedly! Exit code: {p.exitcode} - main.py:147")
                        # Remove dead process and restart all
                        print("🔄 Restarting all processes... - main.py:149")
                        stop_all_processes()
                        if is_within_operating_hours():
                            time.sleep(3)
                            start_all_processes()
                        break
            
            # Check every 30 seconds
            time.sleep(30)
            
    except KeyboardInterrupt:
        signal_handler(None, None)