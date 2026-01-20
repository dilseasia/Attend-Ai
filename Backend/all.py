import subprocess
import time
import shutil

def run_command(cmd):
    """
    Run command in a new terminal window safely.
    """
    print(f"Launching new terminal for: {cmd}")

    # Prefer xterm or konsole to avoid broken GNOME Terminal
    if shutil.which("xterm"):
        subprocess.Popen(['xterm', '-e', f'{cmd}; bash'])
    elif shutil.which("konsole"):
        subprocess.Popen(['konsole', '-e', f'bash -c "{cmd}; exec bash"'])
    else:
        print("⚠️ No terminal emulator found — running directly.")
        subprocess.Popen(cmd, shell=True)

# --- Frontend ---
frontend_cmd = "cd frontend && npm install && npm run dev"
run_command(frontend_cmd)

# --- Small delay before backend ---
time.sleep(5)

# --- Backend ---
backend_cmd = (
    "source myenv/bin/activate && "
    "python -m uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload & "
    "sleep 5; python main.py"
)
run_command(backend_cmd)
