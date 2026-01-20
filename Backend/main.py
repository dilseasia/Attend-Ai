from multiprocessing import Process
import subprocess
import time

def run_entry_camera():
    subprocess.run(["python3", "entry.py"])

def run_exit_camera():
    subprocess.run(["python3", "exit.py"])

def run_entry_vehicle():
    subprocess.run(["python3", "entry_vehicle.py"])

def run_exit_vehicle():
    subprocess.run(["python3", "exit_vehicle.py"])

def run_backend():
    # Start FastAPI with uvicorn
    subprocess.run(["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"])


if __name__ == "__main__":
    print("▶️ Starting Entry/Exit cameras, vehicles, and backend... - main.py:23")

    # Camera processes
    entry_camera_process = Process(target=run_entry_camera)
    exit_camera_process = Process(target=run_exit_camera)

    # Vehicle processes
    entry_vehicle_process = Process(target=run_entry_vehicle)
    exit_vehicle_process = Process(target=run_exit_vehicle)

    # Backend process
    backend_process = Process(target=run_backend)

    # Start all processes
    entry_camera_process.start()
    exit_camera_process.start()
    entry_vehicle_process.start()
    exit_vehicle_process.start()
    backend_process.start()

    try:
        while True:
            time.sleep(1)  # Keep main process alive
    except KeyboardInterrupt:
        print("\n🛑 Terminating all processes... - main.py:47")

        # Terminate all processes
        entry_camera_process.terminate()
        exit_camera_process.terminate()
        entry_vehicle_process.terminate()
        exit_vehicle_process.terminate()
        backend_process.terminate()

        # Wait for all processes to finish
        entry_camera_process.join()
        exit_camera_process.join()
        entry_vehicle_process.join()
        exit_vehicle_process.join()
        backend_process.join()

        print("✅ Shutdown complete. - main.py:63")
