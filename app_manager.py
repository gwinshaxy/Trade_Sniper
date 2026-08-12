import threading
import time
import subprocess
import os
import requests
import uvicorn
from webhook_engine import app as fastapi_app

def run_script(script_name):
    """Generic loop to run background Python scripts and auto-restart them on failure."""
    while True:
        try:
            print(f"Starting background process: {script_name}...")
            subprocess.run(["python", script_name], check=True)
        except Exception as e:
            print(f"Process {script_name} crashed with error: {e}. Restarting in 10 seconds...")
            time.sleep(10)

def run_fastapi_webhook():
    """Runs the FastAPI webhook engine internally on port 8000 (or mapped port)."""
    try:
        print("Starting internal FastAPI webhook engine...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"FastAPI webhook engine crashed: {e}")

def keep_alive():
    """Pings the Render web service URL every 10 minutes to prevent free tier spin-down."""
    app_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not app_url:
        print("RENDER_EXTERNAL_URL not set. Self-ping keep-alive disabled.")
        return
        
    while True:
        try:
            time.sleep(600)
            response = requests.get(app_url, timeout=10)
            print(f"Keep-alive ping sent to {app_url}, status: {response.status_code}")
        except Exception as e:
            print(f"Keep-alive ping failed: {e}")

def start_background_tasks():
    """Spawns all core background engines and the keep-alive thread safely."""
    if os.environ.get("BG_TASKS_STARTED") == "true":
        return
    
    os.environ["BG_TASKS_STARTED"] = "true"
    
    worker_thread = threading.Thread(target=run_script, args=("worker.py",), daemon=True)
    worker_thread.start()
    
    monitor_thread = threading.Thread(target=run_script, args=("price_monitor.py",), daemon=True)
    monitor_thread.start()
    
    optimizer_thread = threading.Thread(target=run_script, args=("optimizer.py",), daemon=True)
    optimizer_thread.start()
    
    webhook_thread = threading.Thread(target=run_fastapi_webhook, daemon=True)
    webhook_thread.start()
    
    ping_thread = threading.Thread(target=keep_alive, daemon=True)
    ping_thread.start()
    
    print("All backend scripts (worker, price_monitor, optimizer, webhook, keep-alive) successfully initialized.")