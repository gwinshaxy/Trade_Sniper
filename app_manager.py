import threading
import time
import subprocess
import os
import sys
import requests
import uvicorn
from webhook_engine import app as fastapi_app

def run_script(script_name):
    """Generic loop to run background Python scripts and stop cleanly on Ctrl+C."""
    while True:
        try:
            print(f"Starting background process: {script_name}...")
            result = subprocess.run([sys.executable, script_name])
            
            if result.returncode in [0, 3221225786, -1073741510]:
                print(f"Process {script_name} stopped cleanly or terminated by user.")
                break
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt received for {script_name}. Terminating loop...")
            break
        except Exception as e:
            print(f"Process {script_name} crashed with error: {e}. Restarting in 10 seconds...")
            time.sleep(10)

def run_fastapi_webhook():
    """Runs the FastAPI webhook engine internally on designated WEBHOOK_PORT."""
    try:
        internal_port = int(os.environ.get("WEBHOOK_PORT", 8080))
        print(f"Starting internal FastAPI webhook engine on port {internal_port}...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=internal_port, log_level="info")
    except Exception as e:
        print(f"FastAPI webhook engine crashed: {e}")

def keep_alive():
    """Pings the web service URL every 10 minutes to prevent free tier spin-down."""
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
    """Spawns core background engines and keep-alive thread safely without duplicates."""
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