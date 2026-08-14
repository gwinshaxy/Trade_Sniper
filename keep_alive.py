import os
import time
import requests

def keep_alive():
    """Pings the web service URL every 10 minutes to prevent free tier spin-down."""
    app_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not app_url:
        print("RENDER_EXTERNAL_URL not set. Self-ping keep-alive disabled.")
        # Idle indefinitely so Supervisor doesn't repeatedly restart this process
        while True:
            time.sleep(3600)
        
    while True:
        try:
            time.sleep(600)
            response = requests.get(app_url, timeout=10)
            print(f"Keep-alive ping sent to {app_url}, status: {response.status_code}")
        except Exception as e:
            print(f"Keep-alive ping failed: {e}")

if __name__ == "__main__":
    keep_alive()