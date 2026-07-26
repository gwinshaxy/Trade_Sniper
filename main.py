import subprocess
import sys
import time

print("🚀 Starting Trade Sniper Agent in background...")
# Launch agent.py in a separate background process
agent_process = subprocess.Popen([sys.executable, "agent.py"])

# Give the agent a few seconds to initialize
time.sleep(3)

print("📊 Starting Streamlit Dashboard...")
# Launch app.py on port 10000 (Render's default public web port)
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port=10000",
    "--server.address=0.0.0.0",
    "--server.headless=true"
])