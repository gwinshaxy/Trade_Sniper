#!/usr/bin/env bash

APP_PORT="${PORT:-10000}"

# Start lightweight inline FastAPI server on port 8080 for Render health checks
python -c "
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def health_check():
    return {'status': 'ok', 'service': 'trade-sniper-dashboard'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, log_level='warning')
" &

# Disable heavy processes temporarily to stay under 512MB RAM
python worker.py &
python price_monitor.py &
python webhook_engine.py &

# Launch Streamlit bound to Render's assigned port
exec streamlit run dashboard.py \
    --server.port="$APP_PORT" \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.headless=true