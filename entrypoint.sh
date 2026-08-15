#!/usr/bin/env bash

APP_PORT="${PORT:-10000}"

# Run background workers
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