#!/usr/bin/env bash

# Store Render's assigned port (defaults to 10000 if not set)
APP_PORT="${PORT:-10000}"

# Launch background microservices
python worker.py &
python price_monitor.py &
python optimizer.py &
python webhook_engine.py &

# Start Streamlit directly bound to Render's assigned port
exec streamlit run dashboard.py \
    --server.port="$APP_PORT" \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.headless=true