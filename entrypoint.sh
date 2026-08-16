#!/usr/bin/env bash
set -e

# Render dynamic port assignment
export PORT="${PORT:-10000}"
export WEBHOOK_PORT="${WEBHOOK_PORT:-8080}"

# Start background workers cleanly in single background jobs
PYTHONUNBUFFERED=1 python worker.py &
PYTHONUNBUFFERED=1 python price_monitor.py &
PYTHONUNBUFFERED=1 python webhook_engine.py &

# Launch primary Streamlit service bound to primary PORT
exec streamlit run dashboard.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.headless=true