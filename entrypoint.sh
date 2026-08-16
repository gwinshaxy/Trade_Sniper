#!/usr/bin/env bash
set -e

# Render dynamic port assignment for the webhook engine
export WEBHOOK_PORT="${PORT:-8080}"

# Start background processes
PYTHONUNBUFFERED=1 python worker.py &
PYTHONUNBUFFERED=1 python price_monitor.py &
PYTHONUNBUFFERED=1 python optimizer.py &

# Run webhook engine as the primary foreground process (keeps container alive)
exec uvicorn webhook_engine:app --host 0.0.0.0 --port "$WEBHOOK_PORT"