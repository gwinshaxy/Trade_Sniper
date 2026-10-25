#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=128000

LOCKDIR="/tmp/bybit_futures_entrypoint.lock"

# Check if lock exists and auto-clean if stale
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    if [ -f "$LOCKDIR/pid" ]; then
        PID=$(cat "$LOCKDIR/pid")
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "⚠️ Found stale lock from inactive PID $PID. Cleaning up..."
            rm -rf "$LOCKDIR"
            mkdir "$LOCKDIR"
        else
            echo "❌ Error: Another instance of entrypoint.sh (PID $PID) is already running! Exiting..."
            exit 1
        fi
    else
        echo "⚠️ Lock directory existed without PID file. Cleaning up..."
        rm -rf "$LOCKDIR"
        mkdir "$LOCKDIR"
    fi
fi

# Store active process ID inside lock folder
echo $$ > "$LOCKDIR/pid"

cleanup() {
    echo "SIGTERM/SIGINT received. Shutting down active background processes gracefully..."
    kill -TERM "$PID_HEALTH" "$PID_MAIN" "$PID_VETTING" 2>/dev/null || true
    wait "$PID_HEALTH" "$PID_MAIN" "$PID_VETTING" 2>/dev/null || true
    rm -rf "$LOCKDIR"
    echo "All processes terminated cleanly."
    exit 0
}

trap cleanup EXIT SIGTERM SIGINT

echo "=================================================="
echo " Starting Event-Driven Bybit Futures Trading Engine"
echo "=================================================="

if [ -f "health_server.py" ]; then
    echo "Launching Standalone Health Check Server..."
    python health_server.py &
    PID_HEALTH=$!
    sleep 2
else
    echo "⚠️ Warning: health_server.py not found. PaaS health checks may fail!"
fi

echo "Verifying and updating database schema..."
python -c "from common import ensure_schema_updated; ensure_schema_updated()"

# 1. Initial Vetting Scan on Startup (if file doesn't exist yet)
if [ ! -f "vetted_symbols.json" ]; then
    echo "🔍 No existing vetted_symbols.json found. Running initial asset vetting scan..."
    python dynamic_asset_filter.py || echo "⚠️ Initial asset vetting encountered an issue, proceeding with fallbacks."
else
    echo "📁 Existing vetted_symbols.json detected. Skipping startup vetting block."
fi

# 2. Launch Main Orchestrator Engine in the background
echo "Launching Main Orchestrator Engine..."
python main.py &
PID_MAIN=$!

# 3. Background 24-Hour Recurring Asset Vetting Loop
echo "Initializing 24-hour background asset re-vetting daemon..."
(
    while true; do
        sleep 86400
        echo "⏰ [24h Scheduler] Starting scheduled daily asset re-vetting scan..."
        python dynamic_asset_filter.py || echo "⚠️ Daily asset vetting sweep failed. Retrying in next cycle."
    done
) &
PID_VETTING=$!

echo "Services running: Health Check [${PID_HEALTH:-N/A}] | Main Engine [${PID_MAIN:-N/A}] | Vetting Daemon [${PID_VETTING:-N/A}]"

while true; do
    if [ -n "$PID_HEALTH" ] && ! kill -0 "$PID_HEALTH" 2>/dev/null; then
        echo "⚠️ Health Check Server (PID $PID_HEALTH) exited unexpectedly!"
        break
    fi
    if [ -n "$PID_MAIN" ] && ! kill -0 "$PID_MAIN" 2>/dev/null; then
        echo "⚠️ Main Engine (PID $PID_MAIN) exited unexpectedly!"
        break
    fi
    sleep 5
done

echo "Triggering system cleanup..."
cleanup