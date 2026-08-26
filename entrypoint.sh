#!/bin/bash
set -e

cleanup() {
    echo "SIGTERM/SIGINT received. Shutting down active background processes gracefully..."
    kill -TERM $PID_HEALTH $PID_MAIN $PID_WS 2>/dev/null
    wait
    echo "All processes terminated cleanly."
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "=================================================="
echo " Starting Decoupled Live Trading Engine System"
echo "=================================================="

# 1. Start Standalone Health Check Server IMMEDIATELY to satisfy Render's port check
if [ -f "health_server.py" ]; then
    echo "Launching Standalone Health Check Server..."
    python health_server.py &
    PID_HEALTH=$!
    sleep 2
fi

# 2. Run schema update check prior to process startup
python -c "from common import ensure_schema_updated; ensure_schema_updated()"

# 3. Run Strategy Optimizer synchronously ONCE (blocks until complete, then frees 100% RAM)
if [ -f "optimizer.py" ]; then
    echo "Running initial strategy optimization (--once)..."
    python optimizer.py --once || echo "Optimizer warning: proceeding with database defaults."
    echo "Optimizer completed. Memory released."
    sleep 3
fi

# 4. Launch Primary Engine Loop in background
echo "Launching Main Engine..."
python main.py &
PID_MAIN=$!

# Short delay to allow Main Engine to connect to DB/Exchange
sleep 5

# 5. Launch Real-time WebSocket Price Monitor in background
if [ -f "ws_monitor.py" ]; then
    echo "Launching WS Monitor..."
    python ws_monitor.py &
    PID_WS=$!
fi

echo "Services running: Health Check [$PID_HEALTH] | Main Engine [$PID_MAIN] | WS Monitor [${PID_WS:-N/A}]"

# Monitor core background process health
while true; do
    if ! kill -0 $PID_HEALTH 2>/dev/null; then
        echo "⚠️ Health Check Server (PID $PID_HEALTH) exited unexpectedly!"
        break
    fi
    if ! kill -0 $PID_MAIN 2>/dev/null; then
        echo "⚠️ Main Engine (PID $PID_MAIN) exited unexpectedly!"
        break
    fi
    if [ -n "$PID_WS" ] && ! kill -0 $PID_WS 2>/dev/null; then
        echo "⚠️ WS Monitor (PID $PID_WS) exited unexpectedly!"
        break
    fi
    sleep 5
done

echo "Triggering system cleanup..."
cleanup