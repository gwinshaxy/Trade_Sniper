#!/bin/bash
set -e

cleanup() {
    echo "SIGTERM/SIGINT received. Shutting down active background processes gracefully..."
    kill -TERM $PID_MAIN $PID_WS $PID_OPT 2>/dev/null
    wait
    echo "All processes terminated cleanly."
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "=================================================="
echo " Starting Decoupled Live Trading Engine System"
echo "=================================================="

# Run schema update check prior to process startup
python -c "from common import ensure_schema_updated; ensure_schema_updated()"

# 1. Launch Strategy Optimizer FIRST
if [ -f "optimizer.py" ]; then
    echo "Launching Optimizer first..."
    python optimizer.py &
    PID_OPT=$!
    
    # Wait 30 seconds for optimizer to finish initial kline fetching/population setup
    echo "Waiting 30 seconds before starting main services..."
    sleep 30
fi

# 2. Launch Primary Engine Loop
echo "Launching Main Engine..."
python main.py &
PID_MAIN=$!

# Wait 5 seconds before starting WS Monitor
sleep 5

# 3. Launch Real-time WebSocket Price Monitor
if [ -f "ws_monitor.py" ]; then
    echo "Launching WS Monitor..."
    python ws_monitor.py &
    PID_WS=$!
fi

echo "Services running: Main Engine [$PID_MAIN] | WS Monitor [${PID_WS:-N/A}] | Optimizer [${PID_OPT:-N/A}]"

# Monitor core background process health
while true; do
    if ! kill -0 $PID_MAIN 2>/dev/null; then
        echo "⚠️ Main Engine (PID $PID_MAIN) exited unexpectedly!"
        break
    fi
    if [ -n "$PID_WS" ] && ! kill -0 $PID_WS 2>/dev/null; then
        echo "⚠️ WS Monitor (PID $PID_WS) exited unexpectedly!"
        break
    fi
    if [ -n "$PID_OPT" ] && ! kill -0 $PID_OPT 2>/dev/null; then
        echo "⚠️ Optimizer (PID $PID_OPT) exited unexpectedly!"
        break
    fi
    sleep 5
done

echo "Triggering system cleanup..."
cleanup