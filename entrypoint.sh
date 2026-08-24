# 1. Run Optimizer synchronously once (blocks until complete, then frees 100% RAM)
if [ -f "optimizer.py" ]; then
    echo "Running initial strategy optimization..."
    python optimizer.py --once || echo "Optimizer warning: proceeding with database defaults."
    
    # Short pause to let system memory settle
    sleep 5
fi

# 2. Launch Main Engine in background
echo "Starting Main Engine..."
python main.py &
PID_MAIN=$!

sleep 5

# 3. Launch WS Monitor in background
if [ -f "ws_monitor.py" ]; then
    echo "Starting WS Monitor..."
    python ws_monitor.py &
    PID_WS=$!
fi