#!/usr/bin/env bash

# Start background services in the background without inheriting web PORT
unset PORT
python worker.py &
python price_monitor.py &
python optimizer.py &
python webhook_engine.py &

# Start main Streamlit app on assigned Render port
export PORT=${PORT:-8000}
exec streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false