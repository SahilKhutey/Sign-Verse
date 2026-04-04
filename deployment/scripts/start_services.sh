#!/bin/bash
# Start all SignVerse services locally

set -e

echo "Starting SignVerse services..."

# Activate virtual environment
source venv/bin/activate

# Start AI Engine (port 8000)
echo "Starting AI Engine on port 8000..."
uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000 &
AI_PID=$!

# Start Backend Server (port 8001)
echo "Starting Backend on port 8001..."
uvicorn backend.app:app --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# Start Avatar Animation API (port 8002)
echo "Starting Avatar API on port 8002..."
uvicorn avatar_animation.unity_bridge.animation_api:app --host 0.0.0.0 --port 8002 &
AVATAR_PID=$!

echo ""
echo "All services started!"
echo "  AI Engine:  http://localhost:8000"
echo "  Backend:    http://localhost:8001"
echo "  Avatar API: http://localhost:8002"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait and handle shutdown
trap "kill $AI_PID $BACKEND_PID $AVATAR_PID 2>/dev/null; exit" INT TERM
wait
