#!/bin/bash
# Start AeroInspectAI FastAPI Backend

cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "AEROINSPECT AI — FASTAPI BACKEND LAUNCHER"
echo "============================================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found"
    echo "Creating .env from .env.example"
    cp ".env.example" ".env"
fi

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "WARNING: Virtual environment not found at ../.venv"
    echo "Please create and activate a virtual environment first"
    echo ""
    echo "Usage:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
fi

# Start the backend
echo "Starting backend on http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
