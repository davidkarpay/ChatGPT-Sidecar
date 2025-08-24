#!/bin/bash

# Safe startup script for Sidecar with OpenMP crash prevention
# Specifically designed for Apple Silicon Mac systems

set -e  # Exit on any error

echo "🚀 Starting Sidecar with OpenMP safety measures..."

# Set critical OpenMP environment variables to prevent threading conflicts
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export OMP_WAIT_POLICY=PASSIVE
export OMP_MAX_ACTIVE_LEVELS=1

# Additional PyTorch/Apple Silicon stability settings
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Python settings to reduce memory fragmentation
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "📁 Working directory: $PROJECT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if virtual environment is recommended
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Running outside virtual environment"
    echo "   For better isolation, consider:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
fi

# Display environment variables being used
echo "🔧 OpenMP Settings:"
echo "   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "   TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo ""

# Check if port 8088 is already in use
if lsof -ti:8088 >/dev/null 2>&1; then
    echo "⚠️  Port 8088 is already in use. Attempting to stop existing process..."
    pkill -f "uvicorn.*8088" || true
    sleep 2
fi

# Start the server with proper error handling
echo "🌟 Starting Sidecar server on http://127.0.0.1:8088"
echo "   Use Ctrl+C to stop the server"
echo ""

# Run with proper signal handling
trap 'echo "🛑 Stopping server..."; kill $!; exit 0' INT TERM

uvicorn app.main:app --host 127.0.0.1 --port 8088 --reload --log-level info &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server started successfully
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo "🌐 Open http://127.0.0.1:8088 in your browser"
    echo "💬 Chat interface: http://127.0.0.1:8088/chat"
    echo ""
else
    echo "❌ Server failed to start"
    exit 1
fi

# Keep script running
wait $SERVER_PID