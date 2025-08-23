#!/bin/bash

# Start Celery worker and beat scheduler for Sidecar
# This script is for development use

set -e

# Source virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Warning: Redis is not running. Please start Redis first:"
    echo "  brew services start redis  # on macOS"
    echo "  sudo systemctl start redis  # on Linux"
    exit 1
fi

echo "Starting Celery worker and beat scheduler..."

# Start Celery worker in background
echo "Starting Celery worker..."
celery -A app.tasks worker --loglevel=info --detach --pidfile=celery_worker.pid --logfile=celery_worker.log

# Start Celery beat scheduler in background
echo "Starting Celery beat scheduler..."
celery -A app.tasks beat --loglevel=info --detach --pidfile=celery_beat.pid --logfile=celery_beat.log --schedule=celerybeat-schedule

echo "Celery services started successfully!"
echo "- Worker PID file: celery_worker.pid"
echo "- Beat PID file: celery_beat.pid" 
echo "- Worker log: celery_worker.log"
echo "- Beat log: celery_beat.log"
echo ""
echo "To stop Celery services, run: ./scripts/stop_celery.sh"
echo "To monitor tasks, run: celery -A app.tasks flower"