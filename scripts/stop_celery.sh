#!/bin/bash

# Stop Celery worker and beat scheduler for Sidecar

set -e

echo "Stopping Celery services..."

# Stop Celery worker
if [ -f "celery_worker.pid" ]; then
    echo "Stopping Celery worker..."
    kill -TERM $(cat celery_worker.pid) 2>/dev/null || echo "Worker already stopped"
    rm -f celery_worker.pid
else
    echo "No Celery worker PID file found"
fi

# Stop Celery beat
if [ -f "celery_beat.pid" ]; then
    echo "Stopping Celery beat scheduler..."
    kill -TERM $(cat celery_beat.pid) 2>/dev/null || echo "Beat already stopped"
    rm -f celery_beat.pid
else
    echo "No Celery beat PID file found"
fi

# Clean up schedule file
rm -f celerybeat-schedule

echo "Celery services stopped!"