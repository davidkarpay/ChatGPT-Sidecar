"""Celery tasks for background processing."""

import os
from celery import Celery
from datetime import timedelta

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("sidecar", broker=REDIS_URL, backend=REDIS_URL)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Import tasks from sync_service module
from .sync_service import sync_user_chatgpt_data, scheduled_sync_all_users

# Additional tasks can be defined here

@celery_app.task(bind=True)
def test_task(self, name="World"):
    """Test task to verify Celery is working."""
    return f"Hello {name}!"

# Celery beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Sync ChatGPT data every 24 hours by default
    'sync-chatgpt-data-daily': {
        'task': 'app.sync_service.scheduled_sync_all_users',
        'schedule': timedelta(hours=int(os.getenv("CHATGPT_SYNC_INTERVAL_HOURS", "24"))),
    },
    # Cleanup expired sessions every hour
    'cleanup-expired-sessions': {
        'task': 'app.tasks.cleanup_expired_sessions',
        'schedule': timedelta(hours=1),
    },
}

@celery_app.task
def cleanup_expired_sessions():
    """Clean up expired user sessions."""
    try:
        from .auth import session_manager
        if session_manager:
            session_manager.cleanup_expired_sessions()
            return "Expired sessions cleaned up"
        return "Session manager not available"
    except Exception as e:
        return f"Error cleaning up sessions: {e}"

# Auto-discover tasks from other modules
celery_app.autodiscover_tasks(['app.sync_service'])