"""
Celery app — async task queue for Dzukku.

Uses Redis as both broker and result backend.

Beat schedule is empty initially — populated as features come online
(Sprint 3: proactive notifications, Sprint 5: B2B reports).
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "dzukku",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.notification_worker"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,
)

celery_app.conf.beat_schedule = {}
