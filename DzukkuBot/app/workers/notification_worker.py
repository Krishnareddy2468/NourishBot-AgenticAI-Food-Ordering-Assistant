"""
Notification worker — proactive AI nudges (Sprint 3).

Currently a skeleton. Will be populated in Sprint 3 with:
  - evaluate_and_send_proactive_notifications()
  - send_monthly_savings_report()
  - send_weekly_nutrition_summary()

Placeholder task for health-check purposes.
"""

import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.notification_worker.ping")
def ping() -> str:
    """Health-check ping for Celery worker."""
    return "pong"
