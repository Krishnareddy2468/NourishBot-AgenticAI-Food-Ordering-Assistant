"""
Notification worker — proactive AI nudges (Sprint 3).

Currently a skeleton. Will be populated in Sprint 3 with:
  - evaluate_and_send_proactive_notifications()
  - send_monthly_savings_report()
  - send_weekly_nutrition_summary()

Placeholder task for health-check purposes.
"""

import logging
import asyncio
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.notification_worker.ping")
def ping() -> str:
    """Health-check ping for Celery worker."""
    return "pong"


@celery_app.task(
    name="app.workers.notification_worker.process_whatsapp_message",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=4,
)
def process_whatsapp_message(wa_id: str, user_name: str, text: str) -> bool:
    """Process a verified WhatsApp event outside the webhook request process."""
    from app.api.routes.whatsapp import _process_message

    asyncio.run(_process_message(wa_id, user_name, text))
    return True
