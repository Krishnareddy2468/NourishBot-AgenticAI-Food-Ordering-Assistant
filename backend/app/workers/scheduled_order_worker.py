"""Activate NourishBot orders shortly before their scheduled meal time."""

from datetime import datetime, timedelta, timezone
import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Order, OutboxEvent
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.scheduled_order_worker.activate_due_orders")
def activate_due_orders() -> dict:
    """Move due in-house orders into the live queue; external MCP orders stay provider-managed."""
    if not settings.DATABASE_URL_SYNC:
        logger.error("Scheduled activation skipped: DATABASE_URL_SYNC is not configured")
        return {"activated": 0, "error": "DATABASE_URL_SYNC is not configured"}

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(minutes=settings.SCHEDULE_ACTIVATION_LEAD_MINUTES)
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)
    activated_refs: list[str] = []

    try:
        with Session(engine) as session:
            orders = session.execute(
                select(Order)
                .where(
                    Order.status == "SCHEDULED",
                    Order.provider == "NOURISHBOT",
                    Order.scheduled_for.is_not(None),
                    Order.scheduled_for <= cutoff,
                )
                .with_for_update(skip_locked=True)
            ).scalars().all()

            for order in orders:
                order.status = "CREATED"
                order.activated_at = now
                activated_refs.append(order.order_ref)
                session.add(OutboxEvent(
                    restaurant_id=order.restaurant_id,
                    event_type="order.status_changed",
                    payload={
                        "order_id": order.id,
                        "order_ref": order.order_ref,
                        "new_status": "CREATED",
                        "activated_from_schedule": True,
                        "order_type": order.order_type,
                    },
                ))
            session.commit()
    finally:
        engine.dispose()

    if activated_refs:
        logger.info("Activated %d scheduled orders: %s", len(activated_refs), activated_refs)
    return {"activated": len(activated_refs), "order_refs": activated_refs}
