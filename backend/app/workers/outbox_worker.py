"""Replica-safe leased outbox delivery for realtime and customer events."""

import asyncio
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from app.core.config import settings
from app.db.models import OutboxEvent
from app.db.session import AsyncSessionLocal
from app.realtime.ws_manager import ws_manager

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0


def _retry_delay(attempt_count: int) -> int:
    return min(
        settings.OUTBOX_RETRY_MAX_SECONDS,
        settings.OUTBOX_RETRY_BASE_SECONDS * (2 ** max(0, attempt_count - 1)),
    )


async def _claim_batch(owner: str, now: datetime) -> list[int]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OutboxEvent)
            .where(
                OutboxEvent.processed_at.is_(None),
                OutboxEvent.dead_lettered_at.is_(None),
                or_(OutboxEvent.next_retry_at.is_(None), OutboxEvent.next_retry_at <= now),
                or_(OutboxEvent.lease_expires_at.is_(None), OutboxEvent.lease_expires_at <= now),
            )
            .order_by(OutboxEvent.created_at, OutboxEvent.id)
            .limit(settings.OUTBOX_BATCH_SIZE)
            .with_for_update(skip_locked=True)
        )
        events = result.scalars().all()
        for event in events:
            event.lease_owner = owner
            event.lease_expires_at = now + timedelta(seconds=settings.OUTBOX_LEASE_SECONDS)
            event.attempt_count += 1
            event.max_attempts = event.max_attempts or settings.OUTBOX_MAX_ATTEMPTS
            event.next_retry_at = None
        await session.commit()
        return [event.id for event in events]


async def _deliver_claimed(event_id: int, owner: str) -> bool:
    async with AsyncSessionLocal() as session:
        event = (await session.execute(
            select(OutboxEvent).where(
                OutboxEvent.id == event_id,
                OutboxEvent.processed_at.is_(None),
                OutboxEvent.lease_owner == owner,
            )
        )).scalar_one_or_none()
        if event is None:
            return False
        restaurant_id = event.restaurant_id
        event_type = event.event_type
        payload = dict(event.payload or {})

    try:
        await ws_manager.broadcast(
            restaurant_id=restaurant_id,
            event={"event_type": event_type, "payload": payload, "event_id": event_id},
        )
        if event_type == "order.status_changed" and payload.get("activated_from_schedule"):
            from app.realtime.notifications import notify_order_status

            delivered = await notify_order_status(
                payload["order_ref"],
                payload["new_status"],
                payload.get("order_type", "PICKUP"),
            )
            if not delivered:
                raise RuntimeError("Customer notification did not confirm delivery")
    except Exception as exc:
        async with AsyncSessionLocal() as session:
            event = (await session.execute(
                select(OutboxEvent)
                .where(OutboxEvent.id == event_id, OutboxEvent.lease_owner == owner)
                .with_for_update()
            )).scalar_one_or_none()
            if event is None:
                return False
            now = datetime.now(timezone.utc)
            event.last_error = (str(exc) or exc.__class__.__name__)[:1000]
            event.lease_owner = None
            event.lease_expires_at = None
            if event.attempt_count >= event.max_attempts:
                event.dead_lettered_at = now
            else:
                event.next_retry_at = now + timedelta(seconds=_retry_delay(event.attempt_count))
            await session.commit()
        logger.error("Failed to process outbox event %d: %s", event_id, exc)
        return False

    async with AsyncSessionLocal() as session:
        event = (await session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.id == event_id, OutboxEvent.lease_owner == owner)
            .with_for_update()
        )).scalar_one_or_none()
        if event is None:
            return False
        event.processed_at = datetime.now(timezone.utc)
        event.lease_owner = None
        event.lease_expires_at = None
        event.last_error = None
        await session.commit()
    return True


async def drain_outbox() -> int:
    """Lease and deliver one batch; safe to run concurrently in many replicas."""
    owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
    event_ids = await _claim_batch(owner, datetime.now(timezone.utc))
    semaphore = asyncio.Semaphore(settings.OUTBOX_CONCURRENCY)

    async def deliver(event_id: int) -> bool:
        async with semaphore:
            return await _deliver_claimed(event_id, owner)

    results = await asyncio.gather(*(deliver(event_id) for event_id in event_ids))
    return sum(1 for delivered in results if delivered)


async def outbox_worker_loop() -> None:
    logger.info("Outbox worker started (poll interval: %ss)", POLL_INTERVAL_S)
    while True:
        try:
            count = await drain_outbox()
            if count:
                logger.debug("Outbox: processed %d events", count)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Outbox worker error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL_S)
