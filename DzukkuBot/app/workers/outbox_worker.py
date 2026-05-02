"""
Outbox worker — drains outbox_events and pushes to the WS/SSE hub.

Runs as a background asyncio task alongside the FastAPI app.
Each cycle:
  1. SELECT unprocessed events (oldest first, LIMIT batch_size)
  2. Publish each to ws_manager
  3. Mark as processed (processed_at = now())
"""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select, update

from app.db.session import AsyncSessionLocal
from app.db.models import OutboxEvent
from app.realtime.ws_manager import ws_manager

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
POLL_INTERVAL_S = 1.0


async def drain_outbox() -> int:
    """Process one batch of unprocessed outbox events. Returns count processed."""
    processed = 0

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.processed_at.is_(None))
            .order_by(OutboxEvent.created_at)
            .limit(BATCH_SIZE)
        )
        events = result.scalars().all()

        for event in events:
            try:
                await ws_manager.broadcast(
                    restaurant_id=event.restaurant_id,
                    event={
                        "event_type": event.event_type,
                        "payload": event.payload,
                        "event_id": event.id,
                    },
                )
                event.processed_at = datetime.now(timezone.utc)
                processed += 1
            except Exception as e:
                logger.error("Failed to process outbox event %d: %s", event.id, e)

        if events:
            await session.commit()

    return processed


async def outbox_worker_loop() -> None:
    """Long-running loop that drains the outbox every POLL_INTERVAL_S seconds."""
    logger.info("Outbox worker started (poll interval: %ss)", POLL_INTERVAL_S)
    while True:
        try:
            count = await drain_outbox()
            if count:
                logger.debug("Outbox: processed %d events", count)
        except Exception as e:
            logger.error("Outbox worker error: %s", e)
        await asyncio.sleep(POLL_INTERVAL_S)
