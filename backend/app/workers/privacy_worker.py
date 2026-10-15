"""Scheduled retention enforcement for message, location, and provider data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.agent.privacy_service import prune_session_history, prune_session_location_data
from app.db.models import (
    ConfirmationChallenge,
    DeliveryLocationEvent,
    IdentityLinkChallenge,
    ProviderAttempt,
    Session as SessionModel,
)
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.privacy_worker.enforce_privacy_retention")
def enforce_privacy_retention() -> dict:
    if not settings.DATABASE_URL_SYNC:
        logger.error("Privacy retention skipped: DATABASE_URL_SYNC is not configured")
        return {"error": "DATABASE_URL_SYNC is not configured"}

    now = datetime.now(timezone.utc)
    message_cutoff = now - timedelta(days=settings.MESSAGE_RETENTION_DAYS)
    location_cutoff = now - timedelta(days=settings.LOCATION_RETENTION_DAYS)
    provider_cutoff = now - timedelta(days=settings.PROVIDER_RESPONSE_RETENTION_DAYS)
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)
    pruned_sessions = 0
    try:
        with Session(engine) as db:
            sessions = db.execute(select(SessionModel)).scalars().all()
            for session in sessions:
                updated_at = session.updated_at
                if updated_at and updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                pruned = prune_session_history(session.history_json, message_cutoff, updated_at)
                pruned = prune_session_location_data(pruned, location_cutoff, updated_at)
                if pruned != session.history_json:
                    session.history_json = pruned
                    pruned_sessions += 1

            location_result = db.execute(
                delete(DeliveryLocationEvent).where(DeliveryLocationEvent.recorded_at < location_cutoff)
            )
            provider_result = db.execute(
                update(ProviderAttempt)
                .where(ProviderAttempt.updated_at < provider_cutoff)
                .values(response_excerpt=None, last_error=None)
            )
            db.execute(
                update(ConfirmationChallenge)
                .where(
                    ConfirmationChallenge.expires_at < now,
                    ConfirmationChallenge.used_at.is_(None),
                    ConfirmationChallenge.invalidated_at.is_(None),
                )
                .values(invalidated_at=now, invalidation_reason="EXPIRED")
            )
            db.execute(
                delete(IdentityLinkChallenge).where(
                    IdentityLinkChallenge.expires_at < now - timedelta(days=1)
                )
            )
            db.commit()
            return {
                "sessions_pruned": pruned_sessions,
                "location_events_deleted": max(location_result.rowcount or 0, 0),
                "provider_records_redacted": max(provider_result.rowcount or 0, 0),
            }
    finally:
        engine.dispose()
