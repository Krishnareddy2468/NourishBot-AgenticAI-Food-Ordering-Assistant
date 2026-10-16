"""Manager-only runtime readiness, alert, and public-release gate status."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text

from app.auth.deps import require_manager, tenant_id
from app.core.config import settings
from app.core.redis_client import get_redis
from app.db.models import OutboxEvent, ProviderAttempt, ScheduledJob, SchedulerHeartbeat
from app.db.session import AsyncSessionLocal
from app.operations.release_gate import evaluate_release_evidence, load_release_evidence


router = APIRouter(prefix="/api/v1/operations", tags=["operations"])


def _alert(code: str, severity: str, message: str, value=None, threshold=None) -> dict:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "value": value,
        "threshold": threshold,
    }


@router.get("/readiness")
async def operations_readiness(user=Depends(require_manager)):
    rid = tenant_id(user)
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        migration_head = (await db.execute(text("SELECT version_num FROM alembic_version"))).scalar_one_or_none()
        heartbeat = (await db.execute(
            select(SchedulerHeartbeat).order_by(SchedulerHeartbeat.last_seen_at.desc()).limit(1)
        )).scalar_one_or_none()
        oldest_due = (await db.execute(select(func.min(ScheduledJob.run_at)).where(
            ScheduledJob.restaurant_id == rid,
            ScheduledJob.state.in_(["PENDING", "RETRY_WAIT"]),
            ScheduledJob.run_at <= now,
        ))).scalar_one_or_none()
        dead_jobs = int((await db.execute(select(func.count(ScheduledJob.id)).where(
            ScheduledJob.restaurant_id == rid,
            ScheduledJob.state == "DEAD_LETTER",
        ))).scalar_one() or 0)
        oldest_outbox = (await db.execute(select(func.min(OutboxEvent.created_at)).where(
            OutboxEvent.restaurant_id == rid,
            OutboxEvent.processed_at.is_(None),
            OutboxEvent.dead_lettered_at.is_(None),
        ))).scalar_one_or_none()
        dead_outbox = int((await db.execute(select(func.count(OutboxEvent.id)).where(
            OutboxEvent.restaurant_id == rid,
            OutboxEvent.dead_lettered_at.is_not(None),
        ))).scalar_one() or 0)
        unknown_provider = int((await db.execute(select(func.count(ProviderAttempt.id)).where(
            ProviderAttempt.restaurant_id == rid,
            ProviderAttempt.status == "UNKNOWN",
        ))).scalar_one() or 0)

    redis_ready = False
    try:
        redis = await get_redis()
        redis_ready = bool(redis and await redis.ping())
    except Exception:
        redis_ready = False

    heartbeat_age = int((now - heartbeat.last_seen_at).total_seconds()) if heartbeat else None
    scheduler_lag = max(0, int((now - oldest_due).total_seconds())) if oldest_due else 0
    outbox_lag = max(0, int((now - oldest_outbox).total_seconds())) if oldest_outbox else 0
    alerts = []
    if heartbeat_age is None or heartbeat_age > settings.SCHEDULER_HEARTBEAT_STALE_SECONDS:
        alerts.append(_alert("SCHEDULER_HEARTBEAT_STALE", "critical", "Scheduler heartbeat is stale or missing", heartbeat_age, settings.SCHEDULER_HEARTBEAT_STALE_SECONDS))
    if scheduler_lag >= settings.ALERT_SCHEDULER_LAG_SECONDS:
        alerts.append(_alert("SCHEDULER_LAG", "critical", "Scheduled jobs are running late", scheduler_lag, settings.ALERT_SCHEDULER_LAG_SECONDS))
    if outbox_lag >= settings.ALERT_OUTBOX_LAG_SECONDS:
        alerts.append(_alert("OUTBOX_LAG", "warning", "Outbox delivery is delayed", outbox_lag, settings.ALERT_OUTBOX_LAG_SECONDS))
    if dead_jobs + dead_outbox >= settings.ALERT_DEAD_LETTER_COUNT:
        alerts.append(_alert("DEAD_LETTERS", "critical", "Manual-review dead letters require attention", dead_jobs + dead_outbox, settings.ALERT_DEAD_LETTER_COUNT))
    if unknown_provider:
        alerts.append(_alert("PROVIDER_UNKNOWN_OUTCOMES", "critical", "Provider outcomes require reconciliation", unknown_provider, 0))
    if migration_head != settings.EXPECTED_ALEMBIC_HEAD:
        alerts.append(_alert("MIGRATION_HEAD_MISMATCH", "critical", "Database migration head does not match the release", migration_head, settings.EXPECTED_ALEMBIC_HEAD))
    if not redis_ready:
        alerts.append(_alert("REDIS_UNAVAILABLE", "critical", "Redis-backed idempotency and rate limiting are unavailable"))

    evidence_gate = {
        "release_gate_passed": False,
        "decision": "KEEP_SAFE_MODE",
        "failed_checks": ["release_evidence_not_configured"],
    }
    if settings.RELEASE_EVIDENCE_FILE:
        try:
            evidence_gate = evaluate_release_evidence(load_release_evidence(settings.RELEASE_EVIDENCE_FILE), now=now)
        except ValueError as exc:
            evidence_gate = {
                "release_gate_passed": False,
                "decision": "KEEP_SAFE_MODE",
                "failed_checks": ["invalid_release_evidence"],
                "detail": str(exc),
            }

    runtime_ready = not alerts
    external_transactions_enabled = bool(
        runtime_ready
        and evidence_gate.get("release_gate_passed")
        and settings.external_transactions_enabled
    )
    return {
        "ok": runtime_ready,
        "runtime": {
            "migration_head": migration_head,
            "expected_migration_head": settings.EXPECTED_ALEMBIC_HEAD,
            "redis_ready": redis_ready,
            "scheduler_heartbeat_age_seconds": heartbeat_age,
            "scheduler_lag_seconds": scheduler_lag,
            "outbox_lag_seconds": outbox_lag,
            "dead_letter_count": dead_jobs + dead_outbox,
            "provider_unknown_count": unknown_provider,
        },
        "alerts": alerts,
        "monitoring_configured": settings.MONITORING_ALERTS_ENABLED,
        "release_evidence": evidence_gate,
        "external_transactions_enabled": external_transactions_enabled,
        "safe_mode": not external_transactions_enabled,
    }
