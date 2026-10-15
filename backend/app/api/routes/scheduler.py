"""Manager-only scheduler health, lag, dead-letter, and manual retry APIs."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from app.auth.deps import require_manager, tenant_id
from app.core.config import settings
from app.db.models import Order, OutboxEvent, ScheduledJob, ScheduledJobAttempt, SchedulerHeartbeat
from app.db.session import AsyncSessionLocal

router = APIRouter(prefix="/api/v1/scheduler", tags=["scheduler"])


def _job_payload(job: ScheduledJob) -> dict:
    return {
        "id": job.id,
        "job_type": job.job_type,
        "state": job.state,
        "order_id": job.order_id,
        "meal_schedule_id": job.meal_schedule_id,
        "run_at": job.run_at.isoformat(),
        "deadline_at": job.deadline_at.isoformat() if job.deadline_at else None,
        "next_retry_at": job.next_retry_at.isoformat() if job.next_retry_at else None,
        "attempt_count": job.attempt_count,
        "max_attempts": job.max_attempts,
        "last_error_code": job.last_error_code,
        "last_error_message": job.last_error_message,
        "next_action": job.next_action,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "dead_lettered_at": job.dead_lettered_at.isoformat() if job.dead_lettered_at else None,
    }


@router.get("/health")
async def scheduler_health(user=Depends(require_manager)):
    rid = tenant_id(user)
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as session:
        counts = dict((await session.execute(
            select(ScheduledJob.state, func.count(ScheduledJob.id))
            .where(ScheduledJob.restaurant_id == rid)
            .group_by(ScheduledJob.state)
        )).all())
        oldest_due = (await session.execute(
            select(func.min(ScheduledJob.run_at)).where(
                ScheduledJob.restaurant_id == rid,
                ScheduledJob.state.in_(["PENDING", "RETRY_WAIT"]),
                ScheduledJob.run_at <= now,
            )
        )).scalar_one_or_none()
        latest_heartbeat = (await session.execute(
            select(SchedulerHeartbeat).order_by(SchedulerHeartbeat.last_seen_at.desc()).limit(1)
        )).scalar_one_or_none()
        outbox_dead = (await session.execute(
            select(func.count(OutboxEvent.id)).where(
                OutboxEvent.restaurant_id == rid,
                OutboxEvent.dead_lettered_at.is_not(None),
            )
        )).scalar_one()
        oldest_outbox = (await session.execute(
            select(func.min(OutboxEvent.created_at)).where(
                OutboxEvent.restaurant_id == rid,
                OutboxEvent.processed_at.is_(None),
                OutboxEvent.dead_lettered_at.is_(None),
            )
        )).scalar_one_or_none()

    heartbeat_age = (
        max(0, int((now - latest_heartbeat.last_seen_at).total_seconds()))
        if latest_heartbeat
        else None
    )
    worker_healthy = heartbeat_age is not None and heartbeat_age <= settings.SCHEDULER_HEARTBEAT_STALE_SECONDS
    scheduler_lag = max(0, int((now - oldest_due).total_seconds())) if oldest_due else 0
    outbox_lag = max(0, int((now - oldest_outbox).total_seconds())) if oldest_outbox else 0
    return {
        "ok": worker_healthy,
        "status": "healthy" if worker_healthy else "stale_or_not_started",
        "worker": {
            "id": latest_heartbeat.worker_id if latest_heartbeat else None,
            "last_seen_at": latest_heartbeat.last_seen_at.isoformat() if latest_heartbeat else None,
            "age_seconds": heartbeat_age,
            "stale_after_seconds": settings.SCHEDULER_HEARTBEAT_STALE_SECONDS,
        },
        "jobs_by_state": counts,
        "scheduler_lag_seconds": scheduler_lag,
        "manual_review_count": counts.get("DEAD_LETTER", 0),
        "outbox": {
            "lag_seconds": outbox_lag,
            "dead_letter_count": outbox_dead,
        },
    }


@router.get("/jobs")
async def list_scheduler_jobs(
    state: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    user=Depends(require_manager),
):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        query = select(ScheduledJob).where(ScheduledJob.restaurant_id == rid)
        if state:
            query = query.where(ScheduledJob.state == state.strip().upper())
        jobs = (await session.execute(
            query.order_by(ScheduledJob.run_at.desc()).limit(limit)
        )).scalars().all()
        return [_job_payload(job) for job in jobs]


@router.get("/manual-review")
async def manual_review_queue(
    limit: int = Query(default=100, ge=1, le=500),
    user=Depends(require_manager),
):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        jobs = (await session.execute(
            select(ScheduledJob)
            .where(ScheduledJob.restaurant_id == rid, ScheduledJob.state == "DEAD_LETTER")
            .order_by(ScheduledJob.dead_lettered_at, ScheduledJob.id)
            .limit(limit)
        )).scalars().all()
        return [_job_payload(job) for job in jobs]


@router.get("/jobs/{job_id}/attempts")
async def scheduler_job_attempts(job_id: int, user=Depends(require_manager)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as session:
        job_exists = (await session.execute(
            select(ScheduledJob.id).where(
                ScheduledJob.id == job_id,
                ScheduledJob.restaurant_id == rid,
            )
        )).scalar_one_or_none()
        if job_exists is None:
            raise HTTPException(404, "Scheduled job not found")
        attempts = (await session.execute(
            select(ScheduledJobAttempt)
            .where(
                ScheduledJobAttempt.scheduled_job_id == job_id,
                ScheduledJobAttempt.restaurant_id == rid,
            )
            .order_by(ScheduledJobAttempt.id)
        )).scalars().all()
        return [
            {
                "id": attempt.id,
                "attempt_number": attempt.attempt_number,
                "worker_id": attempt.worker_id,
                "outcome": attempt.outcome,
                "error_code": attempt.error_code,
                "error_message": attempt.error_message,
                "next_action": attempt.next_action,
                "details": attempt.details_json,
                "started_at": attempt.started_at.isoformat(),
                "completed_at": attempt.completed_at.isoformat(),
            }
            for attempt in attempts
        ]


@router.post("/jobs/{job_id}/retry")
async def retry_scheduler_job(job_id: int, user=Depends(require_manager)):
    rid = tenant_id(user)
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as session:
        job = (await session.execute(
            select(ScheduledJob)
            .where(ScheduledJob.id == job_id, ScheduledJob.restaurant_id == rid)
            .with_for_update()
        )).scalar_one_or_none()
        if not job:
            raise HTTPException(404, "Scheduled job not found")
        if job.state != "DEAD_LETTER":
            raise HTTPException(409, "Only dead-letter jobs can be manually retried")
        if job.job_type == "ACTIVATE_ORDER" and job.order_id:
            order = (await session.execute(
                select(Order).where(Order.id == job.order_id, Order.restaurant_id == rid)
            )).scalar_one_or_none()
            if not order or order.schedule_state != "SCHEDULED":
                raise HTTPException(409, "This meal must be rescheduled before activation can retry")

        prior_attempt_count = job.attempt_count
        job.state = "PENDING"
        job.run_at = now
        job.deadline_at = now + timedelta(minutes=settings.SCHEDULE_LATE_GRACE_MINUTES)
        job.next_retry_at = None
        job.attempt_count = 0
        job.lease_owner = None
        job.lease_expires_at = None
        job.last_error_code = None
        job.last_error_message = None
        job.next_action = "MANUAL_RETRY"
        job.dead_lettered_at = None
        job.completed_at = None
        session.add(ScheduledJobAttempt(
            restaurant_id=rid,
            scheduled_job_id=job.id,
            attempt_number=prior_attempt_count,
            worker_id=f"STAFF:{user.get('user_id', 'unknown')}",
            outcome="MANUALLY_REQUEUED",
            next_action="EXECUTE",
            details_json={},
            started_at=now,
            completed_at=now,
        ))
        await session.commit()
        await session.refresh(job)
        return _job_payload(job)
