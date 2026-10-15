"""Durable, leased execution for meal activation and customer reminders."""

import asyncio
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, func, or_, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    CustomerConsent,
    MealSchedule,
    Order,
    OutboxEvent,
    ScheduledJob,
    ScheduledJobAttempt,
    SchedulerHeartbeat,
)
from app.domain.order_state import transition_order, transition_schedule
from app.domain.scheduling import (
    ACTIVATE_ORDER,
    CONFIRMATION_REMINDER,
    FULFILLMENT_REMINDER,
    RESCHEDULE_REQUIRED,
    retry_delay_seconds,
)
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    outcome: str = "SUCCEEDED"
    details: dict = field(default_factory=dict)


class SchedulerJobError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = True,
        next_action: str = "RETRY",
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.next_action = next_action


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _record_attempt(
    session: Session,
    job: ScheduledJob,
    *,
    worker_id: str,
    outcome: str,
    started_at: datetime,
    error_code: str | None = None,
    error_message: str | None = None,
    next_action: str | None = None,
    details: dict | None = None,
    completed_at: datetime | None = None,
) -> None:
    session.add(ScheduledJobAttempt(
        restaurant_id=job.restaurant_id,
        scheduled_job_id=job.id,
        attempt_number=job.attempt_count,
        worker_id=worker_id,
        outcome=outcome,
        error_code=error_code,
        error_message=(error_message or "")[:1000] or None,
        next_action=next_action,
        details_json=details or {},
        started_at=started_at,
        completed_at=completed_at or datetime.now(timezone.utc),
    ))


def _enqueue_manual_review(
    session: Session,
    job: ScheduledJob,
    *,
    now: datetime,
    reason_code: str,
) -> None:
    order = (
        session.execute(select(Order).where(Order.id == job.order_id).with_for_update()).scalar_one_or_none()
        if job.order_id
        else None
    )
    schedule = (
        session.execute(
            select(MealSchedule).where(MealSchedule.id == job.meal_schedule_id).with_for_update()
        ).scalar_one_or_none()
        if job.meal_schedule_id
        else None
    )

    if job.job_type == ACTIVATE_ORDER and order is not None:
        if order.schedule_state == "SCHEDULED":
            transition_schedule(
                session,
                order,
                "EXPIRED",
                actor_type="SYSTEM",
                source="SCHEDULER",
                reason_code=reason_code,
                meal_schedule=schedule,
                occurred_at=now,
            )
        if order.status in {"CREATED", "CONFIRMED"}:
            transition_order(
                session,
                order,
                "EXPIRED",
                actor_type="SYSTEM",
                source="SCHEDULER",
                reason_code=reason_code,
                occurred_at=now,
            )

        reminder_key = f"order:{order.id}:reschedule-required:v1"
        exists = session.execute(
            select(ScheduledJob.id).where(ScheduledJob.dedupe_key == reminder_key)
        ).scalar_one_or_none()
        if exists is None:
            session.add(ScheduledJob(
                restaurant_id=order.restaurant_id,
                meal_schedule_id=job.meal_schedule_id,
                order_id=order.id,
                job_type=RESCHEDULE_REQUIRED,
                state="PENDING",
                dedupe_key=reminder_key,
                run_at=now,
                deadline_at=now + timedelta(days=1),
                max_attempts=settings.SCHEDULE_JOB_MAX_ATTEMPTS,
                payload_json={"order_ref": order.order_ref, "reason_code": reason_code},
            ))

    session.add(OutboxEvent(
        restaurant_id=job.restaurant_id,
        event_type="schedule.manual_review_required",
        dedupe_key=f"scheduled-job:{job.id}:manual-review:{now.isoformat()}",
        payload={
            "job_id": job.id,
            "job_type": job.job_type,
            "order_id": job.order_id,
            "reason_code": reason_code,
            "next_action": "MANUAL_REVIEW",
        },
        max_attempts=settings.OUTBOX_MAX_ATTEMPTS,
    ))


def _recover_expired_leases(session: Session, now: datetime, worker_id: str) -> int:
    jobs = session.execute(
        select(ScheduledJob)
        .where(ScheduledJob.state == "LEASED", ScheduledJob.lease_expires_at <= now)
        .with_for_update(skip_locked=True)
    ).scalars().all()
    for job in jobs:
        expired_at = job.lease_expires_at or now
        exhausted = job.attempt_count >= job.max_attempts
        job.lease_owner = None
        job.lease_expires_at = None
        job.last_error_code = "LEASE_EXPIRED"
        job.last_error_message = "Worker lease expired before the attempt completed"
        if exhausted:
            job.state = "DEAD_LETTER"
            job.dead_lettered_at = now
            job.next_action = "MANUAL_REVIEW"
            _enqueue_manual_review(session, job, now=now, reason_code="LEASE_EXPIRED")
        else:
            job.state = "RETRY_WAIT"
            job.next_retry_at = now + timedelta(seconds=retry_delay_seconds(job.attempt_count))
            job.next_action = "RETRY_AFTER_LEASE_RECOVERY"
        _record_attempt(
            session,
            job,
            worker_id=worker_id,
            outcome="LEASE_EXPIRED",
            error_code="LEASE_EXPIRED",
            error_message=job.last_error_message,
            next_action=job.next_action,
            started_at=expired_at,
            completed_at=now,
        )
    return len(jobs)


def _claim_due_jobs(
    session: Session,
    *,
    now: datetime,
    lease_owner: str,
) -> list[int]:
    jobs = session.execute(
        select(ScheduledJob)
        .where(
            ScheduledJob.state.in_(["PENDING", "RETRY_WAIT"]),
            ScheduledJob.run_at <= now,
            or_(ScheduledJob.next_retry_at.is_(None), ScheduledJob.next_retry_at <= now),
        )
        .order_by(ScheduledJob.run_at, ScheduledJob.id)
        .limit(settings.SCHEDULE_JOB_BATCH_SIZE)
        .with_for_update(skip_locked=True)
    ).scalars().all()
    for job in jobs:
        job.state = "LEASED"
        job.lease_owner = lease_owner
        job.lease_expires_at = now + timedelta(seconds=settings.SCHEDULE_LEASE_SECONDS)
        job.attempt_count += 1
        job.next_retry_at = None
        job.next_action = "EXECUTE"
    return [job.id for job in jobs]


def _has_reminder_consent(session: Session, order: Order) -> bool:
    if not order.customer_id:
        return False
    decision = session.execute(
        select(CustomerConsent.granted)
        .where(
            CustomerConsent.restaurant_id == order.restaurant_id,
            CustomerConsent.customer_id == order.customer_id,
            CustomerConsent.purpose == "REMINDERS",
        )
        .order_by(CustomerConsent.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    return decision is True


def _send_job_notification(session: Session, job: ScheduledJob, order: Order) -> JobResult:
    if job.job_type != RESCHEDULE_REQUIRED and not _has_reminder_consent(session, order):
        return JobResult("SKIPPED_NO_CONSENT", {"consent": "REMINDERS_NOT_GRANTED"})
    if order.status in {"CANCELLED", "COMPLETED"}:
        return JobResult("SKIPPED_TERMINAL_ORDER", {"order_state": order.status})

    desired = order.desired_for or order.scheduled_for
    desired_text = desired.isoformat() if desired else "the selected time"
    if job.job_type == CONFIRMATION_REMINDER:
        message = (
            f"Reminder for {order.order_ref}: your meal is planned for {desired_text}. "
            "Please review the schedule now if anything needs to change."
        )
    elif job.job_type == FULFILLMENT_REMINDER:
        message = (
            f"Your scheduled meal {order.order_ref} is coming up at {desired_text}. "
            "We'll keep you updated as fulfillment begins."
        )
    else:
        message = (
            f"We couldn't activate scheduled meal {order.order_ref} safely. "
            "Please choose a new time; our team has also been asked to review it."
        )

    from app.realtime.notifications import notify_customer

    delivered = asyncio.run(notify_customer(
        restaurant_id=order.restaurant_id,
        order_ref=order.order_ref,
        event_type=f"schedule.{job.job_type.lower()}",
        message=message,
    ))
    if not delivered:
        raise SchedulerJobError(
            "NOTIFICATION_UNAVAILABLE",
            "Customer notification channel did not confirm delivery",
            next_action="RETRY_NOTIFICATION",
        )
    return JobResult("DELIVERED", {"channel_delivery": "CONFIRMED"})


def _execute_job(session: Session, job: ScheduledJob, now: datetime) -> JobResult:
    deadline_missed = bool(job.deadline_at and now > job.deadline_at)
    if deadline_missed and job.job_type != ACTIVATE_ORDER:
        return JobResult("SKIPPED_DEADLINE", {"deadline_at": job.deadline_at.isoformat()})

    order = (
        session.execute(select(Order).where(Order.id == job.order_id).with_for_update()).scalar_one_or_none()
        if job.order_id
        else None
    )
    if order is None:
        raise SchedulerJobError(
            "ORDER_NOT_FOUND",
            "The scheduled job no longer references an order",
            retryable=False,
            next_action="MANUAL_REVIEW",
        )

    if job.job_type == ACTIVATE_ORDER:
        if order.schedule_state != "SCHEDULED":
            return JobResult("SKIPPED_IDEMPOTENT", {"schedule_state": order.schedule_state})
        if deadline_missed:
            raise SchedulerJobError(
                "SUBMISSION_DEADLINE_MISSED",
                "The activation deadline passed before safe execution",
                retryable=False,
                next_action=settings.SCHEDULE_LATE_POLICY,
            )
        if order.provider.upper() != "NOURISHBOT":
            raise SchedulerJobError(
                "PROVIDER_EXECUTION_NOT_ENABLED",
                "External provider scheduling requires a verified provider adapter",
                retryable=False,
                next_action="MANUAL_REVIEW",
            )
        schedule = (
            session.execute(
                select(MealSchedule)
                .where(MealSchedule.id == job.meal_schedule_id)
                .with_for_update()
            ).scalar_one_or_none()
            if job.meal_schedule_id
            else None
        )
        transition_schedule(
            session,
            order,
            "ACTIVATED",
            actor_type="SYSTEM",
            source="SCHEDULER",
            reason_code="SUBMISSION_WINDOW_OPEN",
            meal_schedule=schedule,
            occurred_at=now,
        )
        if order.status in {"CREATED", "CONFIRMED"}:
            transition_order(
                session,
                order,
                "ACCEPTED",
                actor_type="SYSTEM",
                source="SCHEDULER",
                reason_code="SCHEDULE_ACTIVATED",
                occurred_at=now,
            )
        session.add(OutboxEvent(
            restaurant_id=order.restaurant_id,
            event_type="order.status_changed",
            dedupe_key=f"scheduled-job:{job.id}:order-activated",
            max_attempts=settings.OUTBOX_MAX_ATTEMPTS,
            payload={
                "order_id": order.id,
                "order_ref": order.order_ref,
                "new_status": order.status,
                "activated_from_schedule": True,
                "order_type": order.order_type,
            },
        ))
        return JobResult("ACTIVATED", {"order_ref": order.order_ref})

    if job.job_type in {CONFIRMATION_REMINDER, FULFILLMENT_REMINDER, RESCHEDULE_REQUIRED}:
        return _send_job_notification(session, job, order)

    raise SchedulerJobError(
        "UNSUPPORTED_JOB_TYPE",
        f"Unsupported scheduler job type: {job.job_type}",
        retryable=False,
        next_action="MANUAL_REVIEW",
    )


def _process_claimed_job(
    engine,
    *,
    job_id: int,
    lease_owner: str,
    worker_id: str,
) -> bool:
    started_at = datetime.now(timezone.utc)
    with Session(engine) as session:
        job = session.execute(
            select(ScheduledJob)
            .where(
                ScheduledJob.id == job_id,
                ScheduledJob.state == "LEASED",
                ScheduledJob.lease_owner == lease_owner,
            )
            .with_for_update()
        ).scalar_one_or_none()
        if job is None:
            return False
        now = datetime.now(timezone.utc)
        try:
            with session.begin_nested():
                result = _execute_job(session, job, now)
        except SchedulerJobError as exc:
            exhausted = job.attempt_count >= job.max_attempts
            deadline_exceeded = bool(job.deadline_at and now >= job.deadline_at)
            if exc.retryable and not exhausted and not deadline_exceeded:
                job.state = "RETRY_WAIT"
                job.next_retry_at = now + timedelta(seconds=retry_delay_seconds(job.attempt_count))
                job.next_action = exc.next_action
                outcome = "RETRY_SCHEDULED"
            else:
                job.state = "DEAD_LETTER"
                job.dead_lettered_at = now
                job.next_action = "MANUAL_REVIEW"
                outcome = "DEAD_LETTER"
                _enqueue_manual_review(session, job, now=now, reason_code=exc.code)
            job.last_error_code = exc.code
            job.last_error_message = str(exc)[:1000]
            job.lease_owner = None
            job.lease_expires_at = None
            _record_attempt(
                session,
                job,
                worker_id=worker_id,
                outcome=outcome,
                error_code=exc.code,
                error_message=str(exc),
                next_action=job.next_action,
                started_at=started_at,
                completed_at=now,
            )
            session.commit()
            return False
        except Exception as exc:
            logger.exception("Unexpected scheduler job failure for job %s", job.id)
            error_code = "UNEXPECTED_EXECUTION_ERROR"
            exhausted = job.attempt_count >= job.max_attempts
            if exhausted:
                job.state = "DEAD_LETTER"
                job.dead_lettered_at = now
                job.next_action = "MANUAL_REVIEW"
                _enqueue_manual_review(session, job, now=now, reason_code=error_code)
                outcome = "DEAD_LETTER"
            else:
                job.state = "RETRY_WAIT"
                job.next_retry_at = now + timedelta(seconds=retry_delay_seconds(job.attempt_count))
                job.next_action = "RETRY"
                outcome = "RETRY_SCHEDULED"
            job.last_error_code = error_code
            job.last_error_message = (str(exc) or exc.__class__.__name__)[:1000]
            job.lease_owner = None
            job.lease_expires_at = None
            _record_attempt(
                session,
                job,
                worker_id=worker_id,
                outcome=outcome,
                error_code=error_code,
                error_message=job.last_error_message,
                next_action=job.next_action,
                started_at=started_at,
                completed_at=now,
            )
            session.commit()
            return False

        job.state = "SUCCEEDED"
        job.completed_at = now
        job.lease_owner = None
        job.lease_expires_at = None
        job.last_error_code = None
        job.last_error_message = None
        job.next_action = None
        _record_attempt(
            session,
            job,
            worker_id=worker_id,
            outcome=result.outcome,
            details=result.details,
            started_at=started_at,
            completed_at=now,
        )
        session.commit()
        return True


def _write_heartbeat(
    session: Session,
    *,
    worker_id: str,
    started_at: datetime,
    completed_at: datetime | None,
    claimed: int,
    succeeded: int,
    failed: int,
) -> None:
    now = completed_at or started_at
    oldest_due = session.execute(
        select(func.min(ScheduledJob.run_at)).where(
            ScheduledJob.state.in_(["PENDING", "RETRY_WAIT"]),
            ScheduledJob.run_at <= now,
        )
    ).scalar_one_or_none()
    lag = max(0, int((now - oldest_due).total_seconds())) if oldest_due else 0
    heartbeat = session.get(SchedulerHeartbeat, worker_id)
    if heartbeat is None:
        heartbeat = SchedulerHeartbeat(
            worker_id=worker_id,
            last_seen_at=now,
            tick_started_at=started_at,
        )
        session.add(heartbeat)
    heartbeat.last_seen_at = now
    heartbeat.tick_started_at = started_at
    heartbeat.tick_completed_at = completed_at
    heartbeat.claimed_count = claimed
    heartbeat.succeeded_count = succeeded
    heartbeat.failed_count = failed
    heartbeat.oldest_lag_seconds = lag
    heartbeat.details_json = {"lease_seconds": settings.SCHEDULE_LEASE_SECONDS}


@celery_app.task(name="app.workers.scheduled_order_worker.run_scheduler_tick")
def run_scheduler_tick() -> dict:
    """Recover abandoned work, lease due jobs, and execute each logical job once."""
    if not settings.DATABASE_URL_SYNC:
        logger.error("Scheduler skipped: DATABASE_URL_SYNC is not configured")
        return {"claimed": 0, "error": "DATABASE_URL_SYNC is not configured"}

    started_at = datetime.now(timezone.utc)
    worker_id = _worker_id()
    lease_owner = f"{worker_id}:{uuid.uuid4()}"
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)
    claimed_ids: list[int] = []
    recovered = 0
    try:
        with Session(engine) as session:
            recovered = _recover_expired_leases(session, started_at, worker_id)
            claimed_ids = _claim_due_jobs(session, now=started_at, lease_owner=lease_owner)
            _write_heartbeat(
                session,
                worker_id=worker_id,
                started_at=started_at,
                completed_at=None,
                claimed=len(claimed_ids),
                succeeded=0,
                failed=0,
            )
            session.commit()

        succeeded = sum(
            1
            for job_id in claimed_ids
            if _process_claimed_job(
                engine,
                job_id=job_id,
                lease_owner=lease_owner,
                worker_id=worker_id,
            )
        )
        completed_at = datetime.now(timezone.utc)
        with Session(engine) as session:
            _write_heartbeat(
                session,
                worker_id=worker_id,
                started_at=started_at,
                completed_at=completed_at,
                claimed=len(claimed_ids),
                succeeded=succeeded,
                failed=len(claimed_ids) - succeeded,
            )
            session.commit()
        return {
            "recovered": recovered,
            "claimed": len(claimed_ids),
            "succeeded": succeeded,
            "failed": len(claimed_ids) - succeeded,
            "job_ids": claimed_ids,
        }
    finally:
        engine.dispose()


@celery_app.task(name="app.workers.scheduled_order_worker.activate_due_orders")
def activate_due_orders() -> dict:
    """Compatibility task name retained for older deployment schedules."""
    return run_scheduler_tick()
