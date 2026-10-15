"""Pure scheduling policy helpers and durable job materialization."""

from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.db.models.scheduling import ScheduledJob


ACTIVATE_ORDER = "ACTIVATE_ORDER"
CONFIRMATION_REMINDER = "CONFIRMATION_REMINDER"
FULFILLMENT_REMINDER = "FULFILLMENT_REMINDER"
RESCHEDULE_REQUIRED = "RESCHEDULE_REQUIRED"

JOB_TYPES = {
    ACTIVATE_ORDER,
    CONFIRMATION_REMINDER,
    FULFILLMENT_REMINDER,
    RESCHEDULE_REQUIRED,
}
ACTIVE_JOB_STATES = {"PENDING", "RETRY_WAIT", "LEASED"}
TERMINAL_JOB_STATES = {"SUCCEEDED", "CANCELLED", "DEAD_LETTER"}


def retry_delay_seconds(
    attempt_count: int,
    *,
    base_seconds: int | None = None,
    maximum_seconds: int | None = None,
) -> int:
    """Return deterministic bounded exponential backoff for an attempted job."""
    base = base_seconds or settings.SCHEDULE_RETRY_BASE_SECONDS
    maximum = maximum_seconds or settings.SCHEDULE_RETRY_MAX_SECONDS
    exponent = max(0, int(attempt_count) - 1)
    return min(maximum, base * (2 ** exponent))


def durable_jobs_for_order(
    order,
    meal_schedule,
    *,
    now: datetime | None = None,
) -> list[ScheduledJob]:
    """Build the unique activation and reminder jobs for a scheduled order."""
    desired_for = order.desired_for or order.scheduled_for
    if desired_for is None:
        return []
    now = now or datetime.now(timezone.utc)
    activation_at = (
        order.submission_window_start
        or desired_for - timedelta(minutes=settings.SCHEDULE_ACTIVATION_LEAD_MINUTES)
    )
    activation_deadline = (
        order.submission_window_end or desired_for
    ) + timedelta(minutes=settings.SCHEDULE_LATE_GRACE_MINUTES)
    confirmation_at = desired_for - timedelta(
        minutes=settings.SCHEDULE_CONFIRMATION_REMINDER_MINUTES
    )
    fulfillment_at = desired_for - timedelta(
        minutes=settings.SCHEDULE_FULFILLMENT_REMINDER_MINUTES
    )
    base = {
        "restaurant_id": order.restaurant_id,
        "meal_schedule_id": meal_schedule.id if meal_schedule is not None else None,
        "order_id": order.id,
        "max_attempts": settings.SCHEDULE_JOB_MAX_ATTEMPTS,
        "payload_json": {
            "order_ref": order.order_ref,
            "desired_for": desired_for.isoformat(),
        },
    }
    return [
        ScheduledJob(
            **base,
            job_type=ACTIVATE_ORDER,
            dedupe_key=f"order:{order.id}:activate:v1",
            run_at=max(now, activation_at),
            deadline_at=activation_deadline,
        ),
        ScheduledJob(
            **base,
            job_type=CONFIRMATION_REMINDER,
            dedupe_key=f"order:{order.id}:confirmation-reminder:v1",
            run_at=max(now, confirmation_at),
            deadline_at=activation_at,
        ),
        ScheduledJob(
            **base,
            job_type=FULFILLMENT_REMINDER,
            dedupe_key=f"order:{order.id}:fulfillment-reminder:v1",
            run_at=max(now, fulfillment_at),
            deadline_at=desired_for,
        ),
    ]


def add_durable_jobs(session, order, meal_schedule, *, now: datetime | None = None) -> list[ScheduledJob]:
    """Attach newly constructed jobs to the caller's order transaction."""
    jobs = durable_jobs_for_order(order, meal_schedule, now=now)
    for job in jobs:
        session.add(job)
    return jobs
