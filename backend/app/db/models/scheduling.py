"""Durable scheduler jobs, append-only attempts, and worker heartbeats."""

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, Text, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text

from app.db.base import Base, RestaurantMixin


class ScheduledJob(Base, RestaurantMixin):
    """A durable, uniquely keyed unit of scheduling or reminder work."""

    __tablename__ = "scheduled_jobs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    meal_schedule_id = Column(
        BigInteger,
        ForeignKey("meal_schedules.id", ondelete="SET NULL"),
        index=True,
    )
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="SET NULL"), index=True)
    job_type = Column(Text, nullable=False, index=True)
    state = Column(Text, nullable=False, server_default=text("'PENDING'"), index=True)
    dedupe_key = Column(Text, nullable=False, unique=True)
    run_at = Column(DateTime(timezone=True), nullable=False, index=True)
    deadline_at = Column(DateTime(timezone=True), index=True)
    next_retry_at = Column(DateTime(timezone=True), index=True)
    attempt_count = Column(Integer, nullable=False, server_default="0")
    max_attempts = Column(Integer, nullable=False, server_default="5")
    lease_owner = Column(Text)
    lease_expires_at = Column(DateTime(timezone=True), index=True)
    payload_json = Column(JSONB, nullable=False, default=dict)
    last_error_code = Column(Text)
    last_error_message = Column(Text)
    next_action = Column(Text)
    completed_at = Column(DateTime(timezone=True))
    dead_lettered_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ScheduledJobAttempt(Base, RestaurantMixin):
    """Append-only outcome record for every scheduler execution attempt."""

    __tablename__ = "scheduled_job_attempts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    scheduled_job_id = Column(
        BigInteger,
        ForeignKey("scheduled_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number = Column(Integer, nullable=False)
    worker_id = Column(Text, nullable=False)
    outcome = Column(Text, nullable=False, index=True)
    error_code = Column(Text)
    error_message = Column(Text)
    next_action = Column(Text)
    details_json = Column(JSONB, nullable=False, default=dict)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class SchedulerHeartbeat(Base):
    """Latest liveness and lag sample for a scheduler worker process."""

    __tablename__ = "scheduler_heartbeats"

    worker_id = Column(Text, primary_key=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=False, index=True)
    tick_started_at = Column(DateTime(timezone=True), nullable=False)
    tick_completed_at = Column(DateTime(timezone=True))
    claimed_count = Column(Integer, nullable=False, server_default="0")
    succeeded_count = Column(Integer, nullable=False, server_default="0")
    failed_count = Column(Integer, nullable=False, server_default="0")
    oldest_lag_seconds = Column(Integer, nullable=False, server_default="0")
    details_json = Column(JSONB, nullable=False, default=dict)


def _reject_attempt_mutation(_mapper, _connection, target) -> None:
    raise ValueError(f"{target.__class__.__name__} records are append-only")


event.listen(ScheduledJobAttempt, "before_update", _reject_attempt_mutation)
event.listen(ScheduledJobAttempt, "before_delete", _reject_attempt_mutation)
