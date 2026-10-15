from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes.scheduler import router as scheduler_router
from app.db.models import OutboxEvent, ScheduledJob, ScheduledJobAttempt, SchedulerHeartbeat
from app.domain.scheduling import (
    ACTIVATE_ORDER,
    CONFIRMATION_REMINDER,
    FULFILLMENT_REMINDER,
    durable_jobs_for_order,
    retry_delay_seconds,
)
from app.workers.celery_app import celery_app
from app.workers.scheduled_order_worker import SchedulerJobError, _execute_job


def test_phase3_tables_and_outbox_lease_fields_are_registered():
    assert ScheduledJob.__tablename__ == "scheduled_jobs"
    assert ScheduledJobAttempt.__tablename__ == "scheduled_job_attempts"
    assert SchedulerHeartbeat.__tablename__ == "scheduler_heartbeats"
    assert {
        "dedupe_key",
        "attempt_count",
        "next_retry_at",
        "lease_owner",
        "lease_expires_at",
        "dead_lettered_at",
    }.issubset(OutboxEvent.__table__.columns.keys())


def test_scheduled_order_materializes_unique_activation_and_reminder_jobs():
    now = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)
    desired = now + timedelta(days=2)
    order = SimpleNamespace(
        id=42,
        restaurant_id=3,
        order_ref="NB-42",
        desired_for=desired,
        scheduled_for=desired,
        submission_window_start=desired - timedelta(minutes=20),
        submission_window_end=desired,
    )
    jobs = durable_jobs_for_order(order, SimpleNamespace(id=7), now=now)

    assert {job.job_type for job in jobs} == {
        ACTIVATE_ORDER,
        CONFIRMATION_REMINDER,
        FULFILLMENT_REMINDER,
    }
    assert len({job.dedupe_key for job in jobs}) == 3
    assert all(job.order_id == 42 and job.meal_schedule_id == 7 for job in jobs)
    activation = next(job for job in jobs if job.job_type == ACTIVATE_ORDER)
    assert activation.run_at == desired - timedelta(minutes=20)
    assert activation.deadline_at > desired


def test_retry_backoff_is_exponential_and_bounded():
    assert retry_delay_seconds(1, base_seconds=10, maximum_seconds=60) == 10
    assert retry_delay_seconds(2, base_seconds=10, maximum_seconds=60) == 20
    assert retry_delay_seconds(3, base_seconds=10, maximum_seconds=60) == 40
    assert retry_delay_seconds(10, base_seconds=10, maximum_seconds=60) == 60


def test_late_activation_requires_manual_policy_before_any_mutation():
    class FakeResult:
        def scalar_one_or_none(self):
            return SimpleNamespace(schedule_state="SCHEDULED")

    class FakeSession:
        def execute(self, _query):
            return FakeResult()

    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        job_type=ACTIVATE_ORDER,
        deadline_at=now - timedelta(seconds=1),
        order_id=42,
    )
    with pytest.raises(SchedulerJobError, match="deadline passed") as error:
        _execute_job(FakeSession(), job, now)
    assert error.value.retryable is False


def test_expired_reminder_is_audited_skip_not_a_retry_loop():
    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        job_type=FULFILLMENT_REMINDER,
        deadline_at=now - timedelta(seconds=1),
    )
    result = _execute_job(None, job, now)
    assert result.outcome == "SKIPPED_DEADLINE"


def test_cancelled_activation_is_idempotent_even_after_its_deadline():
    class FakeResult:
        def scalar_one_or_none(self):
            return SimpleNamespace(schedule_state="CANCELLED")

    class FakeSession:
        def execute(self, _query):
            return FakeResult()

    now = datetime.now(timezone.utc)
    job = SimpleNamespace(
        job_type=ACTIVATE_ORDER,
        deadline_at=now - timedelta(minutes=5),
        order_id=42,
    )
    result = _execute_job(FakeSession(), job, now)
    assert result.outcome == "SKIPPED_IDEMPOTENT"


def test_celery_beat_runs_the_durable_scheduler():
    schedule = celery_app.conf.beat_schedule["run-durable-scheduler-every-minute"]
    assert schedule["task"] == "app.workers.scheduled_order_worker.run_scheduler_tick"
    assert schedule["schedule"] == 60.0


def test_manager_scheduler_routes_cover_health_queue_and_retry():
    paths = {route.path for route in scheduler_router.routes}
    assert {
        "/api/v1/scheduler/health",
        "/api/v1/scheduler/jobs",
        "/api/v1/scheduler/manual-review",
        "/api/v1/scheduler/jobs/{job_id}/attempts",
        "/api/v1/scheduler/jobs/{job_id}/retry",
    }.issubset(paths)


def test_phase3_migration_backfills_jobs_and_protects_attempt_history():
    migration = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0012_phase3_reliable_scheduler.py"
    ).read_text()
    assert 'down_revision: Union[str, Sequence[str], None] = "0011"' in migration
    assert "ON CONFLICT (dedupe_key) DO NOTHING" in migration
    assert "chr(58)" in migration
    assert "trg_scheduled_job_attempts_append_only" in migration
    assert "ix_outbox_events_claimable" in migration


def test_workers_claim_with_skip_locked_and_structured_failures():
    backend_root = Path(__file__).resolve().parents[2]
    scheduler_source = (backend_root / "app/workers/scheduled_order_worker.py").read_text()
    outbox_source = (backend_root / "app/workers/outbox_worker.py").read_text()
    assert ".with_for_update(skip_locked=True)" in scheduler_source
    assert "last_error_code" in scheduler_source
    assert "next_action" in scheduler_source
    assert ".with_for_update(skip_locked=True)" in outbox_source
    assert "lease_owner" in outbox_source
