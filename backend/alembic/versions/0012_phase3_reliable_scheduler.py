"""phase 3 durable scheduling, retries, reminders, and leased outbox

Revision ID: 0012
Revises: 0011
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scheduled_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column(
            "meal_schedule_id",
            sa.BigInteger(),
            sa.ForeignKey("meal_schedules.id", ondelete="SET NULL"),
        ),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.id", ondelete="SET NULL")),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), server_default="PENDING", nullable=False),
        sa.Column("dedupe_key", sa.Text(), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True)),
        sa.Column("next_retry_at", sa.DateTime(timezone=True)),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="5", nullable=False),
        sa.Column("lease_owner", sa.Text()),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("last_error_code", sa.Text()),
        sa.Column("last_error_message", sa.Text()),
        sa.Column("next_action", sa.Text()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("dead_lettered_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "state IN ('PENDING','LEASED','RETRY_WAIT','SUCCEEDED','CANCELLED','DEAD_LETTER')",
            name="ck_scheduled_jobs_state",
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_scheduled_jobs_attempt_count"),
        sa.CheckConstraint("max_attempts > 0", name="ck_scheduled_jobs_max_attempts"),
        sa.UniqueConstraint("dedupe_key", name="uq_scheduled_jobs_dedupe_key"),
    )
    op.create_index("ix_scheduled_jobs_restaurant_id", "scheduled_jobs", ["restaurant_id"])
    op.create_index("ix_scheduled_jobs_meal_schedule_id", "scheduled_jobs", ["meal_schedule_id"])
    op.create_index("ix_scheduled_jobs_order_id", "scheduled_jobs", ["order_id"])
    op.create_index("ix_scheduled_jobs_job_type", "scheduled_jobs", ["job_type"])
    op.create_index("ix_scheduled_jobs_state", "scheduled_jobs", ["state"])
    op.create_index("ix_scheduled_jobs_run_at", "scheduled_jobs", ["run_at"])
    op.create_index("ix_scheduled_jobs_deadline_at", "scheduled_jobs", ["deadline_at"])
    op.create_index("ix_scheduled_jobs_next_retry_at", "scheduled_jobs", ["next_retry_at"])
    op.create_index("ix_scheduled_jobs_lease_expires_at", "scheduled_jobs", ["lease_expires_at"])
    op.create_index(
        "ix_scheduled_jobs_due",
        "scheduled_jobs",
        ["state", "run_at", "next_retry_at"],
    )

    op.create_table(
        "scheduled_job_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column(
            "scheduled_job_id",
            sa.BigInteger(),
            sa.ForeignKey("scheduled_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("error_code", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("next_action", sa.Text()),
        sa.Column("details_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_scheduled_job_attempts_restaurant_id", "scheduled_job_attempts", ["restaurant_id"])
    op.create_index("ix_scheduled_job_attempts_scheduled_job_id", "scheduled_job_attempts", ["scheduled_job_id"])
    op.create_index("ix_scheduled_job_attempts_outcome", "scheduled_job_attempts", ["outcome"])

    op.create_table(
        "scheduler_heartbeats",
        sa.Column("worker_id", sa.Text(), primary_key=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tick_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tick_completed_at", sa.DateTime(timezone=True)),
        sa.Column("claimed_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("succeeded_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("oldest_lag_seconds", sa.Integer(), server_default="0", nullable=False),
        sa.Column("details_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    )
    op.create_index("ix_scheduler_heartbeats_last_seen_at", "scheduler_heartbeats", ["last_seen_at"])

    for column in (
        sa.Column("dedupe_key", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="8", nullable=False),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("dead_lettered_at", sa.DateTime(timezone=True), nullable=True),
    ):
        op.add_column("outbox_events", column)
    op.create_index("uq_outbox_events_dedupe_key", "outbox_events", ["dedupe_key"], unique=True)
    op.create_index("ix_outbox_events_next_retry_at", "outbox_events", ["next_retry_at"])
    op.create_index("ix_outbox_events_lease_expires_at", "outbox_events", ["lease_expires_at"])
    op.create_index("ix_outbox_events_dead_lettered_at", "outbox_events", ["dead_lettered_at"])
    op.create_index(
        "ix_outbox_events_claimable",
        "outbox_events",
        ["processed_at", "dead_lettered_at", "next_retry_at", "created_at"],
    )

    op.execute(
        """
        INSERT INTO scheduled_jobs
            (restaurant_id, meal_schedule_id, order_id, job_type, state, dedupe_key,
             run_at, deadline_at, attempt_count, max_attempts, payload_json, created_at, updated_at)
        SELECT o.restaurant_id, ms.id, o.id, 'ACTIVATE_ORDER', 'PENDING',
               concat('order', chr(58), o.id::text, chr(58), 'activate', chr(58), 'v1'),
               GREATEST(now(), COALESCE(o.submission_window_start, o.desired_for - interval '20 minutes')),
               COALESCE(o.submission_window_end, o.desired_for) + interval '10 minutes',
               0, 5,
               jsonb_build_object('order_ref', o.order_ref, 'desired_for', o.desired_for),
               now(), now()
        FROM orders o
        LEFT JOIN meal_schedules ms ON ms.order_id = o.id
        WHERE o.schedule_state = 'SCHEDULED' AND o.desired_for IS NOT NULL
        ON CONFLICT (dedupe_key) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO scheduled_jobs
            (restaurant_id, meal_schedule_id, order_id, job_type, state, dedupe_key,
             run_at, deadline_at, attempt_count, max_attempts, payload_json, created_at, updated_at)
        SELECT o.restaurant_id, ms.id, o.id, 'CONFIRMATION_REMINDER', 'PENDING',
               concat('order', chr(58), o.id::text, chr(58), 'confirmation-reminder', chr(58), 'v1'),
               GREATEST(now(), o.desired_for - interval '24 hours'),
               COALESCE(o.submission_window_start, o.desired_for),
               0, 5,
               jsonb_build_object('order_ref', o.order_ref, 'desired_for', o.desired_for),
               now(), now()
        FROM orders o
        LEFT JOIN meal_schedules ms ON ms.order_id = o.id
        WHERE o.schedule_state = 'SCHEDULED' AND o.desired_for IS NOT NULL
        ON CONFLICT (dedupe_key) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO scheduled_jobs
            (restaurant_id, meal_schedule_id, order_id, job_type, state, dedupe_key,
             run_at, deadline_at, attempt_count, max_attempts, payload_json, created_at, updated_at)
        SELECT o.restaurant_id, ms.id, o.id, 'FULFILLMENT_REMINDER', 'PENDING',
               concat('order', chr(58), o.id::text, chr(58), 'fulfillment-reminder', chr(58), 'v1'),
               GREATEST(now(), o.desired_for - interval '60 minutes'),
               o.desired_for,
               0, 5,
               jsonb_build_object('order_ref', o.order_ref, 'desired_for', o.desired_for),
               now(), now()
        FROM orders o
        LEFT JOIN meal_schedules ms ON ms.order_id = o.id
        WHERE o.schedule_state = 'SCHEDULED' AND o.desired_for IS NOT NULL
        ON CONFLICT (dedupe_key) DO NOTHING
        """
    )

    op.execute(
        "CREATE TRIGGER trg_scheduled_job_attempts_append_only "
        "BEFORE UPDATE OR DELETE ON scheduled_job_attempts "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_scheduled_job_attempts_append_only ON scheduled_job_attempts"
    )

    op.drop_index("ix_outbox_events_claimable", table_name="outbox_events")
    op.drop_index("ix_outbox_events_dead_lettered_at", table_name="outbox_events")
    op.drop_index("ix_outbox_events_lease_expires_at", table_name="outbox_events")
    op.drop_index("ix_outbox_events_next_retry_at", table_name="outbox_events")
    op.drop_index("uq_outbox_events_dedupe_key", table_name="outbox_events")
    for column_name in (
        "dead_lettered_at",
        "last_error",
        "lease_expires_at",
        "lease_owner",
        "next_retry_at",
        "max_attempts",
        "attempt_count",
        "dedupe_key",
    ):
        op.drop_column("outbox_events", column_name)

    op.drop_index("ix_scheduler_heartbeats_last_seen_at", table_name="scheduler_heartbeats")
    op.drop_table("scheduler_heartbeats")
    op.drop_index("ix_scheduled_job_attempts_outcome", table_name="scheduled_job_attempts")
    op.drop_index("ix_scheduled_job_attempts_scheduled_job_id", table_name="scheduled_job_attempts")
    op.drop_index("ix_scheduled_job_attempts_restaurant_id", table_name="scheduled_job_attempts")
    op.drop_table("scheduled_job_attempts")
    op.drop_index("ix_scheduled_jobs_due", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_lease_expires_at", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_next_retry_at", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_deadline_at", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_run_at", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_state", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_job_type", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_order_id", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_meal_schedule_id", table_name="scheduled_jobs")
    op.drop_index("ix_scheduled_jobs_restaurant_id", table_name="scheduled_jobs")
    op.drop_table("scheduled_jobs")
