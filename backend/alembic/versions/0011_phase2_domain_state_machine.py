"""phase 2 planning domain and audited order state machine

Revision ID: 0011
Revises: 0010
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "meal_plans",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="DRAFT", nullable=False),
        sa.Column("starts_on", sa.Date(), nullable=False),
        sa.Column("ends_on", sa.Date(), nullable=False),
        sa.Column("timezone", sa.Text(), server_default="Asia/Kolkata", nullable=False),
        sa.Column("created_by_channel_id", sa.BigInteger(), sa.ForeignKey("channels.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("ends_on >= starts_on", name="ck_meal_plans_date_range"),
    )
    op.create_index("ix_meal_plans_restaurant_id", "meal_plans", ["restaurant_id"])
    op.create_index("ix_meal_plans_customer_id", "meal_plans", ["customer_id"])

    op.create_table(
        "meal_templates",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), server_default="MEAL", nullable=False),
        sa.Column("definition_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("times_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_meal_templates_restaurant_id", "meal_templates", ["restaurant_id"])
    op.create_index("ix_meal_templates_customer_id", "meal_templates", ["customer_id"])

    op.create_table(
        "meal_routines",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("template_id", sa.BigInteger(), sa.ForeignKey("meal_templates.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), server_default="ACTIVE", nullable=False),
        sa.Column("meal_period", sa.Text(), nullable=False),
        sa.Column("timezone", sa.Text(), server_default="Asia/Kolkata", nullable=False),
        sa.Column("recurrence_rule", sa.Text(), nullable=False),
        sa.Column("next_materialization_at", sa.DateTime(timezone=True)),
        sa.Column("paused_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_meal_routines_restaurant_id", "meal_routines", ["restaurant_id"])
    op.create_index("ix_meal_routines_customer_id", "meal_routines", ["customer_id"])
    op.create_index("ix_meal_routines_next_materialization_at", "meal_routines", ["next_materialization_at"])

    op.create_table(
        "meal_budgets",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("period", sa.Text(), nullable=False),
        sa.Column("limit_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.Text(), server_default="INR", nullable=False),
        sa.Column("starts_on", sa.Date(), nullable=False),
        sa.Column("ends_on", sa.Date(), nullable=False),
        sa.Column("alert_thresholds_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("limit_cents >= 0", name="ck_meal_budgets_nonnegative_limit"),
        sa.CheckConstraint("ends_on >= starts_on", name="ck_meal_budgets_date_range"),
        sa.UniqueConstraint(
            "restaurant_id",
            "customer_id",
            "period",
            "starts_on",
            name="uq_meal_budget_customer_period_start",
        ),
    )
    op.create_index("ix_meal_budgets_restaurant_id", "meal_budgets", ["restaurant_id"])
    op.create_index("ix_meal_budgets_customer_id", "meal_budgets", ["customer_id"])

    for column in (
        sa.Column("schedule_state", sa.Text(), server_default="NOT_SCHEDULED", nullable=True),
        sa.Column("provider_state", sa.Text(), server_default="NOT_REQUIRED", nullable=True),
        sa.Column("fulfillment_state", sa.Text(), server_default="NOT_STARTED", nullable=True),
        sa.Column("desired_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ready_by", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_slot_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_slot_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submission_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submission_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_reason", sa.Text(), nullable=True),
    ):
        op.add_column("orders", column)

    op.execute("UPDATE orders SET desired_for = scheduled_for WHERE desired_for IS NULL")
    op.execute(
        "UPDATE orders SET schedule_state = CASE "
        "WHEN scheduled_for IS NULL THEN 'NOT_SCHEDULED' "
        "WHEN status = 'SCHEDULED' THEN 'SCHEDULED' "
        "WHEN status IN ('DELIVERED', 'COMPLETED') THEN 'FULFILLED' "
        "WHEN status = 'CANCELLED' THEN 'CANCELLED' "
        "ELSE 'ACTIVATED' END"
    )
    op.execute(
        "UPDATE orders SET provider_state = CASE "
        "WHEN upper(provider) = 'NOURISHBOT' THEN 'NOT_REQUIRED' "
        "WHEN status IN ('SCHEDULED', 'CREATED') THEN 'PENDING' "
        "WHEN status = 'CANCELLED' THEN 'CANCELLED' "
        "ELSE 'CONFIRMED' END"
    )
    op.execute(
        "UPDATE orders SET fulfillment_state = CASE status "
        "WHEN 'ACCEPTED' THEN 'ACCEPTED' "
        "WHEN 'CONFIRMED_BY_RESTAURANT' THEN 'ACCEPTED' "
        "WHEN 'PREPARING' THEN 'PREPARING' "
        "WHEN 'READY' THEN 'READY' "
        "WHEN 'OUT_FOR_DELIVERY' THEN 'OUT_FOR_DELIVERY' "
        "WHEN 'DELIVERED' THEN 'DELIVERED' "
        "WHEN 'COMPLETED' THEN 'COMPLETED' "
        "WHEN 'CANCELLED' THEN 'CANCELLED' "
        "ELSE 'NOT_STARTED' END"
    )
    op.execute(
        "UPDATE orders SET status = CASE status "
        "WHEN 'SCHEDULED' THEN 'CONFIRMED' "
        "WHEN 'PAID' THEN 'CONFIRMED' "
        "WHEN 'CONFIRMED_BY_RESTAURANT' THEN 'ACCEPTED' "
        "ELSE status END"
    )
    op.alter_column("orders", "schedule_state", nullable=False)
    op.alter_column("orders", "provider_state", nullable=False)
    op.alter_column("orders", "fulfillment_state", nullable=False)
    op.create_index("ix_orders_schedule_state", "orders", ["schedule_state"])
    op.create_index("ix_orders_provider_state", "orders", ["provider_state"])
    op.create_index("ix_orders_fulfillment_state", "orders", ["fulfillment_state"])
    op.create_index("ix_orders_desired_for", "orders", ["desired_for"])

    op.create_table(
        "meal_schedules",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("meal_plan_id", sa.BigInteger(), sa.ForeignKey("meal_plans.id", ondelete="SET NULL")),
        sa.Column("template_id", sa.BigInteger(), sa.ForeignKey("meal_templates.id", ondelete="SET NULL")),
        sa.Column("routine_id", sa.BigInteger(), sa.ForeignKey("meal_routines.id", ondelete="SET NULL")),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.id", ondelete="SET NULL")),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("meal_period", sa.Text(), server_default="UNSPECIFIED", nullable=False),
        sa.Column("state", sa.Text(), server_default="DRAFT", nullable=False),
        sa.Column("desired_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ready_by", sa.DateTime(timezone=True)),
        sa.Column("submission_window_start", sa.DateTime(timezone=True)),
        sa.Column("submission_window_end", sa.DateTime(timezone=True)),
        sa.Column("timezone", sa.Text(), server_default="Asia/Kolkata", nullable=False),
        sa.Column("source", sa.Text(), server_default="WEB", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "submission_window_start IS NULL OR submission_window_end IS NULL OR "
            "submission_window_end >= submission_window_start",
            name="ck_meal_schedules_submission_window",
        ),
    )
    op.create_index("ix_meal_schedules_restaurant_id", "meal_schedules", ["restaurant_id"])
    op.create_index("ix_meal_schedules_customer_id", "meal_schedules", ["customer_id"])
    op.create_index("ix_meal_schedules_meal_plan_id", "meal_schedules", ["meal_plan_id"])
    op.create_index("ix_meal_schedules_routine_id", "meal_schedules", ["routine_id"])
    op.create_index("ix_meal_schedules_order_id", "meal_schedules", ["order_id"], unique=True)
    op.create_index("ix_meal_schedules_state", "meal_schedules", ["state"])
    op.create_index("ix_meal_schedules_desired_for", "meal_schedules", ["desired_for"])

    op.execute(
        "INSERT INTO meal_schedules "
        "(restaurant_id, customer_id, order_id, title, meal_period, state, desired_for, ready_by, timezone, source) "
        "SELECT restaurant_id, customer_id, id, 'Legacy order ' || order_ref, 'UNSPECIFIED', "
        "schedule_state, scheduled_for, COALESCE(ready_by, scheduled_for), 'Asia/Kolkata', source_channel "
        "FROM orders WHERE scheduled_for IS NOT NULL AND customer_id IS NOT NULL"
    )

    op.create_table(
        "order_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("from_state", sa.Text()),
        sa.Column("to_state", sa.Text(), nullable=False),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text()),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("reason_code", sa.Text()),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_order_events_restaurant_id", "order_events", ["restaurant_id"])
    op.create_index("ix_order_events_order_id", "order_events", ["order_id"])
    op.create_index("ix_order_events_event_type", "order_events", ["event_type"])
    op.create_index("ix_order_events_occurred_at", "order_events", ["occurred_at"])

    op.create_table(
        "schedule_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("meal_schedule_id", sa.BigInteger(), sa.ForeignKey("meal_schedules.id", ondelete="CASCADE"), nullable=False),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.id", ondelete="SET NULL")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("from_state", sa.Text()),
        sa.Column("to_state", sa.Text(), nullable=False),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text()),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("reason_code", sa.Text()),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_schedule_events_restaurant_id", "schedule_events", ["restaurant_id"])
    op.create_index("ix_schedule_events_meal_schedule_id", "schedule_events", ["meal_schedule_id"])
    op.create_index("ix_schedule_events_order_id", "schedule_events", ["order_id"])
    op.create_index("ix_schedule_events_event_type", "schedule_events", ["event_type"])
    op.create_index("ix_schedule_events_occurred_at", "schedule_events", ["occurred_at"])

    op.execute(
        "INSERT INTO order_events "
        "(restaurant_id, order_id, event_type, from_state, to_state, actor_type, source, reason_code, metadata_json, occurred_at) "
        "SELECT restaurant_id, id, 'ORDER_MIGRATED', NULL, status, 'SYSTEM', 'MIGRATION', "
        "'LEGACY_STATUS_MIGRATION', '{}'::jsonb, COALESCE(created_at, now()) FROM orders"
    )
    op.execute(
        "INSERT INTO schedule_events "
        "(restaurant_id, meal_schedule_id, order_id, event_type, from_state, to_state, actor_type, source, reason_code, metadata_json, occurred_at) "
        "SELECT restaurant_id, id, order_id, 'SCHEDULE_MIGRATED', NULL, state, 'SYSTEM', 'MIGRATION', "
        "'LEGACY_SCHEDULE_MIGRATION', '{}'::jsonb, created_at FROM meal_schedules"
    )

    op.add_column("provider_attempts", sa.Column("order_id", sa.BigInteger(), nullable=True))
    op.add_column("provider_attempts", sa.Column("meal_schedule_id", sa.BigInteger(), nullable=True))
    op.add_column("provider_attempts", sa.Column("receipt_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("provider_attempts", sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("provider_attempts", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key(
        "fk_provider_attempts_order_id",
        "provider_attempts",
        "orders",
        ["order_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_provider_attempts_meal_schedule_id",
        "provider_attempts",
        "meal_schedules",
        ["meal_schedule_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_provider_attempts_order_id", "provider_attempts", ["order_id"])
    op.create_index("ix_provider_attempts_meal_schedule_id", "provider_attempts", ["meal_schedule_id"])

    op.execute(
        """
        CREATE OR REPLACE FUNCTION nourishbot_reject_event_mutation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_order_events_append_only BEFORE UPDATE OR DELETE ON order_events "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )
    op.execute(
        "CREATE TRIGGER trg_schedule_events_append_only BEFORE UPDATE OR DELETE ON schedule_events "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_schedule_events_append_only ON schedule_events")
    op.execute("DROP TRIGGER IF EXISTS trg_order_events_append_only ON order_events")
    op.execute("DROP FUNCTION IF EXISTS nourishbot_reject_event_mutation()")

    op.drop_index("ix_provider_attempts_meal_schedule_id", table_name="provider_attempts")
    op.drop_index("ix_provider_attempts_order_id", table_name="provider_attempts")
    op.drop_constraint("fk_provider_attempts_meal_schedule_id", "provider_attempts", type_="foreignkey")
    op.drop_constraint("fk_provider_attempts_order_id", "provider_attempts", type_="foreignkey")
    op.drop_column("provider_attempts", "completed_at")
    op.drop_column("provider_attempts", "submitted_at")
    op.drop_column("provider_attempts", "receipt_json")
    op.drop_column("provider_attempts", "meal_schedule_id")
    op.drop_column("provider_attempts", "order_id")

    op.drop_index("ix_schedule_events_occurred_at", table_name="schedule_events")
    op.drop_index("ix_schedule_events_event_type", table_name="schedule_events")
    op.drop_index("ix_schedule_events_order_id", table_name="schedule_events")
    op.drop_index("ix_schedule_events_meal_schedule_id", table_name="schedule_events")
    op.drop_index("ix_schedule_events_restaurant_id", table_name="schedule_events")
    op.drop_table("schedule_events")
    op.drop_index("ix_order_events_occurred_at", table_name="order_events")
    op.drop_index("ix_order_events_event_type", table_name="order_events")
    op.drop_index("ix_order_events_order_id", table_name="order_events")
    op.drop_index("ix_order_events_restaurant_id", table_name="order_events")
    op.drop_table("order_events")

    op.drop_index("ix_meal_schedules_desired_for", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_state", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_order_id", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_routine_id", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_meal_plan_id", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_customer_id", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_restaurant_id", table_name="meal_schedules")
    op.drop_table("meal_schedules")

    op.execute(
        "UPDATE orders SET status = 'SCHEDULED' "
        "WHERE schedule_state = 'SCHEDULED' AND status = 'CONFIRMED'"
    )
    op.drop_index("ix_orders_desired_for", table_name="orders")
    op.drop_index("ix_orders_fulfillment_state", table_name="orders")
    op.drop_index("ix_orders_provider_state", table_name="orders")
    op.drop_index("ix_orders_schedule_state", table_name="orders")
    for column_name in (
        "terminal_reason",
        "expired_at",
        "failed_at",
        "cancelled_at",
        "submission_window_end",
        "submission_window_start",
        "provider_slot_end",
        "provider_slot_start",
        "ready_by",
        "desired_for",
        "fulfillment_state",
        "provider_state",
        "schedule_state",
    ):
        op.drop_column("orders", column_name)

    op.drop_index("ix_meal_budgets_customer_id", table_name="meal_budgets")
    op.drop_index("ix_meal_budgets_restaurant_id", table_name="meal_budgets")
    op.drop_table("meal_budgets")
    op.drop_index("ix_meal_routines_next_materialization_at", table_name="meal_routines")
    op.drop_index("ix_meal_routines_customer_id", table_name="meal_routines")
    op.drop_index("ix_meal_routines_restaurant_id", table_name="meal_routines")
    op.drop_table("meal_routines")
    op.drop_index("ix_meal_templates_customer_id", table_name="meal_templates")
    op.drop_index("ix_meal_templates_restaurant_id", table_name="meal_templates")
    op.drop_table("meal_templates")
    op.drop_index("ix_meal_plans_customer_id", table_name="meal_plans")
    op.drop_index("ix_meal_plans_restaurant_id", table_name="meal_plans")
    op.drop_table("meal_plans")
