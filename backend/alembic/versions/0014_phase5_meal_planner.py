"""phase 5 meal planner and recurring routines

Revision ID: 0014
Revises: 0013
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for column in (
        sa.Column("scope", sa.Text(), server_default="WEEKLY", nullable=False),
        sa.Column("constraints_json", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("estimated_total_cents", sa.Integer(), server_default="0", nullable=False),
        sa.Column("currency", sa.Text(), server_default="INR", nullable=False),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("confirmation_snapshot_hash", sa.Text()),
        sa.Column("confirmed_at", sa.DateTime(timezone=True)),
    ):
        op.add_column("meal_plans", column)
    op.execute(
        "UPDATE meal_plans SET scope = CASE WHEN starts_on = ends_on THEN 'DAILY' ELSE 'WEEKLY' END"
    )
    op.create_check_constraint("ck_meal_plans_scope", "meal_plans", "scope IN ('DAILY','WEEKLY')")
    op.create_check_constraint(
        "ck_meal_plans_estimated_total",
        "meal_plans",
        "estimated_total_cents >= 0",
    )

    for column in (
        sa.Column("time_of_day", sa.Time(), server_default="12:30:00", nullable=False),
        sa.Column("materialization_horizon_days", sa.Integer(), server_default="14", nullable=False),
        sa.Column("materialized_through", sa.Date()),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
    ):
        op.add_column("meal_routines", column)
    op.execute(
        "UPDATE meal_routines SET meal_period = CASE "
        "WHEN upper(meal_period) IN ('BREAKFAST','LUNCH','SNACKS','DINNER') "
        "THEN upper(meal_period) ELSE 'LUNCH' END"
    )
    op.create_check_constraint(
        "ck_meal_routine_horizon",
        "meal_routines",
        "materialization_horizon_days BETWEEN 1 AND 31",
    )
    op.create_check_constraint(
        "ck_meal_routine_period",
        "meal_routines",
        "meal_period IN ('BREAKFAST','LUNCH','SNACKS','DINNER')",
    )

    for column in (
        sa.Column("menu_item_id", sa.BigInteger()),
        sa.Column("selection_json", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("constraints_json", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("estimated_cost_cents", sa.Integer(), server_default="0", nullable=False),
        sa.Column("currency", sa.Text(), server_default="INR", nullable=False),
        sa.Column("diet_tags", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
        sa.Column("allergen_tags", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
        sa.Column("occurrence_key", sa.Text()),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True)),
        sa.Column("skipped_at", sa.DateTime(timezone=True)),
        sa.Column("rescheduled_from", sa.DateTime(timezone=True)),
    ):
        op.add_column("meal_schedules", column)
    op.create_foreign_key(
        "fk_meal_schedules_menu_item_id",
        "meal_schedules",
        "menu_items",
        ["menu_item_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_meal_schedules_menu_item_id", "meal_schedules", ["menu_item_id"])
    op.create_index(
        "ix_meal_schedules_occurrence_key",
        "meal_schedules",
        ["occurrence_key"],
        unique=True,
    )
    op.create_check_constraint(
        "ck_meal_schedules_estimated_cost",
        "meal_schedules",
        "estimated_cost_cents >= 0",
    )

    op.create_table(
        "meal_routine_exceptions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column(
            "routine_id",
            sa.BigInteger(),
            sa.ForeignKey("meal_routines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("occurrence_date", sa.Date(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("replacement_for", sa.DateTime(timezone=True)),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("action IN ('SKIP','RESCHEDULE')", name="ck_meal_routine_exception_action"),
        sa.CheckConstraint(
            "(action = 'SKIP' AND replacement_for IS NULL) OR "
            "(action = 'RESCHEDULE' AND replacement_for IS NOT NULL)",
            name="ck_meal_routine_exception_replacement",
        ),
        sa.UniqueConstraint("routine_id", "occurrence_date", name="uq_meal_routine_exception_date"),
    )
    op.create_index("ix_meal_routine_exceptions_restaurant_id", "meal_routine_exceptions", ["restaurant_id"])
    op.create_index("ix_meal_routine_exceptions_routine_id", "meal_routine_exceptions", ["routine_id"])

    op.create_table(
        "meal_planning_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("meal_plan_id", sa.BigInteger(), sa.ForeignKey("meal_plans.id", ondelete="CASCADE")),
        sa.Column("meal_schedule_id", sa.BigInteger(), sa.ForeignKey("meal_schedules.id", ondelete="CASCADE")),
        sa.Column("routine_id", sa.BigInteger(), sa.ForeignKey("meal_routines.id", ondelete="CASCADE")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text()),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    for name, columns in (
        ("restaurant_id", ["restaurant_id"]),
        ("customer_id", ["customer_id"]),
        ("meal_plan_id", ["meal_plan_id"]),
        ("meal_schedule_id", ["meal_schedule_id"]),
        ("routine_id", ["routine_id"]),
        ("event_type", ["event_type"]),
        ("occurred_at", ["occurred_at"]),
    ):
        op.create_index(f"ix_meal_planning_events_{name}", "meal_planning_events", columns)
    op.execute(
        "CREATE TRIGGER trg_meal_planning_events_append_only "
        "BEFORE UPDATE OR DELETE ON meal_planning_events "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_meal_planning_events_append_only ON meal_planning_events")
    for name in (
        "occurred_at",
        "event_type",
        "routine_id",
        "meal_schedule_id",
        "meal_plan_id",
        "customer_id",
        "restaurant_id",
    ):
        op.drop_index(f"ix_meal_planning_events_{name}", table_name="meal_planning_events")
    op.drop_table("meal_planning_events")
    op.drop_index("ix_meal_routine_exceptions_routine_id", table_name="meal_routine_exceptions")
    op.drop_index("ix_meal_routine_exceptions_restaurant_id", table_name="meal_routine_exceptions")
    op.drop_table("meal_routine_exceptions")

    op.drop_constraint("ck_meal_schedules_estimated_cost", "meal_schedules", type_="check")
    op.drop_index("ix_meal_schedules_occurrence_key", table_name="meal_schedules")
    op.drop_index("ix_meal_schedules_menu_item_id", table_name="meal_schedules")
    op.drop_constraint("fk_meal_schedules_menu_item_id", "meal_schedules", type_="foreignkey")
    for column_name in (
        "rescheduled_from",
        "skipped_at",
        "confirmed_at",
        "revision",
        "occurrence_key",
        "allergen_tags",
        "diet_tags",
        "currency",
        "estimated_cost_cents",
        "constraints_json",
        "selection_json",
        "menu_item_id",
    ):
        op.drop_column("meal_schedules", column_name)

    op.drop_constraint("ck_meal_routine_period", "meal_routines", type_="check")
    op.drop_constraint("ck_meal_routine_horizon", "meal_routines", type_="check")
    for column_name in ("version", "materialized_through", "materialization_horizon_days", "time_of_day"):
        op.drop_column("meal_routines", column_name)

    op.drop_constraint("ck_meal_plans_estimated_total", "meal_plans", type_="check")
    op.drop_constraint("ck_meal_plans_scope", "meal_plans", type_="check")
    for column_name in (
        "confirmed_at",
        "confirmation_snapshot_hash",
        "revision",
        "currency",
        "estimated_total_cents",
        "constraints_json",
        "scope",
    ):
        op.drop_column("meal_plans", column_name)
