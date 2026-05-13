"""add_user_preferences_and_rating

Creates user_preferences table for personalization and adds
a rating column to orders for post-delivery feedback.

Revision ID: 0006
Revises: 0005
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

revision: str = '0006'
down_revision: Union[str, Sequence[str], None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── user_preferences ──────────────────────────────────────────────────
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=False),
        sa.Column("spice_level", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("cuisine_weights", JSONB(), server_default="{}", nullable=False),
        sa.Column("price_band", sa.Text(), server_default="mid"),
        sa.Column("order_timing", JSONB(), server_default="{}", nullable=False),
        sa.Column("platform_preference", sa.Text(), server_default="direct"),
        sa.Column("dietary_flags", ARRAY(sa.Text()), server_default="{}"),
        sa.Column("health_goals", ARRAY(sa.Text()), server_default="{}"),
        sa.Column("allergies", ARRAY(sa.Text()), server_default="{}"),
        sa.Column("health_onboarding_done", sa.Integer(), server_default="0", nullable=False),
        sa.Column("craving_cycles", JSONB(), server_default="{}", nullable=False),
        sa.Column("weather_behavior", JSONB(), server_default="{}"),
        sa.Column("taste_embedding", sa.Text(), nullable=True),
        sa.Column("total_orders", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_spent_cents", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("last_order_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("customer_id"),
    )
    op.create_index("ix_user_preferences_customer_id", "user_preferences", ["customer_id"])

    # ── orders.rating ─────────────────────────────────────────────────────
    op.add_column("orders", sa.Column("rating", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("orders", "rating")
    op.drop_index("ix_user_preferences_customer_id", table_name="user_preferences")
    op.drop_table("user_preferences")
