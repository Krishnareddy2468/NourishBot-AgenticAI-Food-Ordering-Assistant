"""phase 0 order safety records and private tracking tokens

Revision ID: 0008
Revises: 0007
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "confirmation_challenges",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.BigInteger(), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id"), nullable=True),
        sa.Column("snapshot_hash", sa.Text(), nullable=False),
        sa.Column("snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_confirmation_challenges_restaurant_id", "confirmation_challenges", ["restaurant_id"])
    op.create_index("ix_confirmation_challenges_session_id", "confirmation_challenges", ["session_id"])
    op.create_index("ix_confirmation_challenges_customer_id", "confirmation_challenges", ["customer_id"])
    op.create_index("ix_confirmation_challenges_expires_at", "confirmation_challenges", ["expires_at"])

    op.create_table(
        "provider_attempts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("channel_type", sa.Text(), nullable=False),
        sa.Column("external_chat_id", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("request_fingerprint", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("provider_request_id", sa.Text(), nullable=True),
        sa.Column("provider_order_id", sa.Text(), nullable=True),
        sa.Column("response_excerpt", sa.Text(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reconciled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("provider", "idempotency_key", name="uq_provider_attempt_idempotency"),
    )
    op.create_index("ix_provider_attempts_restaurant_id", "provider_attempts", ["restaurant_id"])
    op.create_index("ix_provider_attempts_provider", "provider_attempts", ["provider"])
    op.create_index("ix_provider_attempts_external_chat_id", "provider_attempts", ["external_chat_id"])
    op.create_index("ix_provider_attempts_status", "provider_attempts", ["status"])

    op.add_column("orders", sa.Column("tracking_token", sa.Text(), nullable=True))
    op.add_column("orders", sa.Column("confirmation_id", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_orders_confirmation_id",
        "orders",
        "confirmation_challenges",
        ["confirmation_id"],
        ["id"],
    )
    op.execute(
        "UPDATE orders SET tracking_token = md5(random()::text || clock_timestamp()::text || id::text) "
        "WHERE tracking_token IS NULL"
    )
    op.alter_column("orders", "tracking_token", nullable=False)
    op.create_index("ix_orders_tracking_token", "orders", ["tracking_token"], unique=True)
    op.create_unique_constraint("uq_orders_confirmation_id", "orders", ["confirmation_id"])


def downgrade() -> None:
    op.drop_constraint("uq_orders_confirmation_id", "orders", type_="unique")
    op.drop_index("ix_orders_tracking_token", table_name="orders")
    op.drop_constraint("fk_orders_confirmation_id", "orders", type_="foreignkey")
    op.drop_column("orders", "confirmation_id")
    op.drop_column("orders", "tracking_token")
    op.drop_index("ix_provider_attempts_status", table_name="provider_attempts")
    op.drop_index("ix_provider_attempts_external_chat_id", table_name="provider_attempts")
    op.drop_index("ix_provider_attempts_provider", table_name="provider_attempts")
    op.drop_index("ix_provider_attempts_restaurant_id", table_name="provider_attempts")
    op.drop_table("provider_attempts")
    op.drop_index("ix_confirmation_challenges_expires_at", table_name="confirmation_challenges")
    op.drop_index("ix_confirmation_challenges_customer_id", table_name="confirmation_challenges")
    op.drop_index("ix_confirmation_challenges_session_id", table_name="confirmation_challenges")
    op.drop_index("ix_confirmation_challenges_restaurant_id", table_name="confirmation_challenges")
    op.drop_table("confirmation_challenges")
