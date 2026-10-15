"""add scheduled meal order fields

Revision ID: 0007
Revises: 0006
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "orders",
        sa.Column("provider", sa.Text(), nullable=False, server_default="NOURISHBOT"),
    )
    op.add_column(
        "orders",
        sa.Column("source_channel", sa.Text(), nullable=False, server_default="TELEGRAM"),
    )
    op.add_column("orders", sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True))
    op.add_column("orders", sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_orders_scheduled_for", "orders", ["scheduled_for"])
    op.create_index(
        "ix_orders_status_scheduled_for",
        "orders",
        ["status", "scheduled_for"],
    )


def downgrade() -> None:
    op.drop_index("ix_orders_status_scheduled_for", table_name="orders")
    op.drop_index("ix_orders_scheduled_for", table_name="orders")
    op.drop_column("orders", "activated_at")
    op.drop_column("orders", "scheduled_for")
    op.drop_column("orders", "source_channel")
    op.drop_column("orders", "provider")
