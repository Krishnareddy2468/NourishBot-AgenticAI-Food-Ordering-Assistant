"""add_delivery_proof_columns

Adds proof-of-delivery fields used by the Delivery ORM model and admin
delivery routes.

Revision ID: 0004
Revises: 0003
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("deliveries", sa.Column("proof_url", sa.Text(), nullable=True))
    op.add_column("deliveries", sa.Column("proof_type", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("deliveries", "proof_type")
    op.drop_column("deliveries", "proof_url")
