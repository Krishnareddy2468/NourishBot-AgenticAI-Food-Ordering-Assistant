"""add_session_ordering_platform

Adds ordering_platform column to sessions table so the Telegram bot
can persist the user's platform choice (Dzukku / Zomato / Swiggy).

Revision ID: 0003
Revises: 0002
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'sessions',
        sa.Column('ordering_platform', sa.Text(), server_default=sa.text("''"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('sessions', 'ordering_platform')
