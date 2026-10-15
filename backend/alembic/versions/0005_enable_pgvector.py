"""enable_pgvector

Enables the pgvector PostgreSQL extension for storing and querying
vector embeddings (taste vectors, menu item embeddings, etc.).

Required for Sprint 1 (User Memory + Personalization).

Revision ID: 0005
Revises: 0004
"""
from typing import Sequence, Union

from alembic import op

revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Local PostgreSQL installs (especially Windows/macOS) may not package
    # pgvector. The current preference schema stores embeddings as Text, so
    # enable the extension when available without blocking all later schema.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector') THEN
                CREATE EXTENSION IF NOT EXISTS vector;
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
