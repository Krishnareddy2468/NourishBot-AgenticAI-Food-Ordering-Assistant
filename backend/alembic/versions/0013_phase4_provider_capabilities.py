"""phase 4 provider capability registry, modes, and receipts

Revision ID: 0013
Revises: 0012
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0013"
down_revision: Union[str, Sequence[str], None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "provider_capability_versions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("capabilities_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("tool_schema_hash", sa.Text()),
        sa.Column("tool_schema_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("adapter_version", sa.Text(), server_default="mcp-v1", nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("verified_by_user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="RESTRICT")),
        sa.Column("verification_notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "state IN ('UNVERIFIED','VERIFIED','DEGRADED','DISABLED')",
            name="ck_provider_capability_state",
        ),
        sa.CheckConstraint(
            "mode IN ('DISCOVERY','HANDOFF','TRANSACTIONAL')",
            name="ck_provider_capability_mode",
        ),
        sa.CheckConstraint("version > 0", name="ck_provider_capability_positive_version"),
        sa.UniqueConstraint("provider", "version", name="uq_provider_capability_version"),
    )
    op.create_index("ix_provider_capability_versions_provider", "provider_capability_versions", ["provider"])
    op.create_index("ix_provider_capability_versions_state", "provider_capability_versions", ["state"])
    op.create_index("ix_provider_capability_versions_mode", "provider_capability_versions", ["mode"])
    op.create_index("ix_provider_capability_versions_tool_schema_hash", "provider_capability_versions", ["tool_schema_hash"])

    for column in (
        sa.Column("mode", sa.Text(), server_default="DISCOVERY", nullable=False),
        sa.Column("capability_version_id", sa.BigInteger(), nullable=True),
        sa.Column("tool_schema_hash", sa.Text(), nullable=True),
        sa.Column("response_hash", sa.Text(), nullable=True),
        sa.Column("reconciliation_required", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("reconciliation_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_reconciliation_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_action", sa.Text(), nullable=True),
    ):
        op.add_column("provider_attempts", column)
    op.create_foreign_key(
        "fk_provider_attempts_capability_version_id",
        "provider_attempts",
        "provider_capability_versions",
        ["capability_version_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_provider_attempts_capability_version_id", "provider_attempts", ["capability_version_id"])
    op.execute(
        "UPDATE provider_attempts SET mode = 'TRANSACTIONAL' "
        "WHERE operation IN ('CONFIRM_ORDER','SUBMIT_ORDER','CREATE_ORDER')"
    )
    op.execute(
        "UPDATE provider_attempts SET reconciliation_required = true, "
        "next_action = 'RECONCILE_BEFORE_RETRY' WHERE status = 'UNKNOWN'"
    )

    op.create_table(
        "provider_receipts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("provider_attempt_id", sa.Text(), sa.ForeignKey("provider_attempts.id", ondelete="RESTRICT")),
        sa.Column(
            "capability_version_id",
            sa.BigInteger(),
            sa.ForeignKey("provider_capability_versions.id", ondelete="RESTRICT"),
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text()),
        sa.Column("request_fingerprint", sa.Text(), nullable=False),
        sa.Column("request_redacted_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("response_redacted_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("response_hash", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("provider_request_id", sa.Text()),
        sa.Column("provider_order_id", sa.Text()),
        sa.Column("tool_schema_hash", sa.Text()),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_provider_receipts_restaurant_id", "provider_receipts", ["restaurant_id"])
    op.create_index("ix_provider_receipts_provider_attempt_id", "provider_receipts", ["provider_attempt_id"])
    op.create_index("ix_provider_receipts_capability_version_id", "provider_receipts", ["capability_version_id"])
    op.create_index("ix_provider_receipts_provider", "provider_receipts", ["provider"])
    op.create_index("ix_provider_receipts_operation", "provider_receipts", ["operation"])
    op.create_index("ix_provider_receipts_response_hash", "provider_receipts", ["response_hash"])
    op.create_index("ix_provider_receipts_outcome", "provider_receipts", ["outcome"])
    op.create_index("ix_provider_receipts_occurred_at", "provider_receipts", ["occurred_at"])

    op.execute(
        """
        INSERT INTO provider_capability_versions
            (provider, version, state, mode, capabilities_json, tool_schema_hash,
             tool_schema_json, adapter_version, verified_at, verification_notes, created_at)
        VALUES
            ('NOURISHBOT', 1, 'VERIFIED', 'TRANSACTIONAL',
             jsonb_build_object(
                'discovery', true, 'menu_lookup', true, 'live_price_verification', true,
                'scheduled_ordering', true, 'scheduling_horizon_days', 14,
                'delivery', true, 'pickup', true, 'curbside_drive_through', true,
                'cancellation', true, 'order_status_lookup', true,
                'user_owned_authentication', true, 'stable_idempotency', true
             ),
             'internal-v1', '{}'::jsonb, 'internal-v1', now(),
             'In-house capability baseline.', now()),
            ('SWIGGY', 1, 'UNVERIFIED', 'DISCOVERY',
             jsonb_build_object('discovery', true), NULL, '{}'::jsonb, 'mcp-v1', NULL,
             'External provider defaults to discovery until live verification.', now()),
            ('ZOMATO', 1, 'UNVERIFIED', 'DISCOVERY',
             jsonb_build_object('discovery', true), NULL, '{}'::jsonb, 'mcp-v1', NULL,
             'External provider defaults to discovery until live verification.', now())
        """
    )

    op.execute(
        "CREATE TRIGGER trg_provider_capability_versions_append_only "
        "BEFORE UPDATE OR DELETE ON provider_capability_versions "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )
    op.execute(
        "CREATE TRIGGER trg_provider_receipts_append_only "
        "BEFORE UPDATE OR DELETE ON provider_receipts "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_provider_receipts_append_only ON provider_receipts")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_provider_capability_versions_append_only "
        "ON provider_capability_versions"
    )
    op.drop_index("ix_provider_receipts_occurred_at", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_outcome", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_response_hash", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_operation", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_provider", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_capability_version_id", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_provider_attempt_id", table_name="provider_receipts")
    op.drop_index("ix_provider_receipts_restaurant_id", table_name="provider_receipts")
    op.drop_table("provider_receipts")

    op.drop_index("ix_provider_attempts_capability_version_id", table_name="provider_attempts")
    op.drop_constraint(
        "fk_provider_attempts_capability_version_id",
        "provider_attempts",
        type_="foreignkey",
    )
    for column_name in (
        "next_action",
        "last_reconciliation_at",
        "reconciliation_count",
        "reconciliation_required",
        "response_hash",
        "tool_schema_hash",
        "capability_version_id",
        "mode",
    ):
        op.drop_column("provider_attempts", column_name)

    op.drop_index("ix_provider_capability_versions_tool_schema_hash", table_name="provider_capability_versions")
    op.drop_index("ix_provider_capability_versions_mode", table_name="provider_capability_versions")
    op.drop_index("ix_provider_capability_versions_state", table_name="provider_capability_versions")
    op.drop_index("ix_provider_capability_versions_provider", table_name="provider_capability_versions")
    op.drop_table("provider_capability_versions")
