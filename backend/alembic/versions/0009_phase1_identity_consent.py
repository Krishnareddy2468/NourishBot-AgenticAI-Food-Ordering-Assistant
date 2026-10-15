"""phase 1 identity, consent, privacy, and memory controls

Revision ID: 0009
Revises: 0008
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0009"
down_revision: Union[str, Sequence[str], None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("channels", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("channels", sa.Column("verification_method", sa.Text(), nullable=True))
    op.add_column("channels", sa.Column("linked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "channels",
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.execute(
        "UPDATE channels SET verified_at = now(), verification_method = 'LEGACY_PLATFORM', "
        "linked_at = CASE WHEN customer_id IS NOT NULL THEN now() ELSE NULL END, last_seen_at = now()"
    )
    op.alter_column("channels", "last_seen_at", nullable=False)
    op.drop_constraint("uq_channel_type_external", "channels", type_="unique")
    op.create_unique_constraint(
        "uq_channel_restaurant_type_external",
        "channels",
        ["restaurant_id", "type", "external_id"],
    )

    op.create_table(
        "identity_link_challenges",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_channel_id", sa.BigInteger(), sa.ForeignKey("channels.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_channel_type", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("target_channel_id", sa.BigInteger(), sa.ForeignKey("channels.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_identity_link_challenges_restaurant_id", "identity_link_challenges", ["restaurant_id"])
    op.create_index("ix_identity_link_challenges_customer_id", "identity_link_challenges", ["customer_id"])
    op.create_index("ix_identity_link_challenges_expires_at", "identity_link_challenges", ["expires_at"])

    op.create_table(
        "customer_consents",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("granted", sa.Boolean(), nullable=False),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("source_channel_id", sa.BigInteger(), sa.ForeignKey("channels.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_customer_consents_restaurant_id", "customer_consents", ["restaurant_id"])
    op.create_index("ix_customer_consents_customer_id", "customer_consents", ["customer_id"])
    op.create_index("ix_customer_consents_purpose", "customer_consents", ["purpose"])
    op.create_index("ix_customer_consents_created_at", "customer_consents", ["created_at"])

    op.create_table(
        "privacy_requests",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("request_type", sa.Text(), nullable=False),
        sa.Column("source_channel_id", sa.BigInteger(), sa.ForeignKey("channels.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.Text(), server_default="COMPLETED", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_privacy_requests_restaurant_id", "privacy_requests", ["restaurant_id"])
    op.create_index("ix_privacy_requests_customer_id", "privacy_requests", ["customer_id"])

    op.create_table(
        "memory_learning_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("signal_type", sa.Text(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("order_id", "signal_type", name="uq_memory_learning_order_signal"),
    )
    op.create_index("ix_memory_learning_events_restaurant_id", "memory_learning_events", ["restaurant_id"])
    op.create_index("ix_memory_learning_events_customer_id", "memory_learning_events", ["customer_id"])
    op.create_index("ix_memory_learning_events_order_id", "memory_learning_events", ["order_id"])

    op.add_column("confirmation_challenges", sa.Column("created_by_channel_id", sa.BigInteger(), nullable=True))
    op.add_column("confirmation_challenges", sa.Column("created_by_channel_type", sa.Text(), nullable=True))
    op.add_column("confirmation_challenges", sa.Column("confirmed_by_channel_id", sa.BigInteger(), nullable=True))
    op.add_column("confirmation_challenges", sa.Column("invalidated_by_channel_id", sa.BigInteger(), nullable=True))
    op.add_column("confirmation_challenges", sa.Column("confirmation_actor", sa.Text(), nullable=True))
    op.add_column("confirmation_challenges", sa.Column("invalidation_reason", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_confirmation_created_channel", "confirmation_challenges", "channels",
        ["created_by_channel_id"], ["id"], ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_confirmation_confirmed_channel", "confirmation_challenges", "channels",
        ["confirmed_by_channel_id"], ["id"], ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_confirmation_invalidated_channel", "confirmation_challenges", "channels",
        ["invalidated_by_channel_id"], ["id"], ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_confirmation_invalidated_channel", "confirmation_challenges", type_="foreignkey")
    op.drop_constraint("fk_confirmation_confirmed_channel", "confirmation_challenges", type_="foreignkey")
    op.drop_constraint("fk_confirmation_created_channel", "confirmation_challenges", type_="foreignkey")
    for column in (
        "invalidation_reason", "confirmation_actor", "invalidated_by_channel_id",
        "confirmed_by_channel_id", "created_by_channel_type", "created_by_channel_id",
    ):
        op.drop_column("confirmation_challenges", column)

    op.drop_index("ix_memory_learning_events_order_id", table_name="memory_learning_events")
    op.drop_index("ix_memory_learning_events_customer_id", table_name="memory_learning_events")
    op.drop_index("ix_memory_learning_events_restaurant_id", table_name="memory_learning_events")
    op.drop_table("memory_learning_events")
    op.drop_index("ix_privacy_requests_customer_id", table_name="privacy_requests")
    op.drop_index("ix_privacy_requests_restaurant_id", table_name="privacy_requests")
    op.drop_table("privacy_requests")
    op.drop_index("ix_customer_consents_created_at", table_name="customer_consents")
    op.drop_index("ix_customer_consents_purpose", table_name="customer_consents")
    op.drop_index("ix_customer_consents_customer_id", table_name="customer_consents")
    op.drop_index("ix_customer_consents_restaurant_id", table_name="customer_consents")
    op.drop_table("customer_consents")
    op.drop_index("ix_identity_link_challenges_expires_at", table_name="identity_link_challenges")
    op.drop_index("ix_identity_link_challenges_customer_id", table_name="identity_link_challenges")
    op.drop_index("ix_identity_link_challenges_restaurant_id", table_name="identity_link_challenges")
    op.drop_table("identity_link_challenges")

    op.drop_constraint("uq_channel_restaurant_type_external", "channels", type_="unique")
    op.create_unique_constraint("uq_channel_type_external", "channels", ["type", "external_id"])
    op.drop_column("channels", "last_seen_at")
    op.drop_column("channels", "linked_at")
    op.drop_column("channels", "verification_method")
    op.drop_column("channels", "verified_at")
