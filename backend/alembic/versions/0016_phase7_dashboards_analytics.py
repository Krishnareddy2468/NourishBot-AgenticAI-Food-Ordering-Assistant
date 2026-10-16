"""phase 7 customer dashboards, timestamps, and spending ledger

Revision ID: 0016
Revises: 0015
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0016"
down_revision: Union[str, Sequence[str], None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("customer_id", sa.BigInteger()))
    op.create_foreign_key(
        "fk_users_customer_id",
        "users",
        "customers",
        ["customer_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_users_customer_id", "users", ["customer_id"], unique=True)

    op.execute(
        "UPDATE users u SET customer_id = matched.id FROM ("
        "SELECT DISTINCT ON (u2.id) u2.id AS user_id, c.id "
        "FROM users u2 JOIN customers c ON c.restaurant_id = u2.restaurant_id "
        "AND lower(c.email) = lower(u2.email) WHERE u2.role = 'CUSTOMER' "
        "ORDER BY u2.id, c.id"
        ") matched WHERE u.id = matched.user_id"
    )
    op.execute(
        "INSERT INTO customers (restaurant_id, name, phone, email) "
        "SELECT u.restaurant_id, u.name, 'web-account-' || u.id::text, u.email "
        "FROM users u WHERE u.role = 'CUSTOMER' AND u.customer_id IS NULL"
    )
    op.execute(
        "UPDATE users u SET customer_id = c.id FROM customers c "
        "WHERE u.role = 'CUSTOMER' AND u.customer_id IS NULL "
        "AND c.restaurant_id = u.restaurant_id AND c.phone = 'web-account-' || u.id::text"
    )
    op.create_check_constraint(
        "ck_customer_user_has_profile",
        "users",
        "role <> 'CUSTOMER' OR customer_id IS NOT NULL",
    )
    op.execute(
        "INSERT INTO channels (restaurant_id, type, external_id, customer_id, verified_at, verification_method, linked_at) "
        "SELECT u.restaurant_id, 'WEB', 'account:' || u.id::text, u.customer_id, now(), "
        "'AUTHENTICATED_WEB_ACCOUNT', now() FROM users u "
        "WHERE u.role = 'CUSTOMER' AND u.customer_id IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM channels ch WHERE ch.restaurant_id = u.restaurant_id "
        "AND ch.type = 'WEB' AND ch.external_id = 'account:' || u.id::text)"
    )

    op.add_column("orders", sa.Column("submitted_at", sa.DateTime(timezone=True)))
    op.add_column("orders", sa.Column("completed_at", sa.DateTime(timezone=True)))
    op.create_index("ix_orders_submitted_at", "orders", ["submitted_at"])
    op.create_index("ix_orders_completed_at", "orders", ["completed_at"])
    op.execute(
        "UPDATE orders SET submitted_at = COALESCE(activated_at, created_at) "
        "WHERE status IN ('ACCEPTED','PREPARING','READY','OUT_FOR_DELIVERY','DELIVERED','COMPLETED')"
    )
    op.execute(
        "UPDATE orders SET completed_at = COALESCE(updated_at, created_at) "
        "WHERE status IN ('DELIVERED','COMPLETED')"
    )

    op.create_table(
        "spending_ledger_entries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("order_id", sa.BigInteger(), sa.ForeignKey("orders.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("entry_type", sa.Text(), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.Text(), server_default="INR", nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("amount_cents <> 0", name="ck_spending_ledger_nonzero_amount"),
        sa.CheckConstraint(
            "entry_type IN ('COMPLETED_SPEND','REFUND_CREDIT')",
            name="ck_spending_ledger_entry_type",
        ),
        sa.UniqueConstraint("order_id", "entry_type", name="uq_spending_ledger_order_type"),
    )
    for name in ("restaurant_id", "customer_id", "order_id", "entry_type", "occurred_at"):
        op.create_index(f"ix_spending_ledger_entries_{name}", "spending_ledger_entries", [name])
    op.execute(
        "INSERT INTO spending_ledger_entries "
        "(restaurant_id, customer_id, order_id, entry_type, amount_cents, currency, metadata_json, occurred_at) "
        "SELECT restaurant_id, customer_id, id, 'COMPLETED_SPEND', total_cents, 'INR', "
        "jsonb_build_object('source', 'PHASE7_BACKFILL'), completed_at FROM orders "
        "WHERE customer_id IS NOT NULL AND completed_at IS NOT NULL AND total_cents <> 0"
    )
    op.execute(
        "CREATE TRIGGER trg_spending_ledger_append_only "
        "BEFORE UPDATE OR DELETE ON spending_ledger_entries "
        "FOR EACH ROW EXECUTE FUNCTION nourishbot_reject_event_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_spending_ledger_append_only ON spending_ledger_entries")
    for name in ("occurred_at", "entry_type", "order_id", "customer_id", "restaurant_id"):
        op.drop_index(f"ix_spending_ledger_entries_{name}", table_name="spending_ledger_entries")
    op.drop_table("spending_ledger_entries")

    op.drop_index("ix_orders_completed_at", table_name="orders")
    op.drop_index("ix_orders_submitted_at", table_name="orders")
    op.drop_column("orders", "completed_at")
    op.drop_column("orders", "submitted_at")

    op.drop_constraint("ck_customer_user_has_profile", "users", type_="check")
    op.drop_index("ix_users_customer_id", table_name="users")
    op.drop_constraint("fk_users_customer_id", "users", type_="foreignkey")
    op.drop_column("users", "customer_id")
