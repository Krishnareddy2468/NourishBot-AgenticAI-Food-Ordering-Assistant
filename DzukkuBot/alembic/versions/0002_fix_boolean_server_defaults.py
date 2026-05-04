"""fix_boolean_server_defaults

Adds missing DB-level server defaults that were only set as Python-side
defaults in the ORM (had no effect on raw SQL INSERTs or other clients).

Changes:
  restaurants  : timezone server_default = 'Asia/Kolkata'
  users        : active server_default = true, NOT NULL
  customers    : marketing_opt_in server_default = false, NOT NULL
                 language_pref server_default = 'en'
  dining_tables: active server_default = true, NOT NULL
  menu_items   : available server_default = true, NOT NULL
  modifiers    : available server_default = true, NOT NULL
                 price_cents server_default = 0, NOT NULL
  drivers      : active server_default = true, NOT NULL
  carts        : status server_default = 'OPEN', NOT NULL
  order_items  : status server_default = 'PENDING', NOT NULL
  payments     : provider server_default = 'RAZORPAY'
                 status  server_default = 'CREATED'
                 currency server_default = 'INR', NOT NULL

Revision ID: 0002
Revises: 0001
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── restaurants ───────────────────────────────────────────────────────────
    op.alter_column('restaurants', 'timezone',
                    server_default=sa.text("'Asia/Kolkata'"))

    # ── users ─────────────────────────────────────────────────────────────────
    # Back-fill NULLs before adding NOT NULL
    op.execute("UPDATE users SET active = true WHERE active IS NULL")
    op.alter_column('users', 'active',
                    nullable=False,
                    server_default=sa.text("true"))

    # ── customers ─────────────────────────────────────────────────────────────
    op.execute("UPDATE customers SET marketing_opt_in = false WHERE marketing_opt_in IS NULL")
    op.alter_column('customers', 'marketing_opt_in',
                    nullable=False,
                    server_default=sa.text("false"))

    op.alter_column('customers', 'language_pref',
                    server_default=sa.text("'en'"))

    # ── dining_tables ─────────────────────────────────────────────────────────
    op.execute("UPDATE dining_tables SET active = true WHERE active IS NULL")
    op.alter_column('dining_tables', 'active',
                    nullable=False,
                    server_default=sa.text("true"))

    # ── menu_items ────────────────────────────────────────────────────────────
    op.execute("UPDATE menu_items SET available = true WHERE available IS NULL")
    op.alter_column('menu_items', 'available',
                    nullable=False,
                    server_default=sa.text("true"))

    # ── modifiers ─────────────────────────────────────────────────────────────
    op.execute("UPDATE modifiers SET available = true WHERE available IS NULL")
    op.alter_column('modifiers', 'available',
                    nullable=False,
                    server_default=sa.text("true"))

    op.execute("UPDATE modifiers SET price_cents = 0 WHERE price_cents IS NULL")
    op.alter_column('modifiers', 'price_cents',
                    nullable=False,
                    server_default=sa.text("0"))

    # ── drivers ───────────────────────────────────────────────────────────────
    op.execute("UPDATE drivers SET active = true WHERE active IS NULL")
    op.alter_column('drivers', 'active',
                    nullable=False,
                    server_default=sa.text("true"))

    # ── carts ─────────────────────────────────────────────────────────────────
    op.execute("UPDATE carts SET status = 'OPEN' WHERE status IS NULL")
    op.alter_column('carts', 'status',
                    nullable=False,
                    server_default=sa.text("'OPEN'"))

    # ── order_items ───────────────────────────────────────────────────────────
    op.execute("UPDATE order_items SET status = 'PENDING' WHERE status IS NULL")
    op.alter_column('order_items', 'status',
                    nullable=False,
                    server_default=sa.text("'PENDING'"))

    # ── payments ──────────────────────────────────────────────────────────────
    op.alter_column('payments', 'provider',
                    server_default=sa.text("'RAZORPAY'"))
    op.alter_column('payments', 'status',
                    server_default=sa.text("'CREATED'"))

    op.execute("UPDATE payments SET currency = 'INR' WHERE currency IS NULL")
    op.alter_column('payments', 'currency',
                    nullable=False,
                    server_default=sa.text("'INR'"))


def downgrade() -> None:
    # Remove server defaults (revert to nullable without default)
    for table, column in [
        ('restaurants',   'timezone'),
        ('users',         'active'),
        ('customers',     'marketing_opt_in'),
        ('customers',     'language_pref'),
        ('dining_tables', 'active'),
        ('menu_items',    'available'),
        ('modifiers',     'available'),
        ('modifiers',     'price_cents'),
        ('drivers',       'active'),
        ('carts',         'status'),
        ('order_items',   'status'),
        ('payments',      'provider'),
        ('payments',      'status'),
        ('payments',      'currency'),
    ]:
        op.alter_column(table, column, server_default=None)
