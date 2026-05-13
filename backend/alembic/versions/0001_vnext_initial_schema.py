"""vnext_initial_schema

Revision ID: 0001
Revises:
Create Date: 2026-05-02 13:33:58.324851

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, NUMERIC as Numeric


revision: str = '0001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Independent tables (no FKs) ──────────────────────────────────────

    op.create_table('restaurants',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('phone', sa.Text()),
        sa.Column('address', sa.Text()),
        sa.Column('timezone', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    op.create_table('dining_tables',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('capacity', sa.Integer(), nullable=False),
        sa.Column('active', sa.Boolean()),
    )

    op.create_table('modifier_groups',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('min_select', sa.Integer()),
        sa.Column('max_select', sa.Integer()),
    )

    op.create_table('outbox_events',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('event_type', sa.Text(), nullable=False),
        sa.Column('payload', JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('processed_at', sa.DateTime(timezone=True)),
    )

    # ── Core tables (FK -> restaurants) ──────────────────────────────────

    op.create_table('users',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), sa.ForeignKey('restaurants.id'), nullable=False, server_default='1'),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('phone', sa.Text()),
        sa.Column('email', sa.Text(), unique=True),
        sa.Column('password_hash', sa.Text(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('active', sa.Boolean()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    op.create_table('customers',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), sa.ForeignKey('restaurants.id'), nullable=False, server_default='1'),
        sa.Column('name', sa.Text()),
        sa.Column('phone', sa.Text(), nullable=False),
        sa.Column('email', sa.Text()),
        sa.Column('language_pref', sa.Text()),
        sa.Column('marketing_opt_in', sa.Boolean()),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default='now()'),
        sa.UniqueConstraint('restaurant_id', 'phone', name='uq_customer_restaurant_phone'),
    )

    op.create_table('menu_categories',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), sa.ForeignKey('restaurants.id'), nullable=False, server_default='1'),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('sort_order', sa.Integer()),
    )

    # ── Menu tables ──────────────────────────────────────────────────────

    op.create_table('modifiers',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('group_id', sa.BigInteger(), sa.ForeignKey('modifier_groups.id'), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('price_cents', sa.Integer()),
        sa.Column('available', sa.Boolean()),
    )

    op.create_table('menu_items',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), sa.ForeignKey('restaurants.id'), nullable=False, server_default='1'),
        sa.Column('category_id', sa.BigInteger(), sa.ForeignKey('menu_categories.id')),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('type', sa.Text()),
        sa.Column('price_cents', sa.Integer(), nullable=False),
        sa.Column('special_price_cents', sa.Integer()),
        sa.Column('available', sa.Boolean()),
        sa.Column('stock_qty', sa.Integer()),
        sa.Column('prep_time_sec', sa.Integer()),
        sa.Column('tags', ARRAY(sa.Text())),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    op.create_table('menu_item_images',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('item_id', sa.BigInteger(), sa.ForeignKey('menu_items.id'), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('alt_text', sa.Text()),
        sa.Column('sort_order', sa.Integer()),
        sa.Column('checksum', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    op.create_table('menu_item_modifier_groups',
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('item_id', sa.BigInteger(), sa.ForeignKey('menu_items.id'), primary_key=True),
        sa.Column('group_id', sa.BigInteger(), sa.ForeignKey('modifier_groups.id'), primary_key=True),
        sa.UniqueConstraint('item_id', 'group_id', name='uq_menu_item_modifier_group'),
    )

    # ── Channels ─────────────────────────────────────────────────────────

    op.create_table('channels',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('type', sa.Text(), nullable=False),
        sa.Column('external_id', sa.Text(), nullable=False),
        sa.Column('customer_id', sa.BigInteger(), sa.ForeignKey('customers.id')),
        sa.UniqueConstraint('type', 'external_id', name='uq_channel_type_external'),
    )

    # ── Carts + Cart Items ───────────────────────────────────────────────

    op.create_table('carts',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('customer_id', sa.BigInteger(), sa.ForeignKey('customers.id')),
        sa.Column('status', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    op.create_table('cart_items',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('cart_id', sa.BigInteger(), sa.ForeignKey('carts.id'), nullable=False),
        sa.Column('item_id', sa.BigInteger(), sa.ForeignKey('menu_items.id'), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('unit_price_cents', sa.Integer(), nullable=False),
        sa.Column('modifiers_json', JSONB(astext_type=sa.Text())),
    )

    # ── Sessions ─────────────────────────────────────────────────────────

    op.create_table('sessions',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('channel_id', sa.BigInteger(), sa.ForeignKey('channels.id'), nullable=False),
        sa.Column('state', sa.Text(), nullable=False),
        sa.Column('cart_id', sa.BigInteger(), sa.ForeignKey('carts.id')),
        sa.Column('history_json', JSONB(astext_type=sa.Text())),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    # ── Drivers ──────────────────────────────────────────────────────────

    op.create_table('drivers',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('user_id', sa.BigInteger(), sa.ForeignKey('users.id'), unique=True),
        sa.Column('vehicle_type', sa.Text()),
        sa.Column('vehicle_no', sa.Text()),
        sa.Column('active', sa.Boolean()),
    )

    # ── Orders + Order Items ─────────────────────────────────────────────

    op.create_table('orders',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('order_ref', sa.Text(), nullable=False, unique=True),
        sa.Column('customer_id', sa.BigInteger(), sa.ForeignKey('customers.id')),
        sa.Column('channel_id', sa.BigInteger(), sa.ForeignKey('channels.id')),
        sa.Column('order_type', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('subtotal_cents', sa.Integer(), nullable=False),
        sa.Column('tax_cents', sa.Integer()),
        sa.Column('packing_cents', sa.Integer()),
        sa.Column('discount_cents', sa.Integer()),
        sa.Column('total_cents', sa.Integer(), nullable=False),
        sa.Column('eta_ts', sa.DateTime(timezone=True)),
        sa.Column('idempotency_key', sa.Text(), unique=True),
        sa.Column('notes', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    op.create_table('order_items',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('order_id', sa.BigInteger(), sa.ForeignKey('orders.id'), nullable=False),
        sa.Column('item_id', sa.BigInteger(), sa.ForeignKey('menu_items.id'), nullable=False),
        sa.Column('item_name_snapshot', sa.Text(), nullable=False),
        sa.Column('qty', sa.Integer(), nullable=False),
        sa.Column('unit_price_cents', sa.Integer(), nullable=False),
        sa.Column('modifiers_json', JSONB(astext_type=sa.Text())),
        sa.Column('status', sa.Text()),
    )

    # ── Payments ─────────────────────────────────────────────────────────

    op.create_table('payments',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('order_id', sa.BigInteger(), sa.ForeignKey('orders.id'), nullable=False),
        sa.Column('provider', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('amount_cents', sa.Integer(), nullable=False),
        sa.Column('currency', sa.Text()),
        sa.Column('provider_order_id', sa.Text()),
        sa.Column('provider_payment_id', sa.Text()),
        sa.Column('provider_signature', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    # ── Deliveries ───────────────────────────────────────────────────────

    op.create_table('deliveries',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('order_id', sa.BigInteger(), sa.ForeignKey('orders.id'), nullable=False, unique=True),
        sa.Column('driver_id', sa.BigInteger(), sa.ForeignKey('drivers.id')),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('address_json', JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('customer_phone', sa.Text()),
        sa.Column('assigned_at', sa.DateTime(timezone=True)),
        sa.Column('picked_up_at', sa.DateTime(timezone=True)),
        sa.Column('delivered_at', sa.DateTime(timezone=True)),
    )

    op.create_table('delivery_location_events',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('delivery_id', sa.BigInteger(), sa.ForeignKey('deliveries.id'), nullable=False),
        sa.Column('lat', Numeric(9, 6), nullable=False),
        sa.Column('lng', Numeric(9, 6), nullable=False),
        sa.Column('accuracy_m', sa.Integer()),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False, server_default='now()'),
    )

    # ── Table sessions + association ─────────────────────────────────────

    op.create_table('table_sessions',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('table_id', sa.BigInteger(), sa.ForeignKey('dining_tables.id'), nullable=False),
        sa.Column('waiter_user_id', sa.BigInteger(), sa.ForeignKey('users.id')),
        sa.Column('guests', sa.Integer(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('opened_at', sa.DateTime(timezone=True), server_default='now()'),
        sa.Column('closed_at', sa.DateTime(timezone=True)),
        sa.Column('notes', sa.Text()),
    )

    op.create_table('table_session_orders',
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('table_session_id', sa.BigInteger(), sa.ForeignKey('table_sessions.id'), primary_key=True),
        sa.Column('order_id', sa.BigInteger(), sa.ForeignKey('orders.id'), primary_key=True),
        sa.UniqueConstraint('table_session_id', 'order_id', name='uq_table_session_order'),
    )

    # ── Reservations ─────────────────────────────────────────────────────

    op.create_table('reservations',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('reservation_ref', sa.Text(), nullable=False, unique=True),
        sa.Column('customer_id', sa.BigInteger(), sa.ForeignKey('customers.id')),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('time', sa.Time(), nullable=False),
        sa.Column('guests', sa.Integer(), nullable=False),
        sa.Column('special_request', sa.Text()),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    # ── Invoices ─────────────────────────────────────────────────────────

    op.create_table('invoices',
        sa.Column('id', sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column('restaurant_id', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('invoice_no', sa.Text(), nullable=False, unique=True),
        sa.Column('entity_type', sa.Text(), nullable=False),
        sa.Column('entity_id', sa.BigInteger(), nullable=False),
        sa.Column('subtotal_cents', sa.Integer(), nullable=False),
        sa.Column('tax_cents', sa.Integer()),
        sa.Column('total_cents', sa.Integer(), nullable=False),
        sa.Column('pdf_url', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default='now()'),
    )

    # ── Indexes on restaurant_id for all multi-tenant tables ─────────────

    for table in [
        'users', 'customers', 'channels', 'sessions',
        'menu_categories', 'menu_items', 'menu_item_images',
        'modifier_groups', 'modifiers', 'menu_item_modifier_groups',
        'carts', 'cart_items', 'orders', 'order_items',
        'drivers', 'deliveries', 'delivery_location_events',
        'payments', 'dining_tables', 'table_sessions', 'table_session_orders',
        'reservations', 'invoices', 'outbox_events',
    ]:
        op.create_index(f'ix_{table}_restaurant_id', table, ['restaurant_id'])


def downgrade() -> None:
    for table in reversed([
        'delivery_location_events', 'deliveries',
        'table_session_orders', 'table_sessions',
        'order_items', 'payments',
        'cart_items', 'carts',
        'sessions',
        'menu_item_modifier_groups', 'menu_item_images',
        'modifiers', 'modifier_groups',
        'channels',
        'drivers',
        'reservations', 'invoices',
        'orders',
        'menu_items', 'menu_categories',
        'customers', 'users',
        'dining_tables', 'outbox_events',
        'restaurants',
    ]):
        op.drop_table(table)
