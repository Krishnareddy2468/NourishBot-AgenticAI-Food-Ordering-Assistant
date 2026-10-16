"""phase 6 consented routes and route-aware fulfillment

Revision ID: 0015
Revises: 0014
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0015"
down_revision: Union[str, Sequence[str], None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "saved_locations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("address", sa.Text(), nullable=False),
        sa.Column("place_id", sa.Text()),
        sa.Column("latitude", sa.Float()),
        sa.Column("longitude", sa.Float()),
        sa.Column("consent_id", sa.BigInteger(), sa.ForeignKey("customer_consents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("latitude IS NULL OR latitude BETWEEN -90 AND 90", name="ck_saved_location_latitude"),
        sa.CheckConstraint("longitude IS NULL OR longitude BETWEEN -180 AND 180", name="ck_saved_location_longitude"),
        sa.CheckConstraint("retention_expires_at > created_at", name="ck_saved_location_retention"),
        sa.UniqueConstraint("restaurant_id", "customer_id", "label", name="uq_saved_location_customer_label"),
    )
    for name in ("restaurant_id", "customer_id", "retention_expires_at"):
        op.create_index(f"ix_saved_locations_{name}", "saved_locations", [name])

    op.create_table(
        "saved_routes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("origin_location_id", sa.BigInteger(), sa.ForeignKey("saved_locations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("destination_location_id", sa.BigInteger(), sa.ForeignKey("saved_locations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("route_fingerprint", sa.Text(), nullable=False),
        sa.Column("consent_id", sa.BigInteger(), sa.ForeignKey("customer_consents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("origin_location_id <> destination_location_id", name="ck_saved_route_distinct_locations"),
        sa.CheckConstraint("retention_expires_at > created_at", name="ck_saved_route_retention"),
        sa.UniqueConstraint("restaurant_id", "customer_id", "name", name="uq_saved_route_customer_name"),
    )
    for name in ("restaurant_id", "customer_id", "route_fingerprint", "retention_expires_at"):
        op.create_index(f"ix_saved_routes_{name}", "saved_routes", [name])

    op.create_table(
        "route_discoveries",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("restaurant_id", sa.BigInteger(), sa.ForeignKey("restaurants.id"), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("saved_route_id", sa.BigInteger(), sa.ForeignKey("saved_routes.id", ondelete="CASCADE")),
        sa.Column("consent_id", sa.BigInteger(), sa.ForeignKey("customer_consents.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("query_fingerprint", sa.Text(), nullable=False),
        sa.Column("requested_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("requested_fulfillment_type", sa.Text(), nullable=False),
        sa.Column("selected_place_id", sa.Text()),
        sa.Column("selected_restaurant_name", sa.Text()),
        sa.Column("resolved_fulfillment_type", sa.Text()),
        sa.Column("provider", sa.Text()),
        sa.Column("provider_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("discovery_status", sa.Text(), nullable=False),
        sa.Column("recovery_reason", sa.Text()),
        sa.Column("score_summary_json", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "requested_fulfillment_type IN ('DELIVERY','PICKUP','DRIVE_THROUGH')",
            name="ck_route_discovery_requested_fulfillment",
        ),
        sa.CheckConstraint(
            "resolved_fulfillment_type IS NULL OR resolved_fulfillment_type IN ('DELIVERY','PICKUP','DRIVE_THROUGH')",
            name="ck_route_discovery_resolved_fulfillment",
        ),
        sa.CheckConstraint("retention_expires_at > created_at", name="ck_route_discovery_retention"),
    )
    for name in ("restaurant_id", "customer_id", "saved_route_id", "retention_expires_at"):
        op.create_index(f"ix_route_discoveries_{name}", "route_discoveries", [name])


def downgrade() -> None:
    for name in ("retention_expires_at", "saved_route_id", "customer_id", "restaurant_id"):
        op.drop_index(f"ix_route_discoveries_{name}", table_name="route_discoveries")
    op.drop_table("route_discoveries")
    for name in ("retention_expires_at", "route_fingerprint", "customer_id", "restaurant_id"):
        op.drop_index(f"ix_saved_routes_{name}", table_name="saved_routes")
    op.drop_table("saved_routes")
    for name in ("retention_expires_at", "customer_id", "restaurant_id"):
        op.drop_index(f"ix_saved_locations_{name}", table_name="saved_locations")
    op.drop_table("saved_locations")
