"""Consent-gated saved locations, routes, and minimal discovery history."""

from sqlalchemy import BigInteger, Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text

from app.db.base import Base, RestaurantMixin


class SavedLocation(Base, RestaurantMixin):
    __tablename__ = "saved_locations"
    __table_args__ = (
        UniqueConstraint("restaurant_id", "customer_id", "label", name="uq_saved_location_customer_label"),
        CheckConstraint("latitude IS NULL OR latitude BETWEEN -90 AND 90", name="ck_saved_location_latitude"),
        CheckConstraint("longitude IS NULL OR longitude BETWEEN -180 AND 180", name="ck_saved_location_longitude"),
        CheckConstraint("retention_expires_at > created_at", name="ck_saved_location_retention"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(Text, nullable=False)
    address = Column(Text, nullable=False)
    place_id = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    consent_id = Column(BigInteger, ForeignKey("customer_consents.id", ondelete="RESTRICT"), nullable=False)
    retention_expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class SavedRoute(Base, RestaurantMixin):
    __tablename__ = "saved_routes"
    __table_args__ = (
        UniqueConstraint("restaurant_id", "customer_id", "name", name="uq_saved_route_customer_name"),
        CheckConstraint("origin_location_id <> destination_location_id", name="ck_saved_route_distinct_locations"),
        CheckConstraint("retention_expires_at > created_at", name="ck_saved_route_retention"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    origin_location_id = Column(BigInteger, ForeignKey("saved_locations.id", ondelete="CASCADE"), nullable=False)
    destination_location_id = Column(BigInteger, ForeignKey("saved_locations.id", ondelete="CASCADE"), nullable=False)
    route_fingerprint = Column(Text, nullable=False, index=True)
    consent_id = Column(BigInteger, ForeignKey("customer_consents.id", ondelete="RESTRICT"), nullable=False)
    active = Column(Boolean, nullable=False, server_default=text("true"))
    retention_expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class RouteDiscovery(Base, RestaurantMixin):
    """Minimal retained decision history; raw Maps payloads are never persisted."""

    __tablename__ = "route_discoveries"
    __table_args__ = (
        CheckConstraint(
            "requested_fulfillment_type IN ('DELIVERY','PICKUP','DRIVE_THROUGH')",
            name="ck_route_discovery_requested_fulfillment",
        ),
        CheckConstraint(
            "resolved_fulfillment_type IS NULL OR resolved_fulfillment_type IN ('DELIVERY','PICKUP','DRIVE_THROUGH')",
            name="ck_route_discovery_resolved_fulfillment",
        ),
        CheckConstraint("retention_expires_at > created_at", name="ck_route_discovery_retention"),
    )

    id = Column(Text, primary_key=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    saved_route_id = Column(BigInteger, ForeignKey("saved_routes.id", ondelete="CASCADE"), index=True)
    consent_id = Column(BigInteger, ForeignKey("customer_consents.id", ondelete="RESTRICT"), nullable=False)
    query_fingerprint = Column(Text, nullable=False)
    requested_for = Column(DateTime(timezone=True), nullable=False)
    requested_fulfillment_type = Column(Text, nullable=False)
    selected_place_id = Column(Text)
    selected_restaurant_name = Column(Text)
    resolved_fulfillment_type = Column(Text)
    provider = Column(Text)
    provider_verified = Column(Boolean, nullable=False, server_default=text("false"))
    discovery_status = Column(Text, nullable=False)
    recovery_reason = Column(Text)
    score_summary_json = Column(JSONB, nullable=False, default=dict)
    retention_expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
