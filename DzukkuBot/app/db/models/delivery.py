"""
Delivery models: Driver, Delivery, DeliveryLocationEvent.
"""

from sqlalchemy import Column, BigInteger, Text, Boolean, ForeignKey, Numeric, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


# ── Driver ────────────────────────────────────────────────────────────────────

class Driver(Base, RestaurantMixin):
    __tablename__ = "drivers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    user_id = Column(BigInteger, ForeignKey("users.id"), unique=True)
    vehicle_type = Column(Text)  # BIKE | CAR
    vehicle_no = Column(Text)
    active = Column(Boolean, default=True)

    # relationships
    user = relationship("User")
    deliveries = relationship("Delivery", back_populates="driver", lazy="selectin")


# ── Delivery ──────────────────────────────────────────────────────────────────

class Delivery(Base, RestaurantMixin):
    __tablename__ = "deliveries"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    order_id = Column(BigInteger, ForeignKey("orders.id"), nullable=False, unique=True, index=True)
    driver_id = Column(BigInteger, ForeignKey("drivers.id"), index=True)
    status = Column(Text, nullable=False)  # ASSIGNED | PICKED_UP | EN_ROUTE | DELIVERED | FAILED
    address_json = Column(JSONB, nullable=False)
    customer_phone = Column(Text)
    assigned_at = Column(DateTime(timezone=True))
    picked_up_at = Column(DateTime(timezone=True))
    delivered_at = Column(DateTime(timezone=True))

    # relationships
    order = relationship("Order", back_populates="delivery")
    driver = relationship("Driver", back_populates="deliveries")
    location_events = relationship("DeliveryLocationEvent", back_populates="delivery", lazy="selectin")


# ── Delivery Location Event ───────────────────────────────────────────────────

class DeliveryLocationEvent(Base, RestaurantMixin):
    __tablename__ = "delivery_location_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    delivery_id = Column(BigInteger, ForeignKey("deliveries.id"), nullable=False, index=True)
    lat = Column(Numeric(9, 6), nullable=False)
    lng = Column(Numeric(9, 6), nullable=False)
    accuracy_m = Column(Integer)
    recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # relationships
    delivery = relationship("Delivery", back_populates="location_events")
