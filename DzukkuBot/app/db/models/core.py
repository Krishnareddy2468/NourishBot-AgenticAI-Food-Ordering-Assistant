"""
Core models: Restaurant, User, Customer, Channel, Session.
"""

from sqlalchemy import Column, BigInteger, String, Text, Boolean, DateTime, Integer, UniqueConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


# ── Restaurant ────────────────────────────────────────────────────────────────

class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    phone = Column(Text)
    address = Column(Text)
    timezone = Column(Text, default="Asia/Kolkata")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationships
    users = relationship("User", back_populates="restaurant", lazy="selectin")
    customers = relationship("Customer", back_populates="restaurant", lazy="selectin")
    menu_categories = relationship("MenuCategory", back_populates="restaurant", lazy="selectin")
    menu_items = relationship("MenuItem", back_populates="restaurant", lazy="selectin")


# ── User (staff) ──────────────────────────────────────────────────────────────

class User(Base, RestaurantMixin):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True, server_default="1")
    name = Column(Text, nullable=False)
    phone = Column(Text)
    email = Column(Text, unique=True)
    password_hash = Column(Text, nullable=False)
    role = Column(Text, nullable=False)  # ADMIN | MANAGER | CASHIER | WAITER | KITCHEN | DRIVER
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationships
    restaurant = relationship("Restaurant", back_populates="users")


# ── Customer ──────────────────────────────────────────────────────────────────

class Customer(Base, RestaurantMixin):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("restaurant_id", "phone", name="uq_customer_restaurant_phone"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True, server_default="1")
    name = Column(Text)
    phone = Column(Text, nullable=False)
    email = Column(Text)
    language_pref = Column(Text, default="en")  # en | hi | te | code-mix
    marketing_opt_in = Column(Boolean, default=False)
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    restaurant = relationship("Restaurant", back_populates="customers")
    channels = relationship("Channel", back_populates="customer", lazy="selectin")
    carts = relationship("Cart", back_populates="customer", lazy="selectin")
    orders = relationship("Order", back_populates="customer", lazy="selectin")


# ── Channel ───────────────────────────────────────────────────────────────────

class Channel(Base, RestaurantMixin):
    __tablename__ = "channels"
    __table_args__ = (
        UniqueConstraint("type", "external_id", name="uq_channel_type_external"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    type = Column(Text, nullable=False)  # TELEGRAM | WHATSAPP | WEB
    external_id = Column(Text, nullable=False)  # chat_id / wa_id
    customer_id = Column(BigInteger, ForeignKey("customers.id"), index=True)

    # relationships
    customer = relationship("Customer", back_populates="channels")
    sessions = relationship("Session", back_populates="channel", lazy="selectin")


# ── Session ───────────────────────────────────────────────────────────────────

class Session(Base, RestaurantMixin):
    __tablename__ = "sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    channel_id = Column(BigInteger, ForeignKey("channels.id"), nullable=False, index=True)
    state = Column(Text, nullable=False, default="IDLE")
    cart_id = Column(BigInteger, ForeignKey("carts.id"), index=True)
    history_json = Column(JSONB, default=[])  # last N turns
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    channel = relationship("Channel", back_populates="sessions")
    cart = relationship("Cart")
