"""
Core models: Restaurant, User, Customer, Channel, Session.
"""

from sqlalchemy import Column, BigInteger, String, Text, Boolean, DateTime, Integer, UniqueConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


# ── Restaurant ────────────────────────────────────────────────────────────────

class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    phone = Column(Text)
    address = Column(Text)
    timezone = Column(Text, server_default=text("'Asia/Kolkata'"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationships
    users = relationship("User", back_populates="restaurant", lazy="selectin")
    customers = relationship("Customer", back_populates="restaurant", lazy="selectin")
    menu_categories = relationship("MenuCategory", back_populates="restaurant", lazy="selectin")
    menu_items = relationship("MenuItem", back_populates="restaurant", lazy="selectin")


# ── User (staff and customer website accounts) ───────────────────────────────

class User(Base, RestaurantMixin):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True, server_default="1")
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="SET NULL"), unique=True, index=True)
    name = Column(Text, nullable=False)
    phone = Column(Text)
    email = Column(Text, unique=True)
    password_hash = Column(Text, nullable=False)
    role = Column(Text, nullable=False)  # CUSTOMER | ADMIN | MANAGER | CASHIER | WAITER | KITCHEN | DRIVER
    email_verified = Column(Boolean, nullable=False, server_default=text("false"))
    active = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationships
    restaurant = relationship("Restaurant", back_populates="users")
    oauth_identities = relationship(
        "OAuthIdentity",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    customer_profile = relationship("Customer", foreign_keys=[customer_id])


class OAuthIdentity(Base):
    """External sign-in identity linked to one local account."""

    __tablename__ = "oauth_identities"
    __table_args__ = (
        UniqueConstraint("provider", "provider_subject", name="uq_oauth_provider_subject"),
        UniqueConstraint("user_id", "provider", name="uq_oauth_user_provider"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(Text, nullable=False)
    provider_subject = Column(Text, nullable=False)
    email = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="oauth_identities")


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
    language_pref = Column(Text, server_default=text("'en'"))  # en | hi | te | code-mix
    marketing_opt_in = Column(Boolean, nullable=False, server_default=text("false"))
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
        UniqueConstraint(
            "restaurant_id",
            "type",
            "external_id",
            name="uq_channel_restaurant_type_external",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    type = Column(Text, nullable=False)  # TELEGRAM | WHATSAPP | WEB
    external_id = Column(Text, nullable=False)  # chat_id / wa_id
    customer_id = Column(BigInteger, ForeignKey("customers.id"), index=True)
    verified_at = Column(DateTime(timezone=True))
    verification_method = Column(Text)
    linked_at = Column(DateTime(timezone=True))
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

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
    ordering_platform = Column(Text, default="")  # NourishBot | Zomato | Swiggy
    history_json = Column(JSONB, default=[])  # last N turns
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    channel = relationship("Channel", back_populates="sessions")
    cart = relationship("Cart")
