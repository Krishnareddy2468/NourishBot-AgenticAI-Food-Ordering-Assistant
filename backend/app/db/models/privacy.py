"""Customer identity, consent, privacy requests, and memory audit models."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from app.db.base import Base, RestaurantMixin


class IdentityLinkChallenge(Base, RestaurantMixin):
    """One-time, hashed code used to link a second verified channel."""

    __tablename__ = "identity_link_challenges"

    id = Column(Text, primary_key=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    source_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    target_channel_type = Column(Text, nullable=False)
    token_hash = Column(Text, nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    used_at = Column(DateTime(timezone=True))
    target_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class CustomerConsent(Base, RestaurantMixin):
    """Append-only consent decision. The newest row is the current decision."""

    __tablename__ = "customer_consents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    purpose = Column(Text, nullable=False, index=True)
    granted = Column(Boolean, nullable=False)
    policy_version = Column(Text, nullable=False)
    source_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)


class PrivacyRequest(Base, RestaurantMixin):
    """Audit record for user-requested privacy operations without retaining exports."""

    __tablename__ = "privacy_requests"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    request_type = Column(Text, nullable=False)  # VIEW | EXPORT | DISABLE | DELETE
    source_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"))
    status = Column(Text, nullable=False, server_default="COMPLETED")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MemoryLearningEvent(Base, RestaurantMixin):
    """Idempotency/audit record for each accepted personalization signal."""

    __tablename__ = "memory_learning_events"
    __table_args__ = (
        UniqueConstraint("order_id", "signal_type", name="uq_memory_learning_order_signal"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True)
    signal_type = Column(Text, nullable=False)  # FULFILLMENT | EXPLICIT_FEEDBACK
    rating = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
