"""Safety records for confirmation-bound ordering and provider reconciliation."""

from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base, RestaurantMixin


class ConfirmationChallenge(Base, RestaurantMixin):
    """Immutable order snapshot that a user must explicitly confirm once."""

    __tablename__ = "confirmation_challenges"

    id = Column(Text, primary_key=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True)
    session_id = Column(BigInteger, ForeignKey("sessions.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id"), nullable=True, index=True)
    snapshot_hash = Column(Text, nullable=False)
    snapshot_json = Column(JSONB, nullable=False)
    idempotency_key = Column(Text, nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    used_at = Column(DateTime(timezone=True), nullable=True)
    invalidated_at = Column(DateTime(timezone=True), nullable=True)
    created_by_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True)
    created_by_channel_type = Column(Text, nullable=True)
    confirmed_by_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True)
    invalidated_by_channel_id = Column(BigInteger, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True)
    confirmation_actor = Column(Text, nullable=True)
    invalidation_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ProviderAttempt(Base, RestaurantMixin):
    """Durable record for an external provider action and its final certainty."""

    __tablename__ = "provider_attempts"
    __table_args__ = (
        UniqueConstraint("provider", "idempotency_key", name="uq_provider_attempt_idempotency"),
    )

    id = Column(Text, primary_key=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True)
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True)
    meal_schedule_id = Column(BigInteger, ForeignKey("meal_schedules.id", ondelete="SET NULL"), nullable=True, index=True)
    provider = Column(Text, nullable=False, index=True)
    channel_type = Column(Text, nullable=False)
    external_chat_id = Column(Text, nullable=False, index=True)
    operation = Column(Text, nullable=False)
    mode = Column(Text, nullable=False, server_default="DISCOVERY")
    capability_version_id = Column(
        BigInteger,
        ForeignKey("provider_capability_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    tool_schema_hash = Column(Text, nullable=True)
    idempotency_key = Column(Text, nullable=False)
    request_fingerprint = Column(Text, nullable=False)
    status = Column(Text, nullable=False, index=True)  # SUBMITTING|CONFIRMED|FAILED|UNKNOWN|CANCELLED
    attempt_count = Column(Integer, nullable=False, default=1, server_default="1")
    provider_request_id = Column(Text, nullable=True)
    provider_order_id = Column(Text, nullable=True)
    response_excerpt = Column(Text, nullable=True)
    receipt_json = Column(JSONB, nullable=True)
    response_hash = Column(Text, nullable=True)
    last_error = Column(Text, nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    reconciled_at = Column(DateTime(timezone=True), nullable=True)
    reconciliation_required = Column(Boolean, nullable=False, server_default="false")
    reconciliation_count = Column(Integer, nullable=False, server_default="0")
    last_reconciliation_at = Column(DateTime(timezone=True), nullable=True)
    next_action = Column(Text, nullable=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
