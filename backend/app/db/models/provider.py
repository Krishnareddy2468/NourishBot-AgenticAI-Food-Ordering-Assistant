"""Versioned provider capabilities and append-only response receipts."""

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, Text, UniqueConstraint, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text

from app.db.base import Base, RestaurantMixin


class ProviderCapabilityVersion(Base):
    """Immutable capability assessment for one provider integration version."""

    __tablename__ = "provider_capability_versions"
    __table_args__ = (
        UniqueConstraint("provider", "version", name="uq_provider_capability_version"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    provider = Column(Text, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    state = Column(Text, nullable=False, index=True)  # UNVERIFIED|VERIFIED|DEGRADED|DISABLED
    mode = Column(Text, nullable=False, index=True)  # DISCOVERY|HANDOFF|TRANSACTIONAL
    capabilities_json = Column(JSONB, nullable=False, default=dict)
    tool_schema_hash = Column(Text, index=True)
    tool_schema_json = Column(JSONB, nullable=False, default=dict)
    adapter_version = Column(Text, nullable=False, server_default=text("'mcp-v1'"))
    verified_at = Column(DateTime(timezone=True))
    # Verification is part of this immutable record. Restrict deletion of the
    # verifier instead of silently mutating the historical capability version.
    verified_by_user_id = Column(BigInteger, ForeignKey("users.id", ondelete="RESTRICT"))
    verification_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ProviderReceipt(Base, RestaurantMixin):
    """Append-only redacted receipt for every provider tool response."""

    __tablename__ = "provider_receipts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    provider_attempt_id = Column(
        Text,
        ForeignKey("provider_attempts.id", ondelete="RESTRICT"),
        index=True,
    )
    capability_version_id = Column(
        BigInteger,
        ForeignKey("provider_capability_versions.id", ondelete="RESTRICT"),
        index=True,
    )
    provider = Column(Text, nullable=False, index=True)
    mode = Column(Text, nullable=False)
    operation = Column(Text, nullable=False, index=True)
    tool_name = Column(Text)
    request_fingerprint = Column(Text, nullable=False)
    request_redacted_json = Column(JSONB, nullable=False, default=dict)
    response_redacted_json = Column(JSONB, nullable=False, default=dict)
    response_hash = Column(Text, nullable=False, index=True)
    outcome = Column(Text, nullable=False, index=True)
    provider_request_id = Column(Text)
    provider_order_id = Column(Text)
    tool_schema_hash = Column(Text)
    occurred_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)


def _reject_provider_audit_mutation(_mapper, _connection, target) -> None:
    raise ValueError(f"{target.__class__.__name__} records are append-only")


for _model in (ProviderCapabilityVersion, ProviderReceipt):
    event.listen(_model, "before_update", _reject_provider_audit_mutation)
    event.listen(_model, "before_delete", _reject_provider_audit_mutation)
