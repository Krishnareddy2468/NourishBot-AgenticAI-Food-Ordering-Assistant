"""
Outbox event model: OutboxEvent.
"""

from sqlalchemy import Column, BigInteger, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base, RestaurantMixin


class OutboxEvent(Base, RestaurantMixin):
    __tablename__ = "outbox_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    event_type = Column(Text, nullable=False)
    payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True))
    dedupe_key = Column(Text, unique=True)
    attempt_count = Column(Integer, nullable=False, server_default="0")
    max_attempts = Column(Integer, nullable=False, server_default="8")
    next_retry_at = Column(DateTime(timezone=True), index=True)
    lease_owner = Column(Text)
    lease_expires_at = Column(DateTime(timezone=True), index=True)
    last_error = Column(Text)
    dead_lettered_at = Column(DateTime(timezone=True), index=True)
