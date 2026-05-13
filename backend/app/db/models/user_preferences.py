"""
User preferences model — persistent taste vector + behavioral memory.

Stores per-user food preferences, health goals, craving cycles,
and a pgvector embedding for semantic similarity search.

Used by the Memory Agent (app.agent.memory_agent) and injected
into the Planner prompt for personalized recommendations.
"""

from sqlalchemy import (
    Column, BigInteger, Text, Float, Integer, ForeignKey, DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

from app.db.base import Base


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(
        BigInteger,
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # ── Taste profile ──────────────────────────────────────────────────────
    spice_level = Column(
        Float, nullable=False, server_default="0.5",
        comment="0.0 = no spice → 1.0 = extra hot",
    )
    cuisine_weights = Column(
        JSONB, nullable=False, server_default="{}",
        comment='{"biryani": 0.8, "pizza": 0.3, "idli": 0.6}',
    )
    price_band = Column(
        Text, server_default="mid",
        comment="budget | mid | premium",
    )
    order_timing = Column(
        JSONB, nullable=False, server_default="{}",
        comment='{"breakfast": 0.2, "lunch": 0.7, "dinner": 0.9}',
    )
    platform_preference = Column(
        Text, server_default="direct",
        comment="direct | zomato | swiggy",
    )

    # ── Dietary & health ───────────────────────────────────────────────────
    dietary_flags = Column(
        ARRAY(Text), server_default="{}",
        comment='e.g. {"vegetarian", "no-pork", "less-oil"}',
    )
    health_goals = Column(
        ARRAY(Text), server_default="{}",
        comment='e.g. {"high-protein", "weight-loss"}',
    )
    allergies = Column(
        ARRAY(Text), server_default="{}",
        comment='e.g. {"peanuts", "dairy"}',
    )
    health_onboarding_done = Column(
        Integer, nullable=False, server_default="0",
        comment="0 = not prompted, 1 = prompted, 2 = completed",
    )

    # ── Craving cycles ─────────────────────────────────────────────────────
    craving_cycles = Column(
        JSONB, nullable=False, server_default="{}",
        comment='{"biryani": {"last_ordered": "2026-05-01T12:00:00", "avg_interval_days": 5}}',
    )
    weather_behavior = Column(
        JSONB, server_default="{}",
        comment='{"rainy": "hot_soup", "hot": "cold_drinks"}',
    )

    # ── Vector embedding (pgvector) — 384-dim text-embedding ───────────────
    taste_embedding = (
        Column(Vector(384), nullable=True)
        if HAS_PGVECTOR
        else Column(Text, nullable=True)
    )

    # ── Timestamps ─────────────────────────────────────────────────────────
    total_orders = Column(Integer, nullable=False, server_default="0")
    total_spent_cents = Column(BigInteger, nullable=False, server_default="0")
    last_order_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # relationships
    customer = relationship("Customer")
