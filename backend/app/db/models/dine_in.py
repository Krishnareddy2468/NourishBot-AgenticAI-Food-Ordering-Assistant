"""
Dine-in models: DiningTable, TableSession, TableSessionOrder.
"""

from sqlalchemy import Column, BigInteger, Text, Integer, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func, text
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


# ── Dining Table ──────────────────────────────────────────────────────────────

class DiningTable(Base, RestaurantMixin):
    __tablename__ = "dining_tables"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    name = Column(Text, nullable=False)
    capacity = Column(Integer, nullable=False)
    active = Column(Boolean, nullable=False, server_default=text("true"))

    # relationships
    sessions = relationship("TableSession", back_populates="table", lazy="selectin")


# ── Table Session ─────────────────────────────────────────────────────────────

class TableSession(Base, RestaurantMixin):
    __tablename__ = "table_sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    table_id = Column(BigInteger, ForeignKey("dining_tables.id"), nullable=False, index=True)
    waiter_user_id = Column(BigInteger, ForeignKey("users.id"), index=True)
    guests = Column(Integer, nullable=False)
    status = Column(Text, nullable=False, default="OPEN")  # OPEN | CLOSED | CANCELLED
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    notes = Column(Text)

    # relationships
    table = relationship("DiningTable", back_populates="sessions")
    waiter = relationship("User")
    orders = relationship("TableSessionOrder", back_populates="session", lazy="selectin")


# ── Table Session ↔ Order (association) ───────────────────────────────────────

class TableSessionOrder(Base, RestaurantMixin):
    __tablename__ = "table_session_orders"
    __table_args__ = (
        UniqueConstraint("table_session_id", "order_id", name="uq_table_session_order"),
    )

    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    table_session_id = Column(BigInteger, ForeignKey("table_sessions.id"), nullable=False, primary_key=True)
    order_id = Column(BigInteger, ForeignKey("orders.id"), nullable=False, primary_key=True)

    # relationships
    session = relationship("TableSession", back_populates="orders")
    order = relationship("Order")
