"""Immutable customer spending entries used by budgets and analytics."""

from sqlalchemy import BigInteger, CheckConstraint, Column, DateTime, ForeignKey, Integer, Text, UniqueConstraint, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text

from app.db.base import Base, RestaurantMixin


class SpendingLedgerEntry(Base, RestaurantMixin):
    __tablename__ = "spending_ledger_entries"
    __table_args__ = (
        UniqueConstraint("order_id", "entry_type", name="uq_spending_ledger_order_type"),
        CheckConstraint("amount_cents <> 0", name="ck_spending_ledger_nonzero_amount"),
        CheckConstraint(
            "entry_type IN ('COMPLETED_SPEND','REFUND_CREDIT')",
            name="ck_spending_ledger_entry_type",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True)
    customer_id = Column(BigInteger, ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False, index=True)
    order_id = Column(BigInteger, ForeignKey("orders.id", ondelete="RESTRICT"), nullable=False, index=True)
    entry_type = Column(Text, nullable=False, index=True)  # COMPLETED_SPEND | REFUND_CREDIT
    amount_cents = Column(Integer, nullable=False)  # spend positive, refunds negative
    currency = Column(Text, nullable=False, server_default=text("'INR'"))
    metadata_json = Column(JSONB, nullable=False, default=dict)
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


def _reject_spending_mutation(_mapper, _connection, target) -> None:
    raise ValueError(f"{target.__class__.__name__} records are append-only")


event.listen(SpendingLedgerEntry, "before_update", _reject_spending_mutation)
event.listen(SpendingLedgerEntry, "before_delete", _reject_spending_mutation)
