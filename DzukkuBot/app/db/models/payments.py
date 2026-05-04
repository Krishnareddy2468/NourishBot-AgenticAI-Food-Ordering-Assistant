"""
Payment models: Payment (Razorpay).
"""

from sqlalchemy import Column, BigInteger, Text, Integer, ForeignKey
from sqlalchemy.sql import func, text
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


class Payment(Base, RestaurantMixin):
    __tablename__ = "payments"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    order_id = Column(BigInteger, ForeignKey("orders.id"), nullable=False, index=True)
    provider = Column(Text, nullable=False, server_default=text("'RAZORPAY'"))
    status = Column(Text, nullable=False, server_default=text("'CREATED'"))  # CREATED|AUTHORIZED|CAPTURED|FAILED|REFUNDED
    amount_cents = Column(Integer, nullable=False)
    currency = Column(Text, nullable=False, server_default=text("'INR'"))
    provider_order_id = Column(Text)      # Razorpay order ID
    provider_payment_id = Column(Text)    # Razorpay payment ID
    provider_signature = Column(Text)     # Razorpay signature
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    order = relationship("Order", back_populates="payment")
