"""
Invoice model: Invoice.
"""

from sqlalchemy import Column, BigInteger, Text, Integer, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


class Invoice(Base, RestaurantMixin):
    __tablename__ = "invoices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    invoice_no = Column(Text, unique=True, nullable=False)
    entity_type = Column(Text, nullable=False)  # ORDER | TABLE_SESSION
    entity_id = Column(BigInteger, nullable=False)
    subtotal_cents = Column(Integer, nullable=False)
    tax_cents = Column(Integer, default=0)
    total_cents = Column(Integer, nullable=False)
    pdf_url = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
