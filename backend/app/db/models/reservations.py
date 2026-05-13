"""
Reservation model: Reservation.
"""

from sqlalchemy import Column, BigInteger, Text, Integer, Date, Time, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


class Reservation(Base, RestaurantMixin):
    __tablename__ = "reservations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    reservation_ref = Column(Text, unique=True, nullable=False)  # RSV-XXXXXX
    customer_id = Column(BigInteger, ForeignKey("customers.id"), index=True)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    guests = Column(Integer, nullable=False)
    special_request = Column(Text)
    status = Column(Text, nullable=False, default="CREATED")  # CREATED|CONFIRMED|CANCELLED|NO_SHOW
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationships
    customer = relationship("Customer")
