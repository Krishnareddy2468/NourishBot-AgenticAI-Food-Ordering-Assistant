"""
SQLAlchemy declarative base for all Dzukku models.
"""

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, BigInteger
from sqlalchemy.sql import func
from sqlalchemy import DateTime


class Base(DeclarativeBase):
    pass


class RestaurantMixin:
    """Mixin that adds restaurant_id to models for multi-tenant readiness."""
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
