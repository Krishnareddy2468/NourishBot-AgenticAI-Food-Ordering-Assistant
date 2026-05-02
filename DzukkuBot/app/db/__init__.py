"""
Dzukku vNext — SQLAlchemy ORM models + PostgreSQL session management.

All core tables include `restaurant_id` (multi-tenant ready, default 1).
"""

from app.db.base import Base
from app.db.session import get_async_session, async_engine, AsyncSessionLocal
from app.db.models import (
    Restaurant,
    User,
    Customer,
    Channel,
    Session,
    MenuCategory,
    MenuItem,
    MenuItemImage,
    ModifierGroup,
    Modifier,
    MenuItemModifierGroup,
    Cart,
    CartItem,
    Order,
    OrderItem,
    Driver,
    Delivery,
    DeliveryLocationEvent,
    Payment,
    DiningTable,
    TableSession,
    TableSessionOrder,
    Reservation,
    Invoice,
    OutboxEvent,
)

__all__ = [
    "Base",
    "get_async_session",
    "async_engine",
    "AsyncSessionLocal",
    "Restaurant",
    "User",
    "Customer",
    "Channel",
    "Session",
    "MenuCategory",
    "MenuItem",
    "MenuItemImage",
    "ModifierGroup",
    "Modifier",
    "MenuItemModifierGroup",
    "Cart",
    "CartItem",
    "Order",
    "OrderItem",
    "Driver",
    "Delivery",
    "DeliveryLocationEvent",
    "Payment",
    "DiningTable",
    "TableSession",
    "TableSessionOrder",
    "Reservation",
    "Invoice",
    "OutboxEvent",
]
