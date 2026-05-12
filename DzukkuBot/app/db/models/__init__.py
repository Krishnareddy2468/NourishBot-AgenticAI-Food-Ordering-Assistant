"""
Dzukku vNext — All SQLAlchemy ORM models.

Organised by domain:
  - Core: Restaurant, User, Customer, Channel, Session
  - Menu: MenuCategory, MenuItem, MenuItemImage, ModifierGroup, Modifier, MenuItemModifierGroup
  - Cart+Orders: Cart, CartItem, Order, OrderItem
  - Delivery: Driver, Delivery, DeliveryLocationEvent
  - Payments: Payment
  - Dine-in: DiningTable, TableSession, TableSessionOrder
  - Reservations: Reservation
  - Invoices: Invoice
  - Events: OutboxEvent
"""

from app.db.models.core import Restaurant, User, Customer, Channel, Session
from app.db.models.menu import (
    MenuCategory,
    MenuItem,
    MenuItemImage,
    ModifierGroup,
    Modifier,
    MenuItemModifierGroup,
)
from app.db.models.cart_orders import Cart, CartItem, Order, OrderItem
from app.db.models.delivery import Driver, Delivery, DeliveryLocationEvent
from app.db.models.payments import Payment
from app.db.models.dine_in import DiningTable, TableSession, TableSessionOrder
from app.db.models.reservations import Reservation
from app.db.models.invoices import Invoice
from app.db.models.outbox import OutboxEvent
from app.db.models.user_preferences import UserPreferences

__all__ = [
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
    "UserPreferences",
]
