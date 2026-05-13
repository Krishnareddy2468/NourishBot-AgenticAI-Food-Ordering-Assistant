"""
Cart + Orders models: Cart, CartItem, Order, OrderItem.
"""

from sqlalchemy import Column, BigInteger, Text, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func, text
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


# ── Cart ──────────────────────────────────────────────────────────────────────

class Cart(Base, RestaurantMixin):
    __tablename__ = "carts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    customer_id = Column(BigInteger, ForeignKey("customers.id"), index=True)
    status = Column(Text, nullable=False, server_default=text("'OPEN'"))  # OPEN | CONVERTED | ABANDONED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    customer = relationship("Customer", back_populates="carts")
    items = relationship("CartItem", back_populates="cart", lazy="selectin")


# ── Cart Item ─────────────────────────────────────────────────────────────────

class CartItem(Base, RestaurantMixin):
    __tablename__ = "cart_items"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    cart_id = Column(BigInteger, ForeignKey("carts.id"), nullable=False, index=True)
    item_id = Column(BigInteger, ForeignKey("menu_items.id"), nullable=False, index=True)
    qty = Column(Integer, nullable=False)
    unit_price_cents = Column(Integer, nullable=False)
    modifiers_json = Column(JSONB)

    # relationships
    cart = relationship("Cart", back_populates="items")
    item = relationship("MenuItem")


# ── Order ─────────────────────────────────────────────────────────────────────

class Order(Base, RestaurantMixin):
    __tablename__ = "orders"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    order_ref = Column(Text, nullable=False, unique=True)  # DZK-XXXXXX
    customer_id = Column(BigInteger, ForeignKey("customers.id"), index=True)
    channel_id = Column(BigInteger, ForeignKey("channels.id"), index=True)
    order_type = Column(Text, nullable=False)  # DELIVERY | PICKUP | DINE_IN
    status = Column(Text, nullable=False, default="CREATED")  # CREATED|ACCEPTED|PREPARING|READY|OUT_FOR_DELIVERY|DELIVERED|CANCELLED
    subtotal_cents = Column(Integer, nullable=False)
    tax_cents = Column(Integer, default=0)
    packing_cents = Column(Integer, default=0)
    discount_cents = Column(Integer, default=0)
    total_cents = Column(Integer, nullable=False)
    eta_ts = Column(DateTime(timezone=True))
    idempotency_key = Column(Text, unique=True)
    notes = Column(Text)
    rating = Column(Integer, nullable=True, comment="1–5 star rating after delivery")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    customer = relationship("Customer", back_populates="orders", lazy="selectin")
    channel = relationship("Channel")
    items = relationship("OrderItem", back_populates="order", lazy="selectin")
    delivery = relationship("Delivery", back_populates="order", uselist=False, lazy="selectin")
    payment = relationship("Payment", back_populates="order", uselist=False, lazy="selectin")


# ── Order Item ────────────────────────────────────────────────────────────────

class OrderItem(Base, RestaurantMixin):
    __tablename__ = "order_items"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    order_id = Column(BigInteger, ForeignKey("orders.id"), nullable=False, index=True)
    item_id = Column(BigInteger, ForeignKey("menu_items.id"), nullable=False, index=True)
    item_name_snapshot = Column(Text, nullable=False)
    qty = Column(Integer, nullable=False)
    unit_price_cents = Column(Integer, nullable=False)
    modifiers_json = Column(JSONB)
    status = Column(Text, nullable=False, default="PENDING", server_default=text("'PENDING'"))  # PENDING | IN_PROGRESS | DONE | CANCELLED

    # relationships
    order = relationship("Order", back_populates="items")
    item = relationship("MenuItem")
