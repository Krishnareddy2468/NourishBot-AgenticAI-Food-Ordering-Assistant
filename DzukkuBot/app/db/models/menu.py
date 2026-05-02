"""
Menu models: MenuCategory, MenuItem, MenuItemImage, ModifierGroup, Modifier, MenuItemModifierGroup.
"""

from sqlalchemy import Column, BigInteger, String, Text, Boolean, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship

from app.db.base import Base, RestaurantMixin


# ── Menu Category ─────────────────────────────────────────────────────────────

class MenuCategory(Base, RestaurantMixin):
    __tablename__ = "menu_categories"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True, server_default="1")
    name = Column(Text, nullable=False)
    sort_order = Column(Integer, default=0)

    # relationships
    restaurant = relationship("Restaurant", back_populates="menu_categories")
    items = relationship("MenuItem", back_populates="category", lazy="selectin")


# ── Menu Item ─────────────────────────────────────────────────────────────────

class MenuItem(Base, RestaurantMixin):
    __tablename__ = "menu_items"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, ForeignKey("restaurants.id"), nullable=False, index=True, server_default="1")
    category_id = Column(BigInteger, ForeignKey("menu_categories.id"), index=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    type = Column(Text)  # VEG | NON_VEG | EGG | VEGAN
    price_cents = Column(Integer, nullable=False)  # smallest currency unit
    special_price_cents = Column(Integer)
    available = Column(Boolean, default=True)
    stock_qty = Column(Integer)  # null = unlimited
    prep_time_sec = Column(Integer, default=900)
    tags = Column(ARRAY(Text))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # relationships
    restaurant = relationship("Restaurant", back_populates="menu_items")
    category = relationship("MenuCategory", back_populates="items")
    images = relationship("MenuItemImage", back_populates="item", lazy="selectin")
    modifier_groups = relationship("MenuItemModifierGroup", back_populates="item", lazy="selectin")


# ── Menu Item Image ───────────────────────────────────────────────────────────

class MenuItemImage(Base, RestaurantMixin):
    __tablename__ = "menu_item_images"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    item_id = Column(BigInteger, ForeignKey("menu_items.id"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    alt_text = Column(Text)
    sort_order = Column(Integer, default=0)
    checksum = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # relationships
    item = relationship("MenuItem", back_populates="images")


# ── Modifier Group ────────────────────────────────────────────────────────────

class ModifierGroup(Base, RestaurantMixin):
    __tablename__ = "modifier_groups"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    name = Column(Text, nullable=False)
    min_select = Column(Integer, default=0)
    max_select = Column(Integer, default=1)

    # relationships
    modifiers = relationship("Modifier", back_populates="group", lazy="selectin")


# ── Modifier ──────────────────────────────────────────────────────────────────

class Modifier(Base, RestaurantMixin):
    __tablename__ = "modifiers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    group_id = Column(BigInteger, ForeignKey("modifier_groups.id"), nullable=False, index=True)
    name = Column(Text, nullable=False)
    price_cents = Column(Integer, default=0)
    available = Column(Boolean, default=True)

    # relationships
    group = relationship("ModifierGroup", back_populates="modifiers")


# ── Menu Item ↔ Modifier Group (association) ──────────────────────────────────

class MenuItemModifierGroup(Base, RestaurantMixin):
    __tablename__ = "menu_item_modifier_groups"
    __table_args__ = (
        UniqueConstraint("item_id", "group_id", name="uq_menu_item_modifier_group"),
    )

    restaurant_id = Column(BigInteger, nullable=False, index=True, server_default="1")
    item_id = Column(BigInteger, ForeignKey("menu_items.id"), nullable=False, primary_key=True)
    group_id = Column(BigInteger, ForeignKey("modifier_groups.id"), nullable=False, primary_key=True)

    # relationships
    item = relationship("MenuItem", back_populates="modifier_groups")
    group = relationship("ModifierGroup")
