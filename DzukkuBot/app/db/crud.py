"""
Async CRUD helpers — PostgreSQL equivalents for the old SQLite database.py.

All functions use the async SQLAlchemy session from app.db.session.
These replace the old sync SQLite helpers used by telegram.py and the agents.
"""

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models import (
    Session as SessionModel,
    Customer,
    Channel,
    Order,
    OrderItem,
    MenuCategory,
    MenuItem,
    Reservation,
    Cart,
    CartItem,
)

logger = logging.getLogger(__name__)


# ── Session helpers ───────────────────────────────────────────────────────────

async def _get_channel(chat_id: int) -> Optional[Channel]:
    """Find or create a channel for the given Telegram chat_id."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Channel).where(
                Channel.type == "TELEGRAM",
                Channel.external_id == str(chat_id),
            )
        )
        channel = result.scalar_one_or_none()
        if not channel:
            channel = Channel(
                restaurant_id=1,
                type="TELEGRAM",
                external_id=str(chat_id),
            )
            session.add(channel)
            await session.commit()
            await session.refresh(channel)
        return channel


async def get_session(chat_id: int) -> dict:
    """Return session state as a dict (same shape as old SQLite version)."""
    async with AsyncSessionLocal() as session:
        # Find channel
        ch_result = await session.execute(
            select(Channel).where(
                Channel.type == "TELEGRAM",
                Channel.external_id == str(chat_id),
            )
        )
        channel = ch_result.scalar_one_or_none()
        if not channel:
            return {
                "chat_id": chat_id,
                "state": "new",
                "user_name": "",
                "cart": [],
                "customer_name": "",
                "customer_phone": "",
                "history": [],
                "ordering_platform": "",
            }

        # Find session
        s_result = await session.execute(
            select(SessionModel).where(SessionModel.channel_id == channel.id)
        )
        sess = s_result.scalar_one_or_none()
        if not sess:
            return {
                "chat_id": chat_id,
                "state": "new",
                "user_name": "",
                "cart": [],
                "customer_name": "",
                "customer_phone": "",
                "history": [],
                "ordering_platform": "",
            }

        # Reconstruct cart from Cart + CartItems if linked
        cart_data = []
        if sess.cart_id:
            cart_result = await session.execute(
                select(Cart).where(Cart.id == sess.cart_id)
            )
            cart_obj = cart_result.scalar_one_or_none()
            if cart_obj:
                items_result = await session.execute(
                    select(CartItem).where(CartItem.cart_id == cart_obj.id)
                )
                for ci in items_result.scalars().all():
                    item_result = await session.execute(
                        select(MenuItem).where(MenuItem.id == ci.item_id)
                    )
                    mi = item_result.scalar_one_or_none()
                    cart_data.append({
                        "item_name": mi.name if mi else "Unknown",
                        "qty": ci.qty,
                        "price": ci.unit_price_cents / 100,
                        "type": mi.type if mi else "",
                    })

        history = sess.history_json if isinstance(sess.history_json, list) else []
        return {
            "chat_id": chat_id,
            "state": sess.state,
            "user_name": "",
            "cart": cart_data,
            "customer_name": "",
            "customer_phone": "",
            "history": history,
            "ordering_platform": "",
        }


async def save_session(chat_id: int, updates: dict) -> None:
    """Update session fields from a dict of updates."""
    async with AsyncSessionLocal() as session:
        # Find or create channel
        ch_result = await session.execute(
            select(Channel).where(
                Channel.type == "TELEGRAM",
                Channel.external_id == str(chat_id),
            )
        )
        channel = ch_result.scalar_one_or_none()
        if not channel:
            channel = Channel(
                restaurant_id=1,
                type="TELEGRAM",
                external_id=str(chat_id),
            )
            session.add(channel)
            await session.flush()

        # Find or create session
        s_result = await session.execute(
            select(SessionModel).where(SessionModel.channel_id == channel.id)
        )
        sess = s_result.scalar_one_or_none()
        if not sess:
            sess = SessionModel(
                restaurant_id=1,
                channel_id=channel.id,
                state="new",
                history_json=[],
            )
            session.add(sess)
            await session.flush()

        # Apply updates
        if "state" in updates:
            sess.state = updates["state"]
        if "user_name" in updates:
            pass  # user_name not on session model in vNext; kept for compat
        if "ordering_platform" in updates:
            # Store ordering_platform in history_json metadata for now
            history = sess.history_json if isinstance(sess.history_json, list) else []
            # Add a metadata entry
            pass  # ordering_platform managed at channel/session level differently
        if "history" in updates:
            sess.history_json = updates["history"]
        if "cart" in updates:
            # Cart is handled separately via Cart/CartItem models
            pass

        await session.commit()


async def reset_session(chat_id: int, user_name: str = "") -> None:
    """Reset session to fresh state."""
    await save_session(chat_id, {
        "state": "greeting",
        "history": [],
    })


# ── Menu helpers ──────────────────────────────────────────────────────────────

async def get_menu_items(
    filter_type: str = "All",
    filter_category: str = "",
) -> list[dict]:
    """Return menu items as list of dicts (compatible with old interface)."""
    async with AsyncSessionLocal() as session:
        query = select(MenuItem).where(MenuItem.available == True)
        if filter_type and filter_type != "All":
            query = query.where(MenuItem.type == filter_type)
        if filter_category:
            query = query.join(MenuCategory).where(
                MenuCategory.name == filter_category
            )
        query = query.order_by(MenuItem.name)
        result = await session.execute(query)
        items = result.scalars().all()

        return [
            {
                "item_no": str(item.id),
                "item_name": item.name,
                "description": item.description or "",
                "type": item.type or "",
                "category": "",
                "price": item.price_cents / 100,
                "available": "Yes" if item.available else "No",
                "special_price": (item.special_price_cents / 100) if item.special_price_cents else None,
            }
            for item in items
        ]


async def get_menu_text() -> str:
    """Return formatted menu text for prompt injection."""
    items = await get_menu_items()
    if not items:
        return "Menu not loaded yet."

    lines = ["DZUKKU RESTAURANT MENU:"]
    current_cat = None
    for item in items:
        cat = item.get("category", "Main Course") or "Main Course"
        if cat != current_cat:
            current_cat = cat
            lines.append(f"\n── {current_cat} ──")
        status = "✅" if item["available"] == "Yes" else "❌ Unavailable"
        price_str = f"₹{int(item['price'])}" if item["price"] > 0 else "Ask for price"
        sp = item.get("special_price")
        if sp:
            price_str = f"~~₹{int(item['price'])}~~ ₹{int(sp)} ⭐"
        lines.append(f"  • {item['item_name']} ({item['type']}) — {price_str} {status}")
        if item.get("description"):
            lines.append(f"    {item['description']}")
    return "\n".join(lines)


# ── Order helpers ──────────────────────────────────────────────────────────────

async def save_order(
    customer_name: str,
    customer_phone: str,
    items: list[dict],
    total_price: float,
    platform: str = "Telegram",
    idempotency_key: Optional[str] = None,
) -> str:
    """Create an order + order items. Returns order_ref (DZK-XXXXXX)."""
    order_ref = f"DZK-{uuid.uuid4().hex[:6].upper()}"
    idem_key = idempotency_key or f"{customer_phone}-{order_ref}"
    total_cents = int(total_price * 100)

    async with AsyncSessionLocal() as session:
        # Find or create customer
        cust_result = await session.execute(
            select(Customer).where(
                Customer.restaurant_id == 1,
                Customer.phone == customer_phone,
            )
        )
        customer = cust_result.scalar_one_or_none()
        if not customer:
            customer = Customer(
                restaurant_id=1,
                name=customer_name,
                phone=customer_phone,
            )
            session.add(customer)
            await session.flush()

        order = Order(
            restaurant_id=1,
            order_ref=order_ref,
            customer_id=customer.id,
            order_type="DELIVERY",
            status="CREATED",
            subtotal_cents=total_cents,
            total_cents=total_cents,
            idempotency_key=idem_key,
        )
        session.add(order)
        await session.flush()

        for item in items:
            name = item.get("item_name", item.get("item", ""))
            qty = item.get("qty", 1)
            unit_price = item.get("price", item.get("unitPrice", 0))
            unit_cents = int(float(unit_price) * 100)

            # Find menu item by name
            mi_result = await session.execute(
                select(MenuItem).where(MenuItem.name == name)
            )
            mi = mi_result.scalar_one_or_none()

            order_item = OrderItem(
                restaurant_id=1,
                order_id=order.id,
                item_id=mi.id if mi else 0,
                item_name_snapshot=name,
                qty=qty,
                unit_price_cents=unit_cents,
            )
            session.add(order_item)

        await session.commit()

    logger.info("Order %s created for %s.", order_ref, customer_name)
    return order_ref


async def save_reservation(
    customer_name: str,
    customer_phone: str,
    date: str,
    time: str,
    guests: int,
    special_request: str = "",
) -> str:
    """Create a reservation. Returns reservation_ref (RSV-XXXXXX)."""
    res_ref = f"RSV-{uuid.uuid4().hex[:6].upper()}"

    async with AsyncSessionLocal() as session:
        # Find or create customer
        cust_result = await session.execute(
            select(Customer).where(
                Customer.restaurant_id == 1,
                Customer.phone == customer_phone,
            )
        )
        customer = cust_result.scalar_one_or_none()
        if not customer:
            customer = Customer(
                restaurant_id=1,
                name=customer_name,
                phone=customer_phone,
            )
            session.add(customer)
            await session.flush()

        reservation = Reservation(
            restaurant_id=1,
            reservation_ref=res_ref,
            customer_id=customer.id,
            date=date,
            time=time,
            guests=guests,
            special_request=special_request,
            status="CREATED",
        )
        session.add(reservation)
        await session.commit()

    logger.info("Reservation %s created for %s.", res_ref, customer_name)
    return res_ref


# ── Backward-compat aliases ───────────────────────────────────────────────────

create_order = save_order
create_reservation = save_reservation
get_menu_for_prompt = get_menu_text
