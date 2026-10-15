"""
Async CRUD helpers — PostgreSQL equivalents for the old SQLite database.py.

All functions use the async SQLAlchemy session from app.db.session.
These replace the old sync SQLite helpers used by telegram.py and the agents.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session_factory
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
from app.agent.privacy_service import ensure_channel
from app.core.config import settings
from app.domain.order_state import record_order_event

logger = logging.getLogger(__name__)


# ── Session helpers ───────────────────────────────────────────────────────────

async def _get_channel(chat_id: int) -> Optional[Channel]:
    """Find or create a channel for the given Telegram chat_id."""
    async with get_session_factory()() as session:
        channel = await ensure_channel(
            session,
            restaurant_id=settings.DEFAULT_RESTAURANT_ID,
            channel_type="TELEGRAM",
            external_id=chat_id,
        )
        await session.commit()
        return channel


async def get_session(chat_id: str | int, channel_type: str = "TELEGRAM") -> dict:
    """Return session state as a dict (same shape as old SQLite version).

    Tries Redis cache first; falls back to PostgreSQL on miss.
    """
    # ── Redis cache hit ──────────────────────────────────────────────────
    from app.core.redis_client import get_cached_session, cache_session
    channel_type = channel_type.upper()
    cached = await get_cached_session(chat_id, channel_type)
    if cached is not None:
        return cached

    # ── PostgreSQL fallback ──────────────────────────────────────────────
    async with get_session_factory()() as session:
        # Find channel
        ch_result = await session.execute(
            select(Channel).where(
                Channel.restaurant_id == settings.DEFAULT_RESTAURANT_ID,
                Channel.type == channel_type,
                Channel.external_id == str(chat_id),
            )
        )
        channel = ch_result.scalar_one_or_none()
        if not channel:
            result = {
                "chat_id": chat_id,
                "state": "new",
                "user_name": "",
                "cart": [],
                "customer_name": "",
                "customer_phone": "",
                "history": [],
                "ordering_platform": "",
            }
            await cache_session(chat_id, result, channel_type)
            return result

        # Find session
        s_result = await session.execute(
            select(SessionModel).where(
                SessionModel.channel_id == channel.id,
                SessionModel.restaurant_id == channel.restaurant_id,
            )
        )
        sess = s_result.scalar_one_or_none()
        if not sess:
            result = {
                "chat_id": chat_id,
                "state": "new",
                "user_name": "",
                "cart": [],
                "customer_name": "",
                "customer_phone": "",
                "history": [],
                "ordering_platform": "",
            }
            await cache_session(chat_id, result, channel_type)
            return result

        # Reconstruct cart from Cart + CartItems if linked
        cart_data = []
        if sess.cart_id:
            cart_result = await session.execute(
                select(Cart).where(
                    Cart.id == sess.cart_id,
                    Cart.restaurant_id == channel.restaurant_id,
                )
            )
            cart_obj = cart_result.scalar_one_or_none()
            if cart_obj:
                items_result = await session.execute(
                    select(CartItem).where(
                        CartItem.cart_id == cart_obj.id,
                        CartItem.restaurant_id == channel.restaurant_id,
                    )
                )
                for ci in items_result.scalars().all():
                    item_result = await session.execute(
                        select(MenuItem).where(
                            MenuItem.id == ci.item_id,
                            MenuItem.restaurant_id == channel.restaurant_id,
                        )
                    )
                    mi = item_result.scalar_one_or_none()
                    cart_data.append({
                        "item_name": mi.name if mi else "Unknown",
                        "qty": ci.qty,
                        "price": ci.unit_price_cents / 100,
                        "type": mi.type if mi else "",
                    })

        customer_name = ""
        customer_phone = ""
        if channel.customer_id:
            cust_result = await session.execute(
                select(Customer).where(
                    Customer.id == channel.customer_id,
                    Customer.restaurant_id == channel.restaurant_id,
                )
            )
            customer = cust_result.scalar_one_or_none()
            if customer:
                customer_name = customer.name or ""
                customer_phone = customer.phone or ""

        if isinstance(sess.history_json, list):
            history = sess.history_json
        elif isinstance(sess.history_json, dict):
            history = list(sess.history_json.get("turns") or [])
        else:
            history = []
        result = {
            "chat_id": chat_id,
            "state": sess.state,
            "user_name": "",
            "cart": cart_data,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "history": history,
            "ordering_platform": sess.ordering_platform or "",
        }
        await cache_session(chat_id, result, channel_type)
        return result


async def save_session(
    chat_id: str | int,
    updates: dict,
    channel_type: str = "TELEGRAM",
) -> None:
    """Update session fields from a dict of updates.

    Writes to PostgreSQL first, then refreshes Redis cache.
    """
    from app.core.redis_client import invalidate_session_cache
    channel_type = channel_type.upper()
    async with get_session_factory()() as session:
        # Find or create channel
        channel = await ensure_channel(
            session,
            restaurant_id=settings.DEFAULT_RESTAURANT_ID,
            channel_type=channel_type,
            external_id=chat_id,
        )

        # Find or create session
        s_result = await session.execute(
            select(SessionModel).where(
                SessionModel.channel_id == channel.id,
                SessionModel.restaurant_id == channel.restaurant_id,
            )
        )
        sess = s_result.scalar_one_or_none()
        if not sess:
            sess = SessionModel(
                restaurant_id=settings.DEFAULT_RESTAURANT_ID,
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
            sess.ordering_platform = updates["ordering_platform"]
        if "history" in updates:
            sess.history_json = updates["history"]
        if "cart" in updates:
            # Cart is handled separately via Cart/CartItem models
            pass

        await session.commit()
        await invalidate_session_cache(chat_id, channel_type)


async def reset_session(
    chat_id: str | int,
    user_name: str = "",
    channel_type: str = "TELEGRAM",
) -> None:
    """Reset session to fresh state."""
    async with get_session_factory()() as session:
        channel = await ensure_channel(
            session,
            restaurant_id=settings.DEFAULT_RESTAURANT_ID,
            channel_type=channel_type,
            external_id=chat_id,
        )

        s_result = await session.execute(
            select(SessionModel).where(
                SessionModel.channel_id == channel.id,
                SessionModel.restaurant_id == channel.restaurant_id,
            )
        )
        sess = s_result.scalar_one_or_none()
        if not sess:
            sess = SessionModel(
                restaurant_id=settings.DEFAULT_RESTAURANT_ID,
                channel_id=channel.id,
            )
            session.add(sess)

        sess.state = "greeting"
        sess.cart_id = None
        sess.ordering_platform = ""
        sess.history_json = {"meta": {}, "turns": []}
        await session.commit()
        from app.core.redis_client import invalidate_session_cache
        await invalidate_session_cache(chat_id, channel_type.upper())


# ── User preference helpers ──────────────────────────────────────────────────

async def get_user_preferences(customer_id: int) -> Optional[dict]:
    """Return user preferences as dict, or None if not found."""
    from app.agent.memory_agent import get_user_preferences as get_consent_gated_preferences
    prefs = await get_consent_gated_preferences(customer_id, settings.DEFAULT_RESTAURANT_ID)
    if prefs is None:
        return None
    return {
        "customer_id": prefs.customer_id,
        "spice_level": prefs.spice_level,
        "cuisine_weights": prefs.cuisine_weights,
        "price_band": prefs.price_band,
        "order_timing": prefs.order_timing,
        "dietary_flags": prefs.dietary_flags,
        "health_goals": prefs.health_goals,
        "allergies": prefs.allergies,
        "health_onboarding_done": prefs.health_onboarding_done,
        "craving_cycles": prefs.craving_cycles,
        "total_orders": prefs.total_orders,
        "total_spent_cents": prefs.total_spent_cents,
    }


async def save_order_rating(
    order_ref: str,
    rating: int,
    *,
    chat_id: str | int,
    channel_type: str = "TELEGRAM",
) -> int | None:
    """Save a rating only when the current channel owns the order; return order id."""
    if rating < 1 or rating > 5:
        return None
    async with get_session_factory()() as session:
        channel = (await session.execute(select(Channel).where(
            Channel.restaurant_id == settings.DEFAULT_RESTAURANT_ID,
            Channel.type == channel_type.upper(),
            Channel.external_id == str(chat_id),
        ))).scalar_one_or_none()
        if not channel or not channel.customer_id:
            return None
        order = (await session.execute(select(Order).where(
            Order.order_ref == order_ref,
            Order.restaurant_id == channel.restaurant_id,
            Order.customer_id == channel.customer_id,
        ))).scalar_one_or_none()
        if order is None:
            return None
        order.rating = rating
        await session.commit()
        return order.id


# ── Menu helpers ──────────────────────────────────────────────────────────────

async def get_menu_items(
    filter_type: str = "All",
    filter_category: str = "",
) -> list[dict]:
    """Return menu items as list of dicts (compatible with old interface)."""
    async with get_session_factory()() as session:
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

    lines = ["NOURISHBOT RESTAURANT MENU:"]
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
    """Create an order + order items. Returns order_ref (NB-XXXXXX)."""
    order_ref = f"NB-{uuid.uuid4().hex[:6].upper()}"
    idem_key = idempotency_key or f"{customer_phone}-{order_ref}"
    total_cents = int(total_price * 100)

    async with get_session_factory()() as session:
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
        record_order_event(
            session,
            order,
            event_type="ORDER_CREATED",
            to_state="CREATED",
            actor_type="SYSTEM",
            source="LEGACY_CRUD",
            reason_code="LEGACY_ORDER_CREATED",
        )

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

    async with get_session_factory()() as session:
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
