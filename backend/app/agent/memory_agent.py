"""Consent-gated personalization learned from fulfillment or explicit feedback."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import pytz
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.privacy_service import has_consent_in_session
from app.db.models import MemoryLearningEvent, Order, OrderItem
from app.db.models.user_preferences import UserPreferences
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")
FULFILLED_STATUSES = {"DELIVERED", "COMPLETED"}
LEARNING_SIGNALS = {"FULFILLMENT", "EXPLICIT_FEEDBACK"}


def _time_label(now: datetime) -> str:
    hour = now.astimezone(IST).hour
    if 6 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 18:
        return "snack time"
    if 18 <= hour < 23:
        return "dinner"
    return "late night"


def can_learn_from_order(status: str, signal_type: str, rating: int | None = None) -> bool:
    signal_type = (signal_type or "").upper()
    if signal_type == "FULFILLMENT":
        return (status or "").upper() in FULFILLED_STATUSES
    if signal_type == "EXPLICIT_FEEDBACK":
        return rating is not None and 1 <= int(rating) <= 5
    return False


async def upsert_user_preferences(
    db: AsyncSession,
    customer_id: int,
    updates: dict[str, Any] | None = None,
    *,
    restaurant_id: int | None = None,
) -> UserPreferences | None:
    """Upsert preferences inside the caller transaction, only with active consent."""
    if not await has_consent_in_session(
        db,
        customer_id=customer_id,
        purpose="PERSONALIZATION",
        restaurant_id=restaurant_id,
    ):
        return None
    prefs = (await db.execute(
        select(UserPreferences).where(UserPreferences.customer_id == customer_id).with_for_update()
    )).scalar_one_or_none()
    if prefs is None:
        prefs = UserPreferences(customer_id=customer_id)
        db.add(prefs)
        await db.flush()
    for key, value in (updates or {}).items():
        if hasattr(prefs, key):
            setattr(prefs, key, value)
    return prefs


async def get_or_create_preferences(customer_id: int, restaurant_id: int | None = None) -> UserPreferences | None:
    """Compatibility helper; does not create memory without personalization consent."""
    async with get_session_factory()() as db:
        prefs = await upsert_user_preferences(db, customer_id, restaurant_id=restaurant_id)
        if prefs is None:
            return None
        await db.commit()
        return prefs


async def get_user_preferences(customer_id: int, restaurant_id: int | None = None) -> Optional[UserPreferences]:
    """Return preferences only while personalization consent is active."""
    async with get_session_factory()() as db:
        if not await has_consent_in_session(
            db,
            customer_id=customer_id,
            purpose="PERSONALIZATION",
            restaurant_id=restaurant_id,
        ):
            return None
        return (await db.execute(
            select(UserPreferences).where(UserPreferences.customer_id == customer_id)
        )).scalar_one_or_none()


def _apply_fulfillment_signal(
    prefs: UserPreferences,
    ordered_items: list[dict],
    *,
    occurred_at: datetime,
) -> None:
    weights = dict(prefs.cuisine_weights or {})
    for item in ordered_items:
        cuisine = _extract_cuisine_keyword(str(item.get("item_name") or ""))
        if cuisine:
            quantity = max(1, int(item.get("qty") or 1))
            weights[cuisine] = min(1.0, weights.get(cuisine, 0.0) + 0.08 * quantity)
    for key in list(weights):
        weights[key] = max(0.0, weights[key] - 0.01)
    prefs.cuisine_weights = weights

    timing = dict(prefs.order_timing or {})
    label = _time_label(occurred_at)
    timing[label] = min(1.0, timing.get(label, 0.0) + 0.05)
    prefs.order_timing = timing

    cycles = dict(prefs.craving_cycles or {})
    for item in ordered_items:
        cuisine = _extract_cuisine_keyword(str(item.get("item_name") or ""))
        if not cuisine:
            continue
        entry = dict(cycles.get(cuisine) or {})
        previous = entry.get("last_ordered")
        entry["last_ordered"] = occurred_at.isoformat()
        if previous:
            try:
                previous_at = datetime.fromisoformat(previous)
                if previous_at.tzinfo is None:
                    previous_at = IST.localize(previous_at)
                interval = max(0, (occurred_at - previous_at).days)
                old_average = float(entry.get("avg_interval_days", 7))
                entry["avg_interval_days"] = round(0.3 * interval + 0.7 * old_average, 1)
            except (TypeError, ValueError):
                pass
        cycles[cuisine] = entry
    prefs.craving_cycles = cycles
    prefs.total_orders = (prefs.total_orders or 0) + 1
    prefs.total_spent_cents = (prefs.total_spent_cents or 0) + sum(
        int(item.get("price_cents") or 0) * max(1, int(item.get("qty") or 1))
        for item in ordered_items
    )
    prefs.last_order_at = occurred_at


def _apply_rating_signal(prefs: UserPreferences, ordered_items: list[dict], rating: int) -> None:
    weights = dict(prefs.cuisine_weights or {})
    delta = 0.20 if rating >= 5 else (-0.30 if rating <= 1 else 0.0)
    if delta:
        for item in ordered_items:
            cuisine = _extract_cuisine_keyword(str(item.get("item_name") or ""))
            if cuisine:
                weights[cuisine] = max(0.0, min(1.0, weights.get(cuisine, 0.0) + delta))
    prefs.cuisine_weights = weights


async def learn_from_order(
    order_id: int,
    *,
    signal_type: str,
    rating: int | None = None,
    restaurant_id: int | None = None,
) -> bool:
    """Apply one idempotent learning signal in one database transaction."""
    signal_type = (signal_type or "").upper()
    if signal_type not in LEARNING_SIGNALS:
        return False
    async with get_session_factory()() as db:
        conditions = [Order.id == order_id]
        if restaurant_id is not None:
            conditions.append(Order.restaurant_id == restaurant_id)
        order = (await db.execute(select(Order).where(*conditions))).scalar_one_or_none()
        if not order or not order.customer_id or not can_learn_from_order(order.status, signal_type, rating):
            return False
        if not await has_consent_in_session(
            db,
            customer_id=order.customer_id,
            purpose="PERSONALIZATION",
            restaurant_id=order.restaurant_id,
        ):
            return False
        existing = (await db.execute(
            select(MemoryLearningEvent).where(
                MemoryLearningEvent.order_id == order.id,
                MemoryLearningEvent.signal_type == signal_type,
            )
        )).scalar_one_or_none()
        if existing:
            return False

        rows = (await db.execute(
            select(OrderItem).where(
                OrderItem.order_id == order.id,
                OrderItem.restaurant_id == order.restaurant_id,
            )
        )).scalars().all()
        items = [
            {"item_name": item.item_name_snapshot, "qty": item.qty, "price_cents": item.unit_price_cents}
            for item in rows
        ]
        prefs = await upsert_user_preferences(db, order.customer_id, restaurant_id=order.restaurant_id)
        if prefs is None:
            return False
        occurred_at = order.scheduled_for or datetime.now(IST)
        if occurred_at.tzinfo is None:
            occurred_at = IST.localize(occurred_at)
        if signal_type == "FULFILLMENT":
            _apply_fulfillment_signal(prefs, items, occurred_at=occurred_at)
        else:
            _apply_rating_signal(prefs, items, int(rating))
        db.add(MemoryLearningEvent(
            restaurant_id=order.restaurant_id,
            customer_id=order.customer_id,
            order_id=order.id,
            signal_type=signal_type,
            rating=rating,
        ))
        await db.commit()
        logger.info(
            "Accepted personalization signal customer=%s order=%s signal=%s",
            order.customer_id,
            order.order_ref,
            signal_type,
        )
        return True


async def update_taste_vector(
    customer_id: int,
    ordered_items: list[dict],
    rating: Optional[int] = None,
    *,
    order_id: int | None = None,
) -> bool:
    """Compatibility wrapper that refuses unverifiable free-standing learning."""
    if not order_id:
        logger.warning("Skipped personalization without an order id for customer=%s", customer_id)
        return False
    return await learn_from_order(
        order_id,
        signal_type="EXPLICIT_FEEDBACK" if rating is not None else "FULFILLMENT",
        rating=rating,
    )


async def apply_implicit_feedback(customer_id: int, action: str, item_name: str) -> bool:
    """Legacy entry point: unverifiable implicit events are intentionally ignored."""
    logger.info("Ignored unverified implicit signal customer=%s action=%s", customer_id, action)
    return False


async def get_user_memory_summary(customer_id: int, restaurant_id: int | None = None) -> str:
    if customer_id is None:
        return "(new customer — no preferences yet)"
    prefs = await get_user_preferences(customer_id, restaurant_id)
    if prefs is None:
        return "(personalization disabled or no stored preferences)"
    if (prefs.total_orders or 0) == 0:
        return "(personalization enabled — no fulfilled-order history yet)"
    weights = dict(prefs.cuisine_weights or {})
    top = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
    top_cuisines = [f"{name} ({weight:.0%})" for name, weight in top if weight > 0.1]
    timing = dict(prefs.order_timing or {})
    favorite_time = max(timing, key=timing.get) if timing else "anytime"
    parts = [f"Fulfilled orders learned: {prefs.total_orders}"]
    if top_cuisines:
        parts.append(f"Favorites: {', '.join(top_cuisines)}")
    parts.append(f"Usually orders: {favorite_time}")
    if prefs.price_band:
        parts.append(f"Price range: {prefs.price_band}")
    if prefs.dietary_flags:
        parts.append(f"Diet: {', '.join(prefs.dietary_flags)}")
    if prefs.health_goals:
        parts.append(f"Health: {', '.join(prefs.health_goals)}")
    return "; ".join(parts)


async def get_top_cravings(customer_id: int, restaurant_id: int | None = None) -> list[str]:
    prefs = await get_user_preferences(customer_id, restaurant_id)
    if prefs is None or not prefs.craving_cycles:
        return []
    now = datetime.now(IST)
    cravings = []
    for cuisine, entry in dict(prefs.craving_cycles or {}).items():
        last_value = entry.get("last_ordered")
        average = entry.get("avg_interval_days", 7)
        if not last_value:
            continue
        try:
            last = datetime.fromisoformat(last_value)
            if last.tzinfo is None:
                last = IST.localize(last)
            days_since = (now - last).days
            if days_since >= average * 0.7:
                cravings.append((cuisine, days_since / max(average, 1)))
        except (TypeError, ValueError):
            continue
    cravings.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in cravings[:3]]


_CUISINE_MAP = {
    "biryani": ["biryani", "pulao", "rice"],
    "pizza": ["pizza", "pasta", "italian"],
    "burger": ["burger", "sandwich"],
    "dosa": ["dosa", "idli", "vada", "uttapam", "south indian"],
    "curry": ["curry", "gravy", "masala", "paneer", "chicken curry"],
    "chinese": ["noodles", "fried rice", "manchurian", "chilli"],
    "tandoori": ["tandoori", "kebab", "tikka", "roti", "naan"],
    "snack": ["samosa", "pakora", "chaat", "pani puri", "snack"],
    "drink": ["lassi", "juice", "shake", "coffee", "tea", "drink"],
    "dessert": ["ice cream", "kulfi", "halwa", "gulab", "rasmalai", "sweet"],
    "thali": ["thali", "meal", "combo"],
}


def _extract_cuisine_keyword(item_name: str) -> str | None:
    item_lower = item_name.lower().strip()
    for cuisine, keywords in _CUISINE_MAP.items():
        if any(keyword in item_lower for keyword in keywords):
            return cuisine
    return None
