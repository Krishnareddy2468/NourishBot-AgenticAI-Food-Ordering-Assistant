"""
Memory Agent — User preference learning + retrieval.

Updates the user's Taste Vector after every completed order and
generates a short memory summary for injection into the Planner prompt.

Implicit feedback signals:
  - Reorder of same item:       +0.1 weight
  - Order cancellation:         -0.15 weight
  - Rating 5 stars:             +0.2 weight
  - Rating 1 star:              -0.3 weight
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pytz
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models.user_preferences import UserPreferences
from app.db.models import Order, OrderItem, MenuItem

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

# ── Time-of-day mapping ───────────────────────────────────────────────────────

def _time_label(now: datetime) -> str:
    h = now.astimezone(IST).hour
    if 6 <= h < 11:    return "breakfast"
    elif 11 <= h < 15: return "lunch"
    elif 15 <= h < 18: return "snack time"
    elif 18 <= h < 23: return "dinner"
    else:              return "late night"


# ── Core memory APIs ──────────────────────────────────────────────────────────

async def get_or_create_preferences(customer_id: int) -> UserPreferences:
    """Fetch existing preferences, or create a default row."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.customer_id == customer_id)
        )
        prefs = result.scalar_one_or_none()
        if prefs is not None:
            return prefs
        prefs = UserPreferences(customer_id=customer_id)
        db.add(prefs)
        await db.commit()
        await db.refresh(prefs)
        return prefs


async def get_user_preferences(customer_id: int) -> Optional[UserPreferences]:
    """Return preferences or None if user has never ordered."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.customer_id == customer_id)
        )
        return result.scalar_one_or_none()


async def upsert_user_preferences(
    db: AsyncSession,
    customer_id: int,
    updates: dict[str, Any],
) -> UserPreferences:
    """Apply partial updates to a preferences row, creating it if needed."""
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.customer_id == customer_id)
    )
    prefs = result.scalar_one_or_none()
    if prefs is None:
        prefs = UserPreferences(customer_id=customer_id)
        db.add(prefs)
    for key, value in updates.items():
        setattr(prefs, key, value)
    await db.flush()
    return prefs


# ── Taste vector updates ──────────────────────────────────────────────────────

async def update_taste_vector(
    customer_id: int,
    ordered_items: list[dict],
    rating: Optional[int] = None,
) -> None:
    """
    Update the user's taste profile after a completed order.

    ordered_items: list of {item_name, qty, price_cents, type}
    rating: optional 1–5 star rating for the order
    """
    async with AsyncSessionLocal() as db:
        prefs = await get_or_create_preferences(customer_id)

        # ── Update cuisine weights ────────────────────────────────────────
        weights = dict(prefs.cuisine_weights or {})
        for item in ordered_items:
            name = item.get("item_name", "").lower().strip()
            qty = item.get("qty", 1)
            # Find matching cuisine keyword from item name
            cuisine = _extract_cuisine_keyword(name)
            if cuisine:
                current = weights.get(cuisine, 0.0)
                weights[cuisine] = min(1.0, current + 0.08 * qty)

        # Decay all weights slightly
        for key in list(weights.keys()):
            weights[key] = max(0.0, weights[key] - 0.01)
        prefs.cuisine_weights = weights

        # ── Update order timing ───────────────────────────────────────────
        now = datetime.now(IST)
        label = _time_label(now)
        timing = dict(prefs.order_timing or {})
        timing[label] = min(1.0, timing.get(label, 0.0) + 0.05)
        prefs.order_timing = timing

        # ── Update craving cycles ─────────────────────────────────────────
        cycles = dict(prefs.craving_cycles or {})
        for item in ordered_items:
            name = item.get("item_name", "").strip()
            cuisine = _extract_cuisine_keyword(name.lower())
            if cuisine:
                entry = cycles.get(cuisine, {})
                prev = entry.get("last_ordered")
                entry["last_ordered"] = now.isoformat()
                if prev:
                    try:
                        prev_dt = datetime.fromisoformat(prev)
                        delta = (now - prev_dt).days
                        old_avg = entry.get("avg_interval_days", 7)
                        alpha = 0.3  # smoothing factor
                        entry["avg_interval_days"] = round(alpha * delta + (1 - alpha) * old_avg, 1)
                    except (ValueError, TypeError):
                        pass
                cycles[cuisine] = entry
        prefs.craving_cycles = cycles

        # ── Apply rating feedback ────────────────────────────────────────
        if rating is not None:
            if rating >= 5:
                for key in weights:
                    weights[key] = min(1.0, weights[key] + 0.2)
            elif rating <= 1:
                for key in weights:
                    weights[key] = max(0.0, weights[key] - 0.3)
            prefs.cuisine_weights = weights

        # ── Update stats ──────────────────────────────────────────────────
        prefs.total_orders = (prefs.total_orders or 0) + 1
        total_spent = sum(
            (item.get("price_cents", 0) or 0) * (item.get("qty", 1) or 1)
            for item in ordered_items
        )
        prefs.total_spent_cents = (prefs.total_spent_cents or 0) + total_spent
        prefs.last_order_at = now

        await db.commit()
        logger.debug(
            "Taste vector updated: customer=%d total_orders=%d weights=%s",
            customer_id, prefs.total_orders,
            json.dumps(weights, default=str),
        )


async def apply_implicit_feedback(customer_id: int, action: str, item_name: str) -> None:
    """
    Apply implicit feedback without a full order.

    action: "reorder" | "cancel" | "complaint"
    """
    async with AsyncSessionLocal() as db:
        prefs = await get_or_create_preferences(customer_id)
        weights = dict(prefs.cuisine_weights or {})
        cuisine = _extract_cuisine_keyword(item_name.lower())
        if cuisine:
            delta = {"reorder": 0.10, "cancel": -0.15, "complaint": -0.20}.get(action, 0.0)
            weights[cuisine] = max(0.0, min(1.0, weights.get(cuisine, 0.0) + delta))
        prefs.cuisine_weights = weights
        await db.commit()


# ── Memory summary for LLM injection ──────────────────────────────────────────

async def get_user_memory_summary(customer_id: int) -> str:
    """Return a 1–3 sentence summary of the user's food profile for the Planner."""
    if customer_id is None:
        return "(new customer — no preferences yet)"

    prefs = await get_user_preferences(customer_id)
    if prefs is None or (prefs.total_orders or 0) == 0:
        return "(new customer — no order history)"

    weights = dict(prefs.cuisine_weights or {})
    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    top_cuisines = [f"{c} ({w:.0%})" for c, w in top if w > 0.1]
    timing = dict(prefs.order_timing or {})
    fav_time = max(timing, key=timing.get) if timing else "anytime"

    parts = [f"Total orders: {prefs.total_orders}"]
    if top_cuisines:
        parts.append(f"Favorites: {', '.join(top_cuisines)}")
    parts.append(f"Usually orders: {fav_time}")
    if prefs.price_band:
        parts.append(f"Price range: {prefs.price_band}")
    if prefs.dietary_flags:
        parts.append(f"Diet: {', '.join(prefs.dietary_flags)}")
    if prefs.health_goals:
        parts.append(f"Health: {', '.join(prefs.health_goals)}")

    return "; ".join(parts)


async def get_top_cravings(customer_id: int) -> list[str]:
    """Return names of cuisines the user is most likely craving right now."""
    prefs = await get_user_preferences(customer_id)
    if prefs is None or not prefs.craving_cycles:
        return []

    now = datetime.now(IST)
    cycles = dict(prefs.craving_cycles or {})
    cravings = []
    for cuisine, entry in cycles.items():
        last_str = entry.get("last_ordered")
        avg = entry.get("avg_interval_days", 7)
        if last_str:
            try:
                last = datetime.fromisoformat(last_str)
                days_since = (now - last).days
                if days_since >= avg * 0.7:  # overdue or nearly overdue
                    cravings.append((cuisine, days_since / max(avg, 1)))
            except (ValueError, TypeError):
                pass
    cravings.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in cravings[:3]]


# ── Helpers ───────────────────────────────────────────────────────────────────

_CUISINE_MAP = {
    "biryani": ["biryani", "pulao", "rice"],
    "pizza":   ["pizza", "pasta", "italian"],
    "burger":  ["burger", "sandwich"],
    "dosa":    ["dosa", "idli", "vada", "uttapam", "south indian"],
    "curry":   ["curry", "gravy", "masala", "paneer", "chicken curry"],
    "chinese": ["noodles", "fried rice", "manchurian", "chilli"],
    "tandoori": ["tandoori", "kebab", "tikka", "roti", "naan"],
    "snack":   ["samosa", "pakora", "chaat", "pani puri", "snack"],
    "drink":   ["lassi", "juice", "shake", "coffee", "tea", "drink"],
    "dessert": ["ice cream", "kulfi", "halwa", "gulab", "rasmalai", "sweet"],
    "thali":   ["thali", "meal", "combo"],
}


def _extract_cuisine_keyword(item_name: str) -> str | None:
    """Map an item name to a broad cuisine category."""
    item_lower = item_name.lower().strip()
    for cuisine, keywords in _CUISINE_MAP.items():
        for kw in keywords:
            if kw in item_lower:
                return cuisine
    return None
