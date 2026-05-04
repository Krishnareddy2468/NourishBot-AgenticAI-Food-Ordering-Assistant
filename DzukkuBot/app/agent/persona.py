"""
Persona — §3.5 Restaurant Identity Behaviors

Centralises everything that makes the Dzukku agent consistent and realistic:

  1. Language mirroring  — detects EN / TE+EN / HI+EN from message + history
  2. Tone calibration    — shifts warmth/urgency by time-of-day and bot state
  3. Food-first boundary — detects and deflects off-topic messages gracefully
  4. Live-state realism  — helpers for:
       a. Out-of-stock alternatives (find similar available items)
       b. Kitchen load → ETA label + apology note
       c. Delivery radius validation → offer pickup/dine-in fallback

All functions are pure or async DB-read — no writes, no LLM calls.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import MenuItem
from app.agent.policies import RestaurantPolicy, default_policy

logger = logging.getLogger(__name__)


# ── 1. Language Detection ─────────────────────────────────────────────────────

# Telugu trigger words (common in TE+EN code-mix)
_TE_TOKENS = {
    "emi", "em", "ante", "cheppandi", "anni", "ayindi", "undi", "ledhu",
    "naaku", "mee", "meeru", "aithe", "kadha", "kavali", "icheyandi",
    "cheyandi", "cheyyandi", "pampandi", "teeyandi", "okka", "rendu",
    "moodu", "naalugu", "aidu", "oka", "ela", "enduku", "ekkada",
    "vachindi", "poyindi", "ichindhi", "tindam", "tinali",
}

# Hindi trigger words (common in HI+EN code-mix)
_HI_TOKENS = {
    "kya", "hai", "hain", "mujhe", "chahiye", "bhai", "yaar", "dost",
    "khana", "order", "lagao", "karo", "dijiye", "dijie", "batao",
    "theek", "acha", "accha", "nahi", "nai", "abhi", "jaldi", "please",
    "ek", "do", "teen", "char", "paanch", "aur", "bhi", "toh",
}

# Off-topic signals — things clearly outside food/restaurant domain
_OFF_TOPIC_PATTERNS = [
    r"\b(weather|cricket|ipl|politics|news|stock|crypto|bitcoin)\b",
    r"\b(tell me a joke|joke|poem|story|essay|code|program)\b",
    r"\b(girlfriend|boyfriend|relationship|marriage|divorce)\b",
    r"\b(doctor|medicine|hospital|pharmacy)\b",
    r"\b(flight|train|bus|ticket|hotel)\b",
]
_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)

# Food-adjacent words (override off-topic detection if present)
_FOOD_SIGNALS = {
    "menu", "order", "food", "eat", "drink", "biryani", "pizza", "burger",
    "chicken", "veg", "table", "reserve", "delivery", "pickup", "cart",
    "price", "items", "restaurant", "dzukku", "bill", "invoice",
}


def detect_language(message: str, history: list[dict]) -> str:
    """
    Returns 'en', 'te+en', or 'hi+en'.
    Checks current message first; if ambiguous, scans last 4 turns.
    """
    def _score(text: str) -> tuple[int, int]:
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return len(tokens & _TE_TOKENS), len(tokens & _HI_TOKENS)

    te, hi = _score(message)
    if te == 0 and hi == 0:
        for turn in reversed((history or [])[-4:]):
            t2, h2 = _score(turn.get("content") or "")
            te += t2; hi += h2
            if te or hi:
                break

    if te > hi and te >= 1:
        return "te+en"
    if hi > te and hi >= 1:
        return "hi+en"
    return "en"


_FOOD_SIGNAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(_FOOD_SIGNALS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def is_off_topic(message: str) -> bool:
    """Return True if message is clearly outside food/restaurant scope."""
    # If any food signal appears as a whole word, it's relevant
    if _FOOD_SIGNAL_RE.search(message):
        return False
    return bool(_OFF_TOPIC_RE.search(message))


OFF_TOPIC_REPLY = (
    "Ha! I'm a food-only expert 😄 "
    "Try me on menu, orders, or table bookings!"
)


# ── 2. Tone Calibration ───────────────────────────────────────────────────────

def tone_for_context(time_of_day: str, bot_state_value: str) -> str:
    """
    Returns a short tone instruction string injected into the Responder prompt.
    Shifts register based on time + current pipeline state.
    """
    base = "warm, friendly, concise — like texting a food-loving friend"

    if bot_state_value in ("AWAITING_PAYMENT",):
        return f"{base}; reassure the customer, guide them through payment calmly"
    if bot_state_value in ("ORDER_IN_PROGRESS", "OUT_FOR_DELIVERY"):
        return f"{base}; keep it upbeat — food is on its way!"
    if bot_state_value == "SUPPORT_CASE":
        return f"{base}; empathetic and solution-focused — acknowledge the issue first"
    if bot_state_value == "COMPLETED":
        return f"{base}; warm close — thank them, invite them back"

    if time_of_day == "late night":
        return f"{base}; acknowledge the late hour with a hint of humour"
    if time_of_day == "morning":
        return f"{base}; energetic good-morning vibe"
    if time_of_day in ("lunch time", "dinner time"):
        return f"{base}; peak-hour efficiency — keep it snappy"
    return base


# ── 3. Out-of-stock Alternatives ─────────────────────────────────────────────

async def find_alternatives(
    item_name: str,
    restaurant_id: int,
    veg_only: bool = False,
    price_hint_cents: int = 0,
    limit: int = 3,
) -> list[dict]:
    """
    Find available menu items similar to `item_name`.
    Similarity heuristic:
      1. Same type (Veg/Non-Veg) — if veg_only, restrict to Veg/Vegan
      2. Price within ±50% of price_hint (if given)
      3. Fallback: any available item
    Returns at most `limit` items.
    """
    async with AsyncSessionLocal() as db:
        q = select(MenuItem).where(
            MenuItem.restaurant_id == restaurant_id,
            MenuItem.available == True,
            MenuItem.name != item_name,
        )
        if veg_only:
            q = q.where(MenuItem.type.in_(["VEG", "VEGAN"]))

        result = await db.execute(q.order_by(MenuItem.name))
        candidates = result.scalars().all()

    if not candidates:
        return []

    # Score by name similarity + price proximity
    name_tokens = set(re.findall(r"\w+", item_name.lower()))
    scored = []
    for mi in candidates:
        mi_tokens = set(re.findall(r"\w+", mi.name.lower()))
        overlap = len(name_tokens & mi_tokens)
        price_diff = abs(mi.price_cents - price_hint_cents) if price_hint_cents else 0
        scored.append((overlap * 100 - price_diff // 100, mi))

    scored.sort(key=lambda x: -x[0])
    top = [mi for _, mi in scored[:limit]]

    return [
        {
            "id": mi.id,
            "name": mi.name,
            "type": mi.type or "",
            "price_cents": mi.price_cents,
            "price": mi.price_cents / 100,
        }
        for mi in top
    ]


# ── 4. Kitchen Load → ETA + Apology ──────────────────────────────────────────

class KitchenSignal:
    NORMAL  = "NORMAL"   # < 50% capacity
    BUSY    = "BUSY"     # 50-80%
    VERY_BUSY = "VERY_BUSY"  # > 80%
    FULL    = "FULL"     # at cap


def kitchen_load_signal(kitchen_load: int, policy: RestaurantPolicy) -> str:
    max_orders = policy.max_concurrent_orders
    ratio = kitchen_load / max_orders if max_orders else 0
    if ratio >= 1.0:
        return KitchenSignal.FULL
    if ratio >= 0.8:
        return KitchenSignal.VERY_BUSY
    if ratio >= 0.5:
        return KitchenSignal.BUSY
    return KitchenSignal.NORMAL


def eta_message(eta_seconds: int, signal: str) -> str:
    """Build a realistic ETA string with appropriate apology copy."""
    eta_min = max(1, eta_seconds // 60)

    if signal == KitchenSignal.FULL:
        return (
            f"Kitchen's at full capacity right now 🍳 "
            f"— we can't take new orders for ~{eta_min} mins. "
            f"Please try again shortly or choose pickup!"
        )
    if signal == KitchenSignal.VERY_BUSY:
        return (
            f"It's a busy evening here! 🔥 "
            f"Your food should be ready in about *{eta_min} mins* — "
            f"slightly longer than usual. Sorry for the wait!"
        )
    if signal == KitchenSignal.BUSY:
        return f"Prep time: ~*{eta_min} mins* (kitchen's buzzing today 😄)"
    return f"Prep time: ~*{eta_min} mins*"


# ── 5. Delivery Radius Validation ─────────────────────────────────────────────

# Known serviceable areas for Hyderabad (extend from settings/DB in production)
_SERVICEABLE_AREAS = {
    "hyderabad", "secunderabad", "cyberabad",
    "kondapur", "gachibowli", "madhapur", "hitech city", "hitec city",
    "jubilee hills", "banjara hills", "ameerpet", "begumpet",
    "kukatpally", "miyapur", "madinaguda",
    "sr nagar", "sanjeeva reddy nagar", "ameerpet",
    "dilsukhnagar", "lb nagar", "uppal", "nacharam",
    "tarnaka", "malkajgiri", "alwal", "kompally",
    "manikonda", "nanakramguda", "financial district",
    "tolichowki", "mehdipatnam", "attapur",
    "lakdikapul", "khairatabad", "masab tank",
    "somajiguda", "punjagutta",
}

# Clearly out-of-range cities
_OUT_OF_RANGE_CITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "kolkata",
    "pune", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur",
    "nagpur", "visakhapatnam", "vizag", "vijayawada", "warangal",
    "nellore", "tirupati", "kurnool",
}


def validate_delivery_address(address: str, policy: RestaurantPolicy) -> tuple[bool, str]:
    """
    Returns (is_deliverable, rejection_reason_or_empty).
    Simple area/city heuristic — in production, use coordinates + Haversine.
    """
    if not address:
        return False, "Delivery address is required."

    addr_lower = address.lower()

    # Hard reject: clearly out-of-range city
    for city in _OUT_OF_RANGE_CITIES:
        if city in addr_lower:
            return (
                False,
                f"Sorry, we currently deliver only within Hyderabad 🏙️ "
                f"— {city.title()} is outside our delivery zone. "
                f"Would you like to pick up or book a table instead?",
            )

    # Check if a known serviceable area is mentioned
    for area in _SERVICEABLE_AREAS:
        if area in addr_lower:
            return True, ""

    # Address doesn't mention a known area — allow but warn
    # (real implementation would geocode and check radius)
    if not policy.delivery_available:
        return False, "Delivery is currently unavailable. Please choose pickup or dine-in."

    return True, ""  # allow — we can't detect every sub-locality


# ── 6. Language-aware CTA templates ──────────────────────────────────────────

_CTA = {
    "en": {
        "order_more":   "Want to add anything else to your order?",
        "confirm":      "Shall I place this order?",
        "help":         "Anything else I can help with? 😊",
        "menu_browse":  "Want to explore the menu? I can filter by Veg, Non-Veg, or category!",
        "slot_name":    "What's your name?",
        "slot_order_type": "Is this for delivery, pickup, or dine-in?",
        "slot_phone":   "Can I get your phone number?",
        "slot_address": "What's your delivery address?",
        "slot_date":    "Which date works for you?",
        "slot_time":    "What time would you prefer?",
        "slot_guests":  "How many guests will be joining?",
    },
    "te+en": {
        "order_more":   "Inkemi add cheyyalantara? 😊",
        "confirm":      "Order place cheyyalama?",
        "help":         "Inkemi kavali? 😊",
        "menu_browse":  "Menu chudatam try cheyandamma — Veg leda Non-Veg?",
        "slot_name":    "Meeru peru cheppagalara?",
        "slot_order_type": "Delivery, pickup, leda dine-in ah?",
        "slot_phone":   "Mee phone number ivagalara?",
        "slot_address": "Delivery address cheppandi.",
        "slot_date":    "Evaroj ravataniki convenient ga untundi?",
        "slot_time":    "Ela time select chesukuntaru?",
        "slot_guests":  "Entha mandi vastunnaru?",
    },
    "hi+en": {
        "order_more":   "Kuch aur add karein? 😊",
        "confirm":      "Order place kar doon?",
        "help":         "Kuch aur chahiye? 😊",
        "menu_browse":  "Menu dekhein — Veg ya Non-Veg?",
        "slot_name":    "Aapka naam kya hai?",
        "slot_order_type": "Delivery, pickup, ya dine-in?",
        "slot_phone":   "Aapka phone number dein please?",
        "slot_address": "Delivery address bataiye.",
        "slot_date":    "Kaunsi date theek rahegi?",
        "slot_time":    "Kaunsa time prefer karein?",
        "slot_guests":  "Kitne log aayenge?",
    },
}


def get_cta(key: str, language: str) -> str:
    lang = language if language in _CTA else "en"
    return _CTA[lang].get(key, _CTA["en"].get(key, ""))


def slot_question(slot: str, language: str) -> str:
    slot = {
        "customer_name": "name",
        "customer_phone": "phone",
        "delivery_address": "address",
    }.get(slot, slot)
    key = f"slot_{slot}"
    return get_cta(key, language) or f"Could you provide your {slot}?"
