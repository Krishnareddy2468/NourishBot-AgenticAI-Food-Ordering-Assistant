"""
Context Builder — Stage 1 of the Dzukku Pipeline.

Assembles a complete ContextSnapshot for every incoming message:
  - Customer profile (name, phone, language, opt-in)
  - Live cart contents from DB (Cart / CartItem models)
  - Active orders in flight
  - Kitchen load
  - Available menu snapshot
  - Rolling conversation history (last 8 turns)
  - Session meta: pending_goal, pending_slots, upsell_count
  - Restaurant policy + open/closed status
  - Time-of-day label

The LLM (Planner / Responder) reads this; it never queries the DB itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pytz
from sqlalchemy import select
from sqlalchemy import func as sqlfunc

from app.db.session import AsyncSessionLocal
from app.db.models import (
    Channel,
    Session as SessionModel,
    Customer,
    Cart,
    CartItem,
    MenuItem,
    Order,
)
from app.agent.policies import RestaurantPolicy, default_policy
from app.agent.state_machine import BotState
from app.agent.memory_agent import get_user_memory_summary, get_top_cravings

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


# ── Snapshot dataclass ────────────────────────────────────────────────────────

@dataclass
class ContextSnapshot:
    chat_id: int
    restaurant_id: int = 1
    user_name: str = ""

    # Customer (may be None for first-time users)
    customer_id: Optional[int] = None
    customer_name: str = ""
    customer_phone: str = ""
    language: str = "en"
    user_language: str = "en"  # persisted across sessions (en / te / hi)
    marketing_opt_in: bool = False

    # Cart — list of {item_id, item_name, qty, unit_price_cents, available, type}
    cart: list[dict] = field(default_factory=list)
    cart_total_cents: int = 0
    cart_id: Optional[int] = None

    # Active orders — list of {order_ref, status, total_cents, order_type, created_at}
    active_orders: list[dict] = field(default_factory=list)

    # DB session / channel IDs
    channel_id: Optional[int] = None
    session_id: Optional[int] = None

    # State machine
    current_state: BotState = BotState.IDLE

    # Pipeline state (persisted across turns)
    pending_goal: Optional[str] = None
    pending_slots: list[str] = field(default_factory=list)
    upsell_count: int = 0

    # User memory (Phase 1 — persistent preferences across sessions)
    user_preferences: Optional[dict] = None
    memory_summary: str = ""
    top_cravings: list[str] = field(default_factory=list)

    # Order context (collected progressively)
    order_type: Optional[str] = None               # DELIVERY | PICKUP | DINE_IN
    delivery_address: str = ""
    pending_payment_order_id: Optional[str] = None  # Razorpay order ID
    pending_order_ref: Optional[str] = None         # order ref awaiting payment

    # Restaurant
    policy: Any = field(default_factory=lambda: default_policy)
    is_open: bool = True
    kitchen_load: int = 0
    menu_snapshot: list[dict] = field(default_factory=list)

    # Conversation history [{role, content}, ...]
    last_turns: list[dict] = field(default_factory=list)

    # Time
    time_of_day: str = "evening"
    now: datetime = field(default_factory=datetime.now)

    # Convenience: formatted cart for prompts
    @property
    def cart_summary(self) -> str:
        if not self.cart:
            return "(empty)"
        lines = [f"  {c['qty']}x {c['item_name']} — ₹{c['unit_price_cents'] * c['qty'] // 100}" for c in self.cart]
        lines.append(f"  Total: ₹{self.cart_total_cents // 100}")
        return "\n".join(lines)

    @property
    def active_order_summary(self) -> str:
        if not self.active_orders:
            return "None"
        return "; ".join(f"{o['order_ref']} ({o['status']})" for o in self.active_orders)


# ── Builder ───────────────────────────────────────────────────────────────────

async def build_context(chat_id: int, user_name: str = "") -> ContextSnapshot:
    """Fetch all context from DB and return a populated ContextSnapshot."""
    ctx = ContextSnapshot(chat_id=chat_id, user_name=user_name)

    # Time context
    now = datetime.now(IST)
    ctx.now = now
    h = now.hour
    ctx.time_of_day = (
        "morning"     if 6  <= h < 11 else
        "lunch time"  if 11 <= h < 15 else
        "snack time"  if 15 <= h < 18 else
        "dinner time" if 18 <= h < 23 else
        "late night"
    )
    ctx.is_open = default_policy.is_within_operating_hours(now)

    async with AsyncSessionLocal() as db:
        # ── 1. Channel ────────────────────────────────────────────────────────
        ch_result = await db.execute(
            select(Channel).where(
                Channel.type == "TELEGRAM",
                Channel.external_id == str(chat_id),
            )
        )
        channel = ch_result.scalar_one_or_none()

        if channel:
            ctx.channel_id = channel.id

            # ── 2. Customer ───────────────────────────────────────────────────
            if channel.customer_id:
                cust_result = await db.execute(
                    select(Customer).where(Customer.id == channel.customer_id)
                )
                customer = cust_result.scalar_one_or_none()
                if customer:
                    ctx.customer_id = customer.id
                    ctx.customer_name = customer.name or ""
                    ctx.customer_phone = customer.phone or ""
                    ctx.language = customer.language_pref or "en"
                    ctx.marketing_opt_in = bool(customer.marketing_opt_in)

                    # ── Load user memory for personalization (Phase 1) ───
                    try:
                        ctx.memory_summary = await get_user_memory_summary(ctx.customer_id)
                        ctx.top_cravings = await get_top_cravings(ctx.customer_id)
                    except Exception:
                        pass  # memory is non-critical

            # ── 3. Session + history + cart ───────────────────────────────────
            sess_result = await db.execute(
                select(SessionModel).where(SessionModel.channel_id == channel.id)
            )
            session = sess_result.scalar_one_or_none()

            if session:
                ctx.session_id = session.id
                _parse_session_history(ctx, session.history_json)

                if session.cart_id:
                    await _load_cart(ctx, db, session.cart_id)

        # ── 4. Active orders ──────────────────────────────────────────────────
        if ctx.customer_id:
            await _load_active_orders(ctx, db)

        # ── 5. Kitchen load ───────────────────────────────────────────────────
        kl_result = await db.execute(
            select(sqlfunc.count(Order.id)).where(
                Order.restaurant_id == ctx.restaurant_id,
                Order.status.in_(["ACCEPTED", "PREPARING"]),
            )
        )
        ctx.kitchen_load = kl_result.scalar() or 0

        # ── 6. Menu snapshot ──────────────────────────────────────────────────
        ctx.menu_snapshot = await _load_menu_snapshot(db, ctx.restaurant_id)

    return ctx


# ── Sub-loaders ───────────────────────────────────────────────────────────────

def _parse_session_history(ctx: ContextSnapshot, history_json: Any) -> None:
    """Parse history_json — handles both legacy list format and new dict format."""
    if not history_json:
        return
    if isinstance(history_json, list):
        ctx.last_turns = history_json[-8:]
        return
    if isinstance(history_json, dict):
        meta = history_json.get("meta") or {}
        ctx.pending_goal = meta.get("pending_goal")
        ctx.pending_slots = list(meta.get("pending_slots") or [])
        ctx.upsell_count = int(meta.get("upsell_count") or 0)
        ctx.order_type = meta.get("order_type")
        ctx.delivery_address = meta.get("delivery_address") or ""
        ctx.pending_payment_order_id = meta.get("pending_payment_order_id")
        ctx.pending_order_ref = meta.get("pending_order_ref")
        ctx.current_state = BotState.from_str(meta.get("current_state") or "IDLE")
        ctx.last_turns = (history_json.get("turns") or [])[-8:]


async def _load_cart(ctx: ContextSnapshot, db, cart_id: int) -> None:
    """Load cart items into ctx.cart and compute cart_total_cents."""
    cart_result = await db.execute(
        select(Cart).where(Cart.id == cart_id, Cart.status == "OPEN")
    )
    cart = cart_result.scalar_one_or_none()
    if not cart:
        return

    ctx.cart_id = cart.id
    items_result = await db.execute(
        select(CartItem).where(CartItem.cart_id == cart.id)
    )
    total = 0
    for ci in items_result.scalars().all():
        mi_result = await db.execute(select(MenuItem).where(MenuItem.id == ci.item_id))
        mi = mi_result.scalar_one_or_none()
        ctx.cart.append({
            "item_id": ci.item_id,
            "item_name": mi.name if mi else "Unknown",
            "qty": ci.qty,
            "unit_price_cents": ci.unit_price_cents,
            "available": bool(mi.available) if mi else False,
            "type": mi.type or "" if mi else "",
        })
        total += ci.unit_price_cents * ci.qty
    ctx.cart_total_cents = total


async def _load_active_orders(ctx: ContextSnapshot, db) -> None:
    """Load customer's in-flight orders."""
    result = await db.execute(
        select(Order).where(
            Order.customer_id == ctx.customer_id,
            Order.status.in_(["CREATED", "ACCEPTED", "PREPARING", "READY", "OUT_FOR_DELIVERY"]),
        ).order_by(Order.created_at.desc()).limit(5)
    )
    for order in result.scalars().all():
        ctx.active_orders.append({
            "order_ref": order.order_ref,
            "status": order.status,
            "total_cents": order.total_cents,
            "order_type": order.order_type,
            "created_at": order.created_at.isoformat() if order.created_at else "",
        })


async def _load_menu_snapshot(db, restaurant_id: int) -> list[dict]:
    result = await db.execute(
        select(MenuItem)
        .where(MenuItem.restaurant_id == restaurant_id, MenuItem.available == True)
        .order_by(MenuItem.name)
    )
    return [
        {
            "id": i.id,
            "name": i.name,
            "type": i.type or "",
            "price_cents": i.price_cents,
            "price": round(i.price_cents / 100, 2),
            "description": i.description or "",
            "category_id": i.category_id,
        }
        for i in result.scalars().all()
    ]


# ── Persistence ───────────────────────────────────────────────────────────────

async def save_session_meta(chat_id: int, updates: dict) -> None:
    """Patch specific keys inside history_json.meta without touching turns."""
    async with AsyncSessionLocal() as db:
        ch_result = await db.execute(
            select(Channel).where(Channel.type == "TELEGRAM", Channel.external_id == str(chat_id))
        )
        channel = ch_result.scalar_one_or_none()
        if not channel:
            return
        sess_result = await db.execute(
            select(SessionModel).where(SessionModel.channel_id == channel.id)
        )
        session = sess_result.scalar_one_or_none()
        if not session:
            return
        history_data = session.history_json or {}
        if isinstance(history_data, list):
            history_data = {"meta": {}, "turns": history_data}
        meta = dict(history_data.get("meta") or {})
        meta.update(updates)
        session.history_json = {**history_data, "meta": meta}
        await db.commit()


async def save_pipeline_turn(
    chat_id: int,
    user_message: str,
    assistant_reply: str,
    pending_goal: Optional[str] = None,
    pending_slots: Optional[list[str]] = None,
    upsell_count: int = 0,
    current_state: str = "IDLE",
    order_type: Optional[str] = None,
    delivery_address: str = "",
    pending_payment_order_id: Optional[str] = None,
    pending_order_ref: Optional[str] = None,
) -> None:
    """Append the completed turn to session history and save pipeline meta."""
    async with AsyncSessionLocal() as db:
        # Find or create channel
        ch_result = await db.execute(
            select(Channel).where(
                Channel.type == "TELEGRAM",
                Channel.external_id == str(chat_id),
            )
        )
        channel = ch_result.scalar_one_or_none()
        if not channel:
            channel = Channel(restaurant_id=1, type="TELEGRAM", external_id=str(chat_id))
            db.add(channel)
            await db.flush()

        # Find or create session
        sess_result = await db.execute(
            select(SessionModel).where(SessionModel.channel_id == channel.id)
        )
        session = sess_result.scalar_one_or_none()
        if not session:
            session = SessionModel(
                restaurant_id=1,
                channel_id=channel.id,
                state="IDLE",
                history_json={"meta": {}, "turns": []},
            )
            db.add(session)
            await db.flush()

        # Migrate legacy list → new dict format
        history_data = session.history_json or {}
        if isinstance(history_data, list):
            history_data = {"meta": {}, "turns": history_data}

        turns = list(history_data.get("turns") or [])
        turns.append({"role": "user", "content": user_message})
        turns.append({"role": "assistant", "content": assistant_reply})
        turns = turns[-16:]  # keep last 8 back-and-forth

        session.history_json = {
            "meta": {
                "current_state": current_state,
                "pending_goal": pending_goal,
                "pending_slots": pending_slots or [],
                "upsell_count": upsell_count,
                "order_type": order_type,
                "delivery_address": delivery_address,
                "pending_payment_order_id": pending_payment_order_id,
                "pending_order_ref": pending_order_ref,
            },
            "turns": turns,
        }
        session.state = current_state
        await db.commit()
