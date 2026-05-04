"""
Dzukku Agent — LangGraph ReAct agent for in-house Dzukku Restaurant orders.
============================================================================

Uses PostgreSQL-backed CRUD helpers for sessions, orders, and reservations,
exposed as LangChain tools bound to a Gemini-backed LangGraph create_react_agent.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.db.crud import (
    get_menu_items,
    get_session,
    save_order,
    save_reservation,
    save_session,
)

logger = logging.getLogger(__name__)


# ── Sync wrapper for async CRUD ───────────────────────────────────────────────

def _sync_await(coro):
    """Run an async CRUD function from sync tool code."""
    try:
        loop = asyncio.get_running_loop()
        # We're inside an already-running loop (LangGraph runs async)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly
        return asyncio.run(coro)


# ── Per-chat context (threaded into every tool call) ────────────────────────

_chat_ctx:      contextvars.ContextVar[int | None] = contextvars.ContextVar("dzukku_chat_id",   default=None)
_user_name_ctx: contextvars.ContextVar[str]        = contextvars.ContextVar("dzukku_user_name", default="")


def _current_chat_id() -> int | None:
    return _chat_ctx.get()


def _session() -> dict:
    cid = _current_chat_id()
    if cid is None:
        return {}
    return _sync_await(get_session(cid))


def _persist(updates: dict) -> None:
    cid = _current_chat_id()
    if cid is None:
        return
    _sync_await(save_session(cid, updates))


# ── Lazy LangGraph imports (so the bot still boots without these deps) ──────

_LC_READY = False

ChatGoogleGenerativeAI = None  # type: ignore[assignment]
HumanMessage           = None  # type: ignore[assignment]
SystemMessage          = None  # type: ignore[assignment]
AIMessage              = None  # type: ignore[assignment]
create_react_agent     = None  # type: ignore[assignment]
tool                   = None  # type: ignore[assignment]


def _ensure_imports() -> None:
    global _LC_READY, ChatGoogleGenerativeAI, HumanMessage, SystemMessage, AIMessage  # noqa: PLW0603
    global create_react_agent, tool  # noqa: PLW0603
    if _LC_READY:
        return
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as _Gemini
        from langchain_core.messages import (
            HumanMessage as _H,
            SystemMessage as _S,
            AIMessage as _A,
        )
        from langchain_core.tools import tool as _tool
        from langgraph.prebuilt import create_react_agent as _agent
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Dzukku ReAct agent stack not installed. "
            "Run: pip install -r requirements.txt (langchain, langgraph, langchain-google-genai)"
        ) from e
    ChatGoogleGenerativeAI = _Gemini
    HumanMessage           = _H
    SystemMessage          = _S
    AIMessage              = _A
    create_react_agent     = _agent
    tool                   = _tool
    _LC_READY = True


# ── Tool implementations (stateful via the contextvars above) ────────────────
# These mirror the branches in legacy orchestrator.execute_tool, just split
# into one function each and decorated with @tool.

def _build_tools() -> list:
    _ensure_imports()

    @tool
    def get_menu(filter_type: str = "All", filter_category: str = "") -> list[dict]:
        """
        Fetch the Dzukku Restaurant menu from the PostgreSQL database.
        Each item has: item_no, item_name, description, type ("Veg" or "Non-Veg"), category,
        price (INR), available ("Yes"/"No"), and optional special_price.

        Args:
            filter_type:     "All" (default), "Veg", or "Non-Veg".
            filter_category: optional category name (e.g. "Main Course", "Beverage").

        ALWAYS call this before quoting any item or price. Never fabricate menu data.
        """
        items = _sync_await(get_menu_items())
        if filter_type and filter_type != "All":
            items = [i for i in items if i.get("type") == filter_type]
        if filter_category:
            items = [i for i in items if (i.get("category") or "").lower() == filter_category.lower()]
        return items

    @tool
    def add_to_cart(items: list[dict]) -> dict:
        """
        Add one or more items to the customer's cart. Call this whenever the customer
        says they want to order something. Each item must reference an item_name that
        exists in the get_menu output (case-insensitive substring match is also OK).

        Args:
            items: list of {"item_name": str, "qty": int (default 1)}.

        Returns: {"added": [..], "not_found": [..], "cart_total": <inr>}.
        """
        sess     = _session()
        cart     = list(sess.get("cart", []) or [])
        menu     = _sync_await(get_menu_items())
        menu_map = {i["item_name"].lower(): i for i in menu}
        added, not_found = [], []

        for req in items or []:
            name = (req.get("item_name") or "").strip()
            qty  = int(req.get("qty", 1) or 1)
            if not name:
                continue
            match = menu_map.get(name.lower())
            if not match:
                for k, v in menu_map.items():
                    if name.lower() in k or k in name.lower():
                        match = v
                        break
            if not match:
                not_found.append(name)
                continue
            if match.get("available") != "Yes":
                not_found.append(f"{name} (unavailable)")
                continue
            existing = next((c for c in cart if c["item_name"] == match["item_name"]), None)
            if existing:
                existing["qty"] += qty
            else:
                cart.append({
                    "item_name": match["item_name"],
                    "qty":       qty,
                    "price":     float(match.get("price") or 0),
                    "type":      match.get("type", ""),
                })
            added.append(f"{qty}x {match['item_name']} (₹{int(float(match.get('price') or 0) * qty)})")

        _persist({"cart": cart})
        return {
            "added":      added,
            "not_found":  not_found,
            "cart_total": sum(c["qty"] * c["price"] for c in cart),
        }

    @tool
    def view_cart() -> dict:
        """Return the current cart contents and running total. Always call this before showing an order summary."""
        cart  = _session().get("cart", []) or []
        total = sum(c["qty"] * c["price"] for c in cart)
        return {"items": cart, "total": total, "count": len(cart)}

    @tool
    def clear_cart() -> dict:
        """Remove all items from the customer's cart. Call when the customer says cancel/start over/empty cart."""
        _persist({"cart": []})
        return {"cleared": True}

    @tool
    def update_customer_info(customer_name: str = "", customer_phone: str = "") -> dict:
        """
        Persist the customer's name and/or phone to the session. Call this whenever the
        user supplies a name, a phone, or both. MUST be called before place_order or
        make_reservation if these fields aren't already on file.
        """
        name  = (customer_name  or "").strip()
        phone = (customer_phone or "").strip()
        if not (name or phone):
            return {"error": "Provide at least customer_name or customer_phone."}
        updates: dict[str, Any] = {}
        if name:  updates["customer_name"]  = name
        if phone: updates["customer_phone"] = phone
        _persist(updates)
        return {"saved": updates}

    @tool
    def place_order(customer_name: str = "", customer_phone: str = "") -> dict:
        """
        Finalise and place the customer's order. ONLY call after:
          1. The cart has at least one item (use view_cart to verify).
          2. customer_name and customer_phone are known (call update_customer_info first).
          3. The customer has explicitly confirmed they want to place this order.

        Persists the order to PostgreSQL. The order's `platform` field is tagged
        with the user's chosen platform from the welcome prompt (Telegram by default).
        """
        sess  = _session()
        cart  = sess.get("cart", []) or []
        if not cart:
            return {"error": "Cart is empty. Add items first."}

        name  = (customer_name  or sess.get("customer_name", "")).strip()
        phone = (customer_phone or sess.get("customer_phone", "")).strip()
        if not name or not phone:
            return {"error": "Missing customer info. Ask for the missing field, then call update_customer_info, then retry place_order."}

        total = sum(c["qty"] * c["price"] for c in cart)
        platform_choice = (sess.get("ordering_platform") or "").strip()
        platform_tag    = platform_choice if platform_choice in ("Zomato", "Swiggy") else "Telegram"

        order_ref = _sync_await(save_order(name, phone, cart, total, platform=platform_tag))

        _persist({
            "cart":           [],
            "state":          "completed",
            "customer_name":  name,
            "customer_phone": phone,
        })

        items_summary = "\n".join(f"  {c['qty']}x {c['item_name']} — ₹{int(c['qty'] * c['price'])}" for c in cart)
        return {
            "order_ref":     order_ref,
            "customer_name": name,
            "total":         total,
            "items_summary": items_summary,
            "eta":           "20-30 mins",
            "platform":      platform_tag,
        }

    @tool
    def make_reservation(
        customer_name:   str,
        customer_phone:  str,
        date:            str,
        time:            str,
        guests:          int,
        special_request: str = "",
    ) -> dict:
        """
        Create a table reservation. Call after collecting all required fields and the
        customer has confirmed. Persists to PostgreSQL.

        Args:
            date: e.g. "2026-05-10"
            time: e.g. "7:30 PM"
            guests: integer headcount
            special_request: optional free-form note (allergy, occasion, seating, etc.)
        """
        guests_i = int(guests)
        res_ref  = _sync_await(save_reservation(customer_name, customer_phone, date, time, guests_i, special_request or ""))

        _persist({
            "state":          "res_confirmed",
            "customer_name":  customer_name,
            "customer_phone": customer_phone,
        })
        return {
            "reservation_ref": res_ref,
            "customer_name":   customer_name,
            "date":            date,
            "time":            time,
            "guests":          guests_i,
            "special_request": special_request or "",
        }

    @tool
    def get_restaurant_info() -> dict:
        """Return static restaurant info (timings, location, delivery, contact, cuisine)."""
        return {
            "name":        settings.RESTAURANT_NAME,
            "tagline":     settings.RESTAURANT_TAGLINE,
            "timings":     settings.RESTAURANT_TIMINGS,
            "location":    settings.RESTAURANT_LOCATION,
            "delivery":    settings.RESTAURANT_DELIVERY,
            "cuisine":     settings.RESTAURANT_CUISINE,
            "reservation": "Book directly through this chat",
        }

    return [
        get_menu,
        add_to_cart,
        view_cart,
        clear_cart,
        update_customer_info,
        place_order,
        make_reservation,
        get_restaurant_info,
    ]


# ── Agent build / cache ──────────────────────────────────────────────────────

_agent_cache: Any = None
_build_lock: asyncio.Lock | None = None  # lazy-init: see _get_build_lock


def _get_build_lock() -> asyncio.Lock:
    """Lazily create the lock the first time async code uses it.
    Avoids 'no current event loop' errors on Python 3.9 at module-load time."""
    global _build_lock  # noqa: PLW0603
    if _build_lock is None:
        _build_lock = asyncio.Lock()
    return _build_lock


async def _get_agent() -> Any:
    global _agent_cache  # noqa: PLW0603
    if _agent_cache is not None:
        return _agent_cache
    async with _get_build_lock():
        if _agent_cache is not None:
            return _agent_cache
        _ensure_imports()
        tools = _build_tools()
        api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set; cannot build Dzukku ReAct agent.")

        def _llm(model: str) -> Any:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.4,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
            )

        # Fallback chain: 2.5-flash → 2.0-flash → 1.5-flash
        llm = _llm(settings.GEMINI_PRIMARY).with_fallbacks([
            _llm(settings.GEMINI_FALLBACK),
            _llm(settings.GEMINI_FALLBACK_2),
        ])
        _agent_cache = create_react_agent(llm, tools)
        logger.info("Dzukku ReAct agent built with %d tools.", len(tools))
        return _agent_cache


# ── System prompt (mirrors legacy orchestrator's voice + guardrails) ─────────

def _system_prompt(user_name: str, session: dict) -> str:
    now        = datetime.now()
    hour       = now.hour
    time_label = (
        "morning"    if 6  <= hour < 11 else
        "lunch time" if 11 <= hour < 15 else
        "snack time" if 15 <= hour < 18 else
        "dinner time"if 18 <= hour < 23 else
        "late night"
    )
    customer_name  = session.get("customer_name") or ""
    customer_phone = session.get("customer_phone") or ""
    cart           = session.get("cart", []) or []
    cart_lines = "\n".join(
        f"  {c['qty']}x {c['item_name']} — ₹{int(c['qty']*c['price'])}"
        for c in cart
    ) or "  (empty)"
    cart_total = sum(c["qty"] * c["price"] for c in cart)
    known_name  = customer_name  or "(not collected yet)"
    known_phone = customer_phone or "(not collected yet)"

    return f"""You are Dzukku — the warm, witty restaurant assistant for Dzukku Restaurant ("Where every bite hits different ❤️"). You are running as a ReAct agent: you can call local tools (get_menu, add_to_cart, view_cart, place_order, make_reservation, etc.) and you reason between tool calls.

CONTEXT
- Time: {time_label} ({now.strftime('%I:%M %p')})
- Customer first name: {user_name or "there"}
- Customer name on file:  {known_name}
- Customer phone on file: {known_phone}
- Cart:
{cart_lines}
- Cart total: ₹{int(cart_total)}

HARD RULES
- NEVER fabricate menu items, prices, or availability. ALWAYS call `get_menu` before quoting anything from the menu.
- ALWAYS call `add_to_cart` when the customer says they want something.
- ALWAYS call `view_cart` before showing an order summary.
- ALWAYS call `update_customer_info` the moment you learn a name or a phone number.
- NEVER call `place_order` without explicit confirmation ("yes", "place it", "go ahead") AND both name + phone on file.
- After `place_order`, narrate the order ID, items, total, and ETA from the tool result.
- If a value on file is "(not collected yet)", you MUST ask for it before placing an order.
- Bare answers like "Krishna" or "9999999999" should be treated as the answer to your previous question.

TONE
- Warm, witty, concise — like texting a friend who runs an amazing restaurant.
- 2–4 lines for chat. Long messages only for menu listings or order summaries.
- Mirror the user's language (English, Telugu+English, Hindi+English).
- One gentle upsell per conversation, not pushy.

CONVERSATION FLOWS
Ordering:
  add_to_cart → ask if they want anything else → view_cart → ask for missing name/phone → update_customer_info → confirm summary → place_order → narrate confirmation.
Reservation:
  Ask date → time → guests → name (if not known) → phone → optional special request → confirm → make_reservation.
Off-topic:
  "Ha! I'm a food-only expert 😄 Try me on Menu, Orders, or Reservations!"
"""


# ── Public entry point ──────────────────────────────────────────────────────

async def get_dzukku_response(user_message: str, chat_id: int, user_name: str = "") -> str | None:
    """
    Run a single ReAct turn for the in-house Dzukku flow.
    Returns the assistant text, or None if the agent can't be built (so the
    caller can fall back to the legacy orchestrator).
    """
    try:
        agent = await _get_agent()
    except Exception as e:
        logger.error("Dzukku ReAct agent build failed: %s", e, exc_info=True)
        return None

    _ensure_imports()

    sess    = await get_session(chat_id)
    history = (sess.get("history") or [])[-8:]

    messages: list = [SystemMessage(content=_system_prompt(user_name, sess))]
    for turn in history:
        role    = turn.get("role")
        content = turn.get("content") or ""
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    # Set per-turn context so tools can read/write the right session
    tok_chat = _chat_ctx.set(chat_id)
    tok_user = _user_name_ctx.set(user_name or "")
    try:
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": messages}),
                timeout=60,
            )
        except asyncio.TimeoutError:
            logger.warning("Dzukku ReAct agent: timeout for chat_id=%s", chat_id)
            return "Hmm, that took a moment 🐢 — could you try again?"
        except Exception as e:
            logger.error("Dzukku ReAct agent: invoke failed: %s", e, exc_info=True)
            return None

        # Extract final assistant text
        out_messages = result.get("messages") if isinstance(result, dict) else None
        final_text   = ""
        if out_messages:
            for m in reversed(out_messages):
                content = getattr(m, "content", None)
                mtype   = getattr(m, "type", "") or m.__class__.__name__
                if isinstance(content, str) and content.strip() and mtype.lower() in ("ai", "aimessage", "assistant"):
                    final_text = content.strip()
                    break
            if not final_text:
                last = out_messages[-1]
                final_text = (getattr(last, "content", "") or "").strip()

        if not final_text:
            final_text = "Got it! 😊 What would you like to do next?"

        # Persist rolling history (cap last 16)
        new_history = list(sess.get("history") or [])
        new_history.append({"role": "user",      "content": user_message})
        new_history.append({"role": "assistant", "content": final_text})
        new_history = new_history[-16:]
        await save_session(chat_id, {"history": new_history})

        return final_text
    finally:
        _chat_ctx.reset(tok_chat)
        _user_name_ctx.reset(tok_user)
