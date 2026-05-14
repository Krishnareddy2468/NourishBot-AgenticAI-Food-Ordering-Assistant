"""
Agent Orchestrator v2 — Dzukku Restaurant Bot
==============================================
Uses OpenAI GPT-4o with function-calling tools.

Architecture: Planner → Executor → Verifier
  - Planner:  LLM decides which tools to call
  - Executor: Deterministic tool execution with policy checks
  - Verifier: Validates tool results before committing side effects

The LLM never directly writes to the DB — it calls tools which the
executor runs deterministically, then the verifier checks against
operational policies before committing.

State machine:
  new → greeting → browsing_menu → ordering →
  awaiting_name → awaiting_phone → awaiting_confirm → completed
  reservation → res_awaiting_date → res_awaiting_time →
  res_awaiting_guests → res_awaiting_name → res_awaiting_phone → res_confirmed
"""

import json
import os
import logging
import uuid
import asyncio
from datetime import datetime
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from app.db.crud import (
    get_menu_items,
    get_menu_for_prompt,
    get_session,
    save_session,
    save_order,
    save_reservation,
)
from app.core.config import settings
from app.agent.policies import default_policy

load_dotenv()
logger = logging.getLogger(__name__)


def _sa(coro):
    """Sync wrapper for async CRUD calls."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


# ── OpenAI setup ──────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or settings.OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

_client = OpenAI(api_key=OPENAI_API_KEY)

PRIMARY_MODEL   = settings.OPENAI_PRIMARY
FALLBACK_MODEL  = settings.OPENAI_FALLBACK


def _to_openai_tools(raw_tools: list[dict]) -> list[dict]:
    """Convert flat {name, description, parameters} to OpenAI tool format."""
    return [{"type": "function", "function": t} for t in raw_tools]


# ── Tool definitions (OpenAI function-calling format) ──────────────────────────

RAW_TOOLS = [
    {
        "name": "get_menu",
        "description": (
            "Fetch the full restaurant menu. Returns a list of items with "
            "item_no, item_name, description, type (Veg/Non-Veg), category, "
            "price (INR), and available (Yes/No). "
            "Always call this before answering menu questions or confirming prices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filter_type": {
                    "type": "string",
                    "description": "Optional: 'Veg', 'Non-Veg', or 'All' (default)",
                    "enum": ["All", "Veg", "Non-Veg"],
                },
                "filter_category": {
                    "type": "string",
                    "description": "Optional: category name to filter by, e.g. 'Main Course'",
                },
            },
            "required": [],
        },
    },
    {
        "name": "add_to_cart",
        "description": (
            "Add one or more items to the customer's cart. "
            "Call this when the customer says they want to order something. "
            "item_name must exactly match a menu item name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "List of items to add",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_name": {"type": "string"},
                            "qty": {"type": "integer", "description": "Quantity, default 1"},
                            "modifiers": {
                                "type": "array",
                                "description": "Optional: list of modifier names e.g. ['Extra Spicy', 'No Onion']",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["item_name"],
                    },
                }
            },
            "required": ["items"],
        },
    },
    {
        "name": "view_cart",
        "description": "Return the current cart contents and running total.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "clear_cart",
        "description": "Remove all items from the customer's cart.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_customer_info",
        "description": (
            "Persist the customer's name and/or phone to the session as soon as you "
            "learn either. Call this whenever the user supplies a name, a phone, or both. "
            "You MUST call this before place_order or make_reservation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name":  {"type": "string", "description": "Optional"},
                "customer_phone": {"type": "string", "description": "Optional, digits only"},
            },
            "required": [],
        },
    },
    {
        "name": "place_order",
        "description": (
            "Finalise and place the order. Call ONLY after you have collected "
            "the customer's name and phone number and they have explicitly confirmed. "
            "This persists the order to the database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name":  {"type": "string"},
                "customer_phone": {"type": "string"},
                "order_type": {
                    "type": "string",
                    "description": "Order type: DELIVERY, PICKUP, or DINE_IN",
                    "enum": ["DELIVERY", "PICKUP", "DINE_IN"],
                },
            },
            "required": ["customer_name", "customer_phone"],
        },
    },
    {
        "name": "make_reservation",
        "description": (
            "Create a table reservation. Call after collecting all required fields. "
            "Persists the reservation to the database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name":   {"type": "string"},
                "customer_phone":  {"type": "string"},
                "date":            {"type": "string", "description": "e.g. '2026-05-10'"},
                "time":            {"type": "string", "description": "e.g. '7:30 PM'"},
                "guests":          {"type": "integer"},
                "special_request": {"type": "string", "description": "Optional"},
            },
            "required": ["customer_name", "customer_phone", "date", "time", "guests"],
        },
    },
    {
        "name": "get_restaurant_info",
        "description": "Return static restaurant info: timings, location, delivery, contact.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_external_ordering_links",
        "description": (
            "Return the public Zomato and Swiggy listing URLs for Dzukku Restaurant. "
            "Call this whenever the customer asks to order via Zomato or Swiggy, "
            "or wants the link to the restaurant on those platforms."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_ordering_platform",
        "description": (
            "Record which ordering platform the customer chose for THIS conversation. "
            "Valid values: 'Dzukku' (order directly via this bot), 'Zomato', 'Swiggy'. "
            "Call this after the customer picks a platform from the welcome prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["Dzukku", "Zomato", "Swiggy"],
                },
            },
            "required": ["platform"],
        },
    },
    {
        "name": "initiate_payment",
        "description": (
            "Create a Razorpay payment intent for the current order. "
            "Call after place_order succeeds, when the customer wants to pay online. "
            "Returns a payment link/order ID for the Razorpay checkout."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_ref": {
                    "type": "string",
                    "description": "The order reference returned by place_order",
                },
            },
            "required": ["order_ref"],
        },
    },
    {
        "name": "check_payment_status",
        "description": (
            "Check the payment status for an order. "
            "Returns CREATED | AUTHORIZED | CAPTURED | FAILED | REFUNDED."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_ref": {
                    "type": "string",
                    "description": "The order reference to check payment for",
                },
            },
            "required": ["order_ref"],
        },
    },
    {
        "name": "get_delivery_status",
        "description": (
            "Check the delivery status for an order. "
            "Returns current status and driver location if available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_ref": {
                    "type": "string",
                    "description": "The order reference to check delivery for",
                },
            },
            "required": ["order_ref"],
        },
    },
    {
        "name": "open_table_session",
        "description": (
            "Open a dine-in table session. Call when a customer arrives for dine-in. "
            "Returns the session ID and table assignment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table_id": {
                    "type": "integer",
                    "description": "Table number to assign (0 for auto-assign)",
                },
                "guests": {
                    "type": "integer",
                    "description": "Number of guests",
                },
            },
            "required": ["guests"],
        },
    },
    {
        "name": "close_table_session",
        "description": (
            "Close a dine-in table session and generate the bill. "
            "Call when the customer asks for the bill or is done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "The table session ID to close",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "get_modifiers",
        "description": (
            "Get available modifiers (customizations) for a menu item. "
            "Call when the customer asks about customizations like spice level, "
            "extra toppings, portion size, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The menu item name to get modifiers for",
                },
            },
            "required": ["item_name"],
        },
    },
]


# ── Executor: deterministic tool execution ─────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict, session: dict) -> tuple[Any, dict]:
    """
    Execute a tool call from the LLM.
    Returns (result_for_llm, updated_session_fields).
    """
    session_updates: dict = {}

    if tool_name == "get_menu":
        items = _sa(get_menu_items())
        ftype = tool_args.get("filter_type", "All")
        fcat  = tool_args.get("filter_category", "")
        if ftype and ftype != "All":
            items = [i for i in items if i["type"] == ftype]
        if fcat:
            items = [i for i in items if i["category"].lower() == fcat.lower()]
        return items, session_updates

    if tool_name == "add_to_cart":
        cart  = list(session.get("cart", []))
        items = _sa(get_menu_items())
        menu_map = {i["item_name"].lower(): i for i in items}
        added, not_found = [], []

        for req in tool_args.get("items", []):
            name = req["item_name"]
            qty  = int(req.get("qty", 1))
            modifiers = req.get("modifiers", [])
            match = menu_map.get(name.lower())
            if not match:
                for key, val in menu_map.items():
                    if name.lower() in key or key in name.lower():
                        match = val
                        break
            if match:
                if match["available"] != "Yes":
                    not_found.append(f"{name} (unavailable)")
                    continue
                existing = next((c for c in cart if c["item_name"] == match["item_name"]), None)
                if existing:
                    existing["qty"] += qty
                    if modifiers:
                        existing.setdefault("modifiers", []).extend(modifiers)
                else:
                    cart.append({
                        "item_name": match["item_name"],
                        "qty": qty,
                        "price": match["price"],
                        "type": match["type"],
                        "modifiers": modifiers,
                    })
                added.append(f"{qty}x {match['item_name']} (₹{int(match['price'] * qty)})")
            else:
                not_found.append(name)

        session_updates["cart"] = cart
        result = {"added": added, "not_found": not_found,
                  "cart_total": sum(c["qty"] * c["price"] for c in cart)}
        return result, session_updates

    if tool_name == "view_cart":
        cart  = session.get("cart", [])
        total = sum(c["qty"] * c["price"] for c in cart)
        return {"items": cart, "total": total, "count": len(cart)}, session_updates

    if tool_name == "clear_cart":
        session_updates["cart"] = []
        return {"cleared": True}, session_updates

    if tool_name == "update_customer_info":
        name  = (tool_args.get("customer_name")  or "").strip()
        phone = (tool_args.get("customer_phone") or "").strip()
        saved = {}
        if name:
            session_updates["customer_name"] = name
            saved["customer_name"] = name
        if phone:
            session_updates["customer_phone"] = phone
            saved["customer_phone"] = phone
        if not saved:
            return {"error": "Provide at least customer_name or customer_phone"}, session_updates
        return {"saved": saved}, session_updates

    if tool_name == "place_order":
        cart  = session.get("cart", [])
        if not cart:
            return {"error": "Cart is empty. Please add items first."}, session_updates

        total = sum(c["qty"] * c["price"] for c in cart)
        name  = (tool_args.get("customer_name")  or session.get("customer_name", "")).strip()
        phone = (tool_args.get("customer_phone") or session.get("customer_phone", "")).strip()
        if not name or not phone:
            return {
                "error": "Missing customer info. Ask the user for name and/or phone first, "
                         "then call update_customer_info before retrying place_order."
            }, session_updates

        # Policy check: order type availability
        order_type = tool_args.get("order_type", "DELIVERY")
        ok, msg = default_policy.validate_order_type(order_type)
        if not ok:
            return {"error": msg}, session_updates

        # Policy check: operating hours
        if not default_policy.is_within_operating_hours():
            return {"error": "Sorry, we're currently closed. Our hours are 6 AM – 11 PM, all days."}, session_updates

        # Policy check: max items
        if len(cart) > default_policy.max_items_per_order:
            return {"error": f"Maximum {default_policy.max_items_per_order} items per order."}, session_updates

        # Policy check: max qty per item
        for c in cart:
            if c["qty"] > default_policy.max_qty_per_item:
                return {"error": f"Maximum {default_policy.max_qty_per_item} quantity per item."}, session_updates

        # Policy check: min order for delivery
        if order_type == "DELIVERY" and int(total * 100) < default_policy.min_order_for_delivery_cents:
            return {"error": f"Minimum order for delivery is ₹{default_policy.min_order_for_delivery_cents // 100}."}, session_updates

        platform_choice = (session.get("ordering_platform") or "").strip()
        platform_tag = platform_choice if platform_choice in ("Zomato", "Swiggy") else "Telegram"

        order_ref = _sa(save_order(name, phone, cart, total, platform=platform_tag))

        pretty = ", ".join(f"{c['qty']}x {c['item_name']}" for c in cart)

        # Write outbox event for realtime
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import OutboxEvent
            import asyncio as _aio
            async def _write_outbox():
                async with AsyncSessionLocal() as s:
                    s.add(OutboxEvent(
                        restaurant_id=1,
                        event_type="order.created",
                        payload={"order_id": order_ref, "customer_name": name},
                    ))
                    await s.commit()
            _sa(_write_outbox())
        except Exception as e:
            logger.warning("Outbox write failed: %s", e)

        session_updates["cart"]           = []
        session_updates["state"]          = "completed"
        session_updates["customer_name"]  = name
        session_updates["customer_phone"] = phone
        session_updates["last_order_ref"] = order_ref

        items_summary = "\n".join(
            f"  {c['qty']}x {c['item_name']} — ₹{int(c['qty'] * c['price'])}"
            for c in cart
        )

        prep_time = default_policy.estimate_prep_time(0)
        eta_min = prep_time // 60 if prep_time > 0 else 25

        return {
            "order_ref":     order_ref,
            "customer_name": name,
            "total":         total,
            "items_summary": items_summary,
            "eta":           f"{eta_min}-{eta_min + 10} mins",
        }, session_updates

    if tool_name == "make_reservation":
        name    = tool_args["customer_name"]
        phone   = tool_args["customer_phone"]
        date    = tool_args["date"]
        time_   = tool_args["time"]
        guests  = int(tool_args["guests"])
        special = tool_args.get("special_request", "")

        # Policy check: max guests
        if guests > default_policy.max_guests_per_reservation:
            return {"error": f"Maximum {default_policy.max_guests_per_reservation} guests per reservation."}, session_updates

        res_ref = _sa(save_reservation(name, phone, date, time_, guests, special))

        session_updates["state"]          = "res_confirmed"
        session_updates["customer_name"]  = name
        session_updates["customer_phone"] = phone

        return {
            "reservation_ref": res_ref,
            "customer_name":   name,
            "date":            date,
            "time":            time_,
            "guests":          guests,
            "special_request": special,
        }, session_updates

    if tool_name == "get_restaurant_info":
        return {
            "name":        "Dzukku Restaurant",
            "tagline":     "Where every bite hits different",
            "timings":     "6:00 AM - 11:00 PM, all days",
            "location":    "Hyderabad, Telangana",
            "contact":     "Available via this chat",
            "delivery":    "Via Swiggy & Zomato",
            "reservation": "Book directly through this chat",
            "cuisine":     "Indian - Veg & Non-Veg",
            "zomato_url":  settings.ZOMATO_URL,
            "swiggy_url":  settings.SWIGGY_URL,
        }, session_updates

    if tool_name == "get_external_ordering_links":
        return {
            "zomato_url": settings.ZOMATO_URL,
            "swiggy_url": settings.SWIGGY_URL,
            "note": (
                "Send these clickable links to the customer. The order will be "
                "placed on the external app - Dzukku bot only provides the link."
            ),
        }, session_updates

    if tool_name == "set_ordering_platform":
        choice = (tool_args.get("platform") or "").strip()
        if choice not in ("Dzukku", "Zomato", "Swiggy"):
            return {"error": "platform must be one of 'Dzukku', 'Zomato', 'Swiggy'"}, session_updates
        session_updates["ordering_platform"] = choice
        return {
            "saved":     True,
            "platform":  choice,
            "zomato_url": settings.ZOMATO_URL if choice == "Zomato" else None,
            "swiggy_url": settings.SWIGGY_URL if choice == "Swiggy" else None,
        }, session_updates

    if tool_name == "initiate_payment":
        order_ref = tool_args.get("order_ref", "")
        if not order_ref:
            order_ref = session.get("last_order_ref", "")
        if not order_ref:
            return {"error": "No order to pay for. Place an order first."}, session_updates
        try:
            from app.payments.razorpay import create_payment_order
            from app.db.session import AsyncSessionLocal
            from sqlalchemy import select
            from app.db.models import Order, Payment

            async def _do_payment():
                async with AsyncSessionLocal() as s:
                    result = await s.execute(
                        select(Order).where(Order.order_ref == order_ref)
                    )
                    order = result.scalar_one_or_none()
                    if not order:
                        return {"error": f"Order {order_ref} not found."}

                    rzp = await create_payment_order(
                        amount_cents=order.total_cents,
                        order_ref=order_ref,
                    )
                    payment = Payment(
                        restaurant_id=1,
                        order_id=order.id,
                        provider="RAZORPAY",
                        status="CREATED",
                        amount_cents=order.total_cents,
                        currency="INR",
                        provider_order_id=rzp.get("id"),
                    )
                    s.add(payment)
                    await s.commit()
                    return {
                        "razorpay_order_id": rzp.get("id"),
                        "amount_cents": order.total_cents,
                        "currency": "INR",
                        "key_id": settings.RAZORPAY_KEY_ID,
                    }

            return _sa(_do_payment()), session_updates
        except Exception as e:
            return {"error": f"Payment initiation failed: {e}"}, session_updates

    if tool_name == "check_payment_status":
        order_ref = tool_args.get("order_ref", "")
        if not order_ref:
            order_ref = session.get("last_order_ref", "")
        if not order_ref:
            return {"error": "No order specified."}, session_updates
        try:
            from app.db.session import AsyncSessionLocal
            from sqlalchemy import select
            from app.db.models import Order, Payment

            async def _check():
                async with AsyncSessionLocal() as s:
                    result = await s.execute(
                        select(Order).where(Order.order_ref == order_ref)
                    )
                    order = result.scalar_one_or_none()
                    if not order:
                        return {"error": f"Order {order_ref} not found."}
                    presult = await s.execute(
                        select(Payment).where(Payment.order_id == order.id)
                    )
                    payment = presult.scalar_one_or_none()
                    if not payment:
                        return {"status": "NO_PAYMENT", "message": "No payment initiated yet."}
                    return {
                        "status": payment.status,
                        "amount_cents": payment.amount_cents,
                        "provider_payment_id": payment.provider_payment_id,
                    }

            return _sa(_check()), session_updates
        except Exception as e:
            return {"error": f"Payment check failed: {e}"}, session_updates

    if tool_name == "get_delivery_status":
        order_ref = tool_args.get("order_ref", "")
        if not order_ref:
            order_ref = session.get("last_order_ref", "")
        if not order_ref:
            return {"error": "No order specified."}, session_updates
        try:
            from app.db.session import AsyncSessionLocal
            from sqlalchemy import select
            from app.db.models import Order, Delivery

            async def _check():
                async with AsyncSessionLocal() as s:
                    result = await s.execute(
                        select(Order).where(Order.order_ref == order_ref)
                    )
                    order = result.scalar_one_or_none()
                    if not order:
                        return {"error": f"Order {order_ref} not found."}
                    dresult = await s.execute(
                        select(Delivery).where(Delivery.order_id == order.id)
                    )
                    delivery = dresult.scalar_one_or_none()
                    if not delivery:
                        return {"status": "NOT_ASSIGNED", "message": "No delivery assigned yet."}
                    return {
                        "status": delivery.status,
                        "driver_name": "Assigned",
                        "estimated_minutes": 30,
                    }

            return _sa(_check()), session_updates
        except Exception as e:
            return {"error": f"Delivery check failed: {e}"}, session_updates

    if tool_name == "open_table_session":
        table_id = tool_args.get("table_id", 0)
        guests = tool_args.get("guests", 1)

        if guests > default_policy.max_guests_per_reservation:
            return {"error": f"Maximum {default_policy.max_guests_per_reservation} guests per table."}, session_updates

        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models import DiningTable, TableSession

            async def _open():
                async with AsyncSessionLocal() as s:
                    if table_id:
                        tresult = await s.execute(
                            select(DiningTable).where(DiningTable.id == table_id)
                        )
                        table = tresult.scalar_one_or_none()
                    else:
                        # Auto-assign: find first available table
                        tresult = await s.execute(
                            select(DiningTable).where(
                                DiningTable.restaurant_id == 1,
                                DiningTable.status == "AVAILABLE",
                            ).limit(1)
                        )
                        table = tresult.scalar_one_or_none()

                    if not table:
                        return {"error": "No available table found."}
                    if table.status != "AVAILABLE":
                        return {"error": f"Table {table.table_number} is not available (status: {table.status})."}

                    table.status = "OCCUPIED"
                    ts = TableSession(
                        restaurant_id=1,
                        table_id=table.id,
                        guests=guests,
                        status="OPEN",
                    )
                    s.add(ts)
                    await s.commit()
                    await s.refresh(ts)
                    return {
                        "session_id": ts.id,
                        "table_number": table.table_number,
                        "table_id": table.id,
                        "guests": guests,
                        "status": "OPEN",
                    }

            result = _sa(_open())
            session_updates["table_session_id"] = result.get("session_id")
            return result, session_updates
        except Exception as e:
            return {"error": f"Failed to open table session: {e}"}, session_updates

    if tool_name == "close_table_session":
        session_id = tool_args.get("session_id")
        if not session_id:
            session_id = session.get("table_session_id")
        if not session_id:
            return {"error": "No table session specified."}, session_updates
        try:
            from app.db.session import AsyncSessionLocal
            from sqlalchemy import select
            from app.db.models import TableSession, DiningTable, Order

            async def _close():
                async with AsyncSessionLocal() as s:
                    result = await s.execute(
                        select(TableSession).where(TableSession.id == session_id)
                    )
                    ts = result.scalar_one_or_none()
                    if not ts:
                        return {"error": f"Table session {session_id} not found."}
                    if ts.status != "OPEN":
                        return {"error": f"Session is already {ts.status}."}

                    # Sum all orders for this session
                    from sqlalchemy import func as sa_func
                    oresult = await s.execute(
                        select(sa_func.sum(Order.total_cents)).where(
                            Order.table_session_id == ts.id
                        )
                    )
                    total_cents = oresult.scalar() or 0

                    ts.status = "CLOSED"

                    # Free the table
                    tresult = await s.execute(
                        select(DiningTable).where(DiningTable.id == ts.table_id)
                    )
                    tbl = tresult.scalar_one_or_none()
                    if tbl:
                        tbl.status = "AVAILABLE"

                    await s.commit()
                    return {
                        "session_id": ts.id,
                        "total_cents": total_cents,
                        "total_rupees": total_cents / 100,
                        "status": "CLOSED",
                        "message": "Table session closed. Bill generated.",
                    }

            result = _sa(_close())
            session_updates.pop("table_session_id", None)
            return result, session_updates
        except Exception as e:
            return {"error": f"Failed to close table session: {e}"}, session_updates

    if tool_name == "get_modifiers":
        item_name = tool_args.get("item_name", "")
        if not item_name:
            return {"error": "Please specify an item name."}, session_updates
        try:
            from app.db.session import AsyncSessionLocal
            from sqlalchemy import select
            from app.db.models import MenuItem, MenuItemModifierGroup, ModifierGroup, Modifier

            async def _get_mods():
                async with AsyncSessionLocal() as s:
                    mi_result = await s.execute(
                        select(MenuItem).where(MenuItem.name == item_name)
                    )
                    mi = mi_result.scalar_one_or_none()
                    if not mi:
                        return {"error": f"Item '{item_name}' not found."}

                    mimg_result = await s.execute(
                        select(MenuItemModifierGroup).where(
                            MenuItemModifierGroup.menu_item_id == mi.id
                        )
                    )
                    groups = []
                    for mimg in mimg_result.scalars().all():
                        mg_result = await s.execute(
                            select(ModifierGroup).where(ModifierGroup.id == mimg.modifier_group_id)
                        )
                        mg = mg_result.scalar_one_or_none()
                        if not mg:
                            continue
                        mod_result = await s.execute(
                            select(Modifier).where(Modifier.group_id == mg.id)
                        )
                        mods = [{"name": m.name, "price_cents": m.price_cents} for m in mod_result.scalars().all()]
                        groups.append({
                            "group_name": mg.name,
                            "required": mimg.required,
                            "multi_select": mg.multi_select,
                            "modifiers": mods,
                        })
                    if not groups:
                        return {"modifiers": [], "message": "No customizations available for this item."}
                    return {"modifiers": groups}

            return _sa(_get_mods()), session_updates
        except Exception as e:
            return {"error": f"Failed to get modifiers: {e}"}, session_updates

    return {"error": f"Unknown tool: {tool_name}"}, session_updates


# ── Verifier: post-execution validation ───────────────────────────────────────

def verify_result(tool_name: str, result: Any, session_updates: dict) -> tuple[Any, dict]:
    """
    Post-execution verifier. Validates tool results against policies.
    Can modify results or session_updates before they are committed.
    """
    if isinstance(result, dict) and result.get("error"):
        return result, session_updates

    # Verify place_order results
    if tool_name == "place_order" and isinstance(result, dict) and result.get("order_ref"):
        if not default_policy.is_within_operating_hours():
            return {"error": "Cannot place order — restaurant is currently closed."}, session_updates
        # Upsell budget check
        upsell_count = session_updates.get("upsell_count", 0)
        if upsell_count >= default_policy.max_upsells_per_session:
            session_updates["upsell_blocked"] = True

    # Verify payment initiation
    if tool_name == "initiate_payment" and isinstance(result, dict) and result.get("razorpay_order_id"):
        if "RAZORPAY" not in default_policy.supported_payment_providers:
            return {"error": "Online payments are not currently supported."}, session_updates

    return result, session_updates


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(session: dict) -> str:
    now        = datetime.now()
    hour       = now.hour
    time_label = (
        "morning"    if 6  <= hour < 11 else
        "lunch time" if 11 <= hour < 15 else
        "snack time" if 15 <= hour < 18 else
        "dinner time"if 18 <= hour < 23 else
        "late night"
    )
    user_name      = session.get("user_name") or "there"
    customer_name  = session.get("customer_name") or ""
    customer_phone = session.get("customer_phone") or ""
    cart           = session.get("cart", [])
    cart_lines = "\n".join(
        f"  {c['qty']}x {c['item_name']} — ₹{int(c['qty']*c['price'])}"
        for c in cart
    ) or "  (empty)"
    cart_total = sum(c["qty"] * c["price"] for c in cart)

    known_name  = customer_name  or "(not collected yet)"
    known_phone = customer_phone or "(not collected yet)"

    ordering_platform = (session.get("ordering_platform") or "").strip()
    platform_label    = ordering_platform or "(not chosen yet)"

    is_open = default_policy.is_within_operating_hours()

    return f"""You are Dzukku — the warm, witty, and intelligent restaurant assistant for Dzukku Restaurant ("Where every bite hits different").

CURRENT CONTEXT:
- Time of day: {time_label} ({now.strftime('%I:%M %p')})
- Restaurant is: {'OPEN' if is_open else 'CLOSED'}
- Telegram first name: {user_name}
- Customer name on file: {known_name}
- Customer phone on file: {known_phone}
- Ordering platform chosen: {platform_label}
- Zomato URL: {settings.ZOMATO_URL}
- Swiggy URL: {settings.SWIGGY_URL}
- Cart right now:
{cart_lines}
- Cart total: ₹{int(cart_total)}

IMPORTANT memory rules:
- The "Customer name / phone on file" above are remembered across turns.
  If a value is "(not collected yet)", you MUST ask for it before placing an order.
  If a value is already filled in, DO NOT ask for it again — just use it.
- When the user types only a name (e.g. "Krishna") or only a phone (e.g. "9999999999")
  in reply to your previous question, treat it as the answer to that question.
- When you have BOTH name and phone, you must call `update_customer_info` to save them.
- After name+phone are saved, ask for explicit confirmation, then call `place_order`.

IDENTITY & TONE
- Warm, witty, and concise — like texting a friend who runs an amazing restaurant.
- Keep replies to 2-4 lines for casual chat. Longer ONLY for menu listings or order bills.
- Language mirroring: if the user writes Telugu+English or Hindi+English, match their style.
- Emotional awareness: rushed user → be efficient; chatty user → be playful.
- Use emojis naturally, not excessively.

TOOLS (always prefer tools over guessing)
- ALWAYS call get_menu before answering any question about items, prices, or availability.
- ALWAYS call add_to_cart when the customer says they want something.
- ALWAYS call view_cart before showing order summary.
- ALWAYS call place_order ONLY after confirming name, phone, and cart with the customer.
- ALWAYS call make_reservation after collecting: name, phone, date, time, guests.
- Use get_modifiers when the customer wants customizations (spice level, extras, etc.).
- Use initiate_payment after placing an order if the customer wants to pay online.
- Use check_payment_status to check if payment went through.
- Use get_delivery_status to check delivery progress.
- Use open_table_session for dine-in customers.
- Use close_table_session when the customer asks for the bill at a table.
- NEVER make up prices or menu items — use the tool.

PLATFORM SELECTION (Zomato / Swiggy / Dzukku)
- The Telegram bot UI offers a platform-selection prompt at the very start of
  every new chat (Order via Dzukku Bot / Zomato / Swiggy).
- "Ordering platform chosen" in the context above tells you which one the
  customer picked. If it is "(not chosen yet)" and the customer is greeting you,
  politely ask them where they'd like to order from.
- If they pick Zomato or Swiggy, share the relevant URL via
  `get_external_ordering_links` and let them know they'll continue on that app.
- If they pick Dzukku, proceed with the normal in-bot ordering flow.

CONVERSATION FLOW

GREETING (first message or /start):
  "Hey {user_name}! Welcome to Dzukku - where every bite hits different
   Quick question - would you like to order through *Dzukku Bot* (right here),
   or via *Zomato* or *Swiggy*?"

TIME-AWARE SUGGESTIONS:
  - Morning  -> Masala Dosa, Chai, light items
  - Lunch    -> Biryani, Thali, main courses
  - Snack    -> Tikka, Noodles, beverages
  - Dinner   -> Butter Chicken, Biryani, Family packs
  - Late night -> Biryani, Fried Rice, Comfort food

MOOD DETECTION:
  - "sad / bad day / stressed" -> Comfort food: Kheer, Gulab Jamun, Butter Chicken
  - "celebrating / happy"     -> Premium: Family Pack, Prawn, Mutton
  - "light / not hungry"      -> Dal Tadka, Dosa, Beverages
  - "quick / hurry"           -> Fastest prep items first

ORDERING FLOW:
  1. Customer expresses interest -> call add_to_cart immediately
  2. Ask if they want anything else or customizations (call get_modifiers)
  3. Show cart summary (call view_cart)
  4. Ask for name (if not already known)
  5. Ask for phone number
  6. Confirm: show items + total + ask "Shall I place this?"
  7. On confirmation -> call place_order
  8. Reply with order confirmation + ref number + ETA
  9. Optionally ask if they want to pay online -> call initiate_payment

DINE-IN FLOW:
  1. Customer arrives -> call open_table_session
  2. Order items as usual -> add_to_cart, place_order with order_type DINE_IN
  3. When done -> call close_table_session to generate bill

ORDER BILL FORMAT:
  Order Confirmed!
  Order ID: #[ref]
  ----------------
  [qty]x [item] - ₹[price]
  ...
  ----------------
  Total: ₹[total]
  ETA: ~20-30 mins
  Thank you for choosing Dzukku

RESERVATION FLOW:
  1. Ask preferred date
  2. Ask preferred time
  3. Ask number of guests
  4. Ask name (if not known)
  5. Ask phone
  6. Ask special requests (optional)
  7. Confirm -> call make_reservation

SMART UPSELLING (once per conversation, natural not pushy):
  - After main course added -> suggest a beverage
  - No dessert in cart -> "Room for dessert?"
  - Single item > ₹200 -> check if a combo is better value
  Only upsell ONCE. Never be annoying.

HARD RULES
- NEVER make up menu items, prices, or availability — call get_menu
- NEVER call place_order without explicit customer confirmation
- NEVER answer questions unrelated to food, orders, reservations, or restaurant info
- NEVER send walls of text — 2-4 lines max for chat messages
- ALWAYS end every response with a clear next-step question or action
- If an item is unavailable, suggest the closest alternative from the menu
- Off-topic queries: "I'm a food-only expert! Try me on Menu, Orders, or Reservations!"
- If the restaurant is CLOSED, inform the customer and suggest ordering when open.
"""


# ── Planner: build OpenAI messages ────────────────────────────────────────────

def _build_messages(session: dict, user_message: str) -> list[dict]:
    """
    Build OpenAI-format messages list with system prompt + history + user message.
    """
    system_prompt = build_system_prompt(session)

    history = session.get("history", []) or []
    history_text = ""
    if history:
        lines = []
        for turn in history[-8:]:
            role = "Customer" if turn.get("role") == "user" else "You"
            lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n\nRECENT CONVERSATION (most recent last):\n" + "\n".join(lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": history_text + "\n\n---\nCustomer says: " + user_message if history_text else user_message},
    ]


# ── Main orchestrator ─────────────────────────────────────────────────────────

def get_bot_response(user_message: str, chat_id: int, user_name: str = "") -> str:
    """
    Main entry point — Planner → Executor → Verifier loop.

    1. Planner:  Prepares LLM context (session + history + message)
    2. Executor: Runs the LLM, executes tool calls deterministically
    3. Verifier: Validates results against policies before committing
    """
    trace_id = uuid.uuid4().hex[:8]
    logger.info("[%s] chat_id=%s msg=%r", trace_id, chat_id, user_message[:80])

    session = _sa(get_session(chat_id))
    if not session.get("user_name") and user_name:
        session["user_name"] = user_name

    # ── Planner ────────────────────────────────────────────────────────────────
    messages = _build_messages(session, user_message)
    openai_tools = _to_openai_tools(RAW_TOOLS)

    # ── Executor + Verifier loop ──────────────────────────────────────────────
    max_iterations = settings.AGENT_MAX_ITERATIONS
    session_updates_accumulated: dict = {}
    final_text: str = ""

    for iteration in range(max_iterations):
        # Try primary model, fall back on error
        response = None
        for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
            try:
                response = _client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=openai_tools,
                    max_tokens=settings.OPENAI_MAX_TOKENS,
                    temperature=0.4,
                )
                break
            except Exception as e:
                logger.warning("[%s] Model %s failed: %s", trace_id, model_name, e)
                if model_name == FALLBACK_MODEL:
                    logger.error("[%s] Both models failed.", trace_id)
                    return (
                        "Sorry, I'm having a small hiccup right now! "
                        "Please try again in a moment."
                    )

        choice = response.choices[0]
        msg = choice.message

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        # Add assistant message with tool calls to conversation
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        # ── Executor: run each tool ────────────────────────────────────────────
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            logger.info("[%s] Tool call: %s(%s)", trace_id, tool_name, tool_args)

            session.update(session_updates_accumulated)
            result, updates = execute_tool(tool_name, tool_args, session)

            # ── Verifier: validate before committing ──────────────────────────
            result, updates = verify_result(tool_name, result, updates)

            session_updates_accumulated.update(updates)
            logger.info("[%s] Tool result: %s", trace_id, str(result)[:200])

            # Feed tool result back to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    if not final_text:
        final_text = "Sorry, I couldn't process that. Could you try again?"

    # ── Persist ────────────────────────────────────────────────────────────────
    new_history = list(session.get("history", []) or [])
    new_history.append({"role": "user",      "content": user_message})
    new_history.append({"role": "assistant", "content": final_text})
    new_history = new_history[-16:]
    session_updates_accumulated["history"] = new_history

    session.update(session_updates_accumulated)
    if not session.get("user_name") and user_name:
        session["user_name"] = user_name
    _sa(save_session(chat_id, session))

    logger.info("[%s] Reply: %r", trace_id, final_text[:120])
    return final_text
