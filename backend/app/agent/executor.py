"""
Executor — Stage 3 of the Dzukku Pipeline.

Deterministic tool runner. The LLM proposes actions; the executor:
  1. Validates every input (schema, ranges, policy)
  2. Commits side effects to DB
  3. Returns ExecutionResult — no LLM involvement

Hard rules:
  - Prices are ALWAYS read from MenuItem.price_cents — never from planner args.
  - Cart mutations go through Cart / CartItem models.
  - place_order re-reads menu prices; never trusts cart totals passed in.
  - Policy (operating hours, kitchen capacity, cancellation window) checked before every commit.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.db.models import (
    Channel, Session as SessionModel,
    Customer, Cart, CartItem, MenuItem, MenuItemImage,
    ModifierGroup, Modifier, MenuItemModifierGroup,
    Order, OrderItem, Reservation,
    Payment, DiningTable, TableSession, TableSessionOrder,
    Driver, Delivery, Invoice,
)
from app.agent.context_builder import ContextSnapshot, save_session_meta
from app.agent.policies import default_policy
from app.agent.persona import find_alternatives, validate_delivery_address

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    tool: str
    ok: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    committed: list[ActionResult] = field(default_factory=list)
    rejected: list[ActionResult] = field(default_factory=list)
    # Updated state snapshots (for verifier)
    updated_cart: list[dict] = field(default_factory=list)
    cart_total_cents: int = 0
    cart_id: Optional[int] = None
    order_ref: Optional[str] = None
    reservation_ref: Optional[str] = None
    tracking_info: Optional[dict] = None
    menu_items: list[dict] = field(default_factory=list)
    info_payload: Optional[dict] = None
    eta_seconds: Optional[int] = None

    @property
    def all_ok(self) -> bool:
        return len(self.rejected) == 0 and len(self.committed) > 0

    @property
    def error_messages(self) -> list[str]:
        return [r.error for r in self.rejected if r.error]


# ── Executor ──────────────────────────────────────────────────────────────────

class Executor:

    @classmethod
    async def run(
        cls,
        proposed_actions: list[dict],
        ctx: ContextSnapshot,
    ) -> ExecutionResult:
        """Run proposed actions sequentially. Stop on hard errors."""
        result = ExecutionResult()

        for action in proposed_actions:
            tool = action.get("tool", "")
            args = action.get("args") or {}
            handler = _TOOL_REGISTRY.get(tool)

            if not handler:
                result.rejected.append(ActionResult(tool=tool, ok=False, error=f"Unknown tool: {tool}"))
                continue

            try:
                action_result = await handler(args, ctx, result)
                if action_result.ok:
                    result.committed.append(action_result)
                else:
                    result.rejected.append(action_result)
                    # Stop pipeline on hard failures (order / reservation)
                    if tool in ("place_order", "cancel_order", "make_reservation"):
                        break
            except Exception as e:
                logger.error("Tool %s raised: %s", tool, e, exc_info=True)
                result.rejected.append(ActionResult(tool=tool, ok=False, error=str(e)))
                break

        return result


# ── Tool implementations ──────────────────────────────────────────────────────

async def _tool_add_to_cart(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    items_requested = args.get("items") or []
    if not items_requested:
        return ActionResult("add_to_cart", False, error="No items specified.")

    added, not_found, unavailable = [], [], []

    async with AsyncSessionLocal() as db:
        cart = await _get_or_create_cart(db, ctx)

        for req in items_requested:
            name = str(req.get("name") or "").strip()
            qty = max(1, min(int(req.get("qty") or 1), ctx.policy.max_qty_per_item))
            if not name:
                continue

            mi = await _find_menu_item(db, ctx.restaurant_id, name)
            if not mi:
                not_found.append(name)
                continue
            if not mi.available:
                alts = await find_alternatives(
                    mi.name, ctx.restaurant_id,
                    veg_only=(mi.type in ("VEG", "VEGAN")),
                    price_hint_cents=mi.price_cents,
                )
                unavailable.append({"name": mi.name, "alternatives": alts})
                continue

            existing = (await db.execute(
                select(CartItem).where(
                    CartItem.cart_id == cart.id,
                    CartItem.item_id == mi.id,
                )
            )).scalar_one_or_none()

            if existing:
                existing.qty = min(existing.qty + qty, ctx.policy.max_qty_per_item)
            else:
                db.add(CartItem(
                    restaurant_id=ctx.restaurant_id,
                    cart_id=cart.id,
                    item_id=mi.id,
                    qty=qty,
                    unit_price_cents=mi.price_cents,
                ))
            added.append({"name": mi.name, "qty": qty, "unit_price_cents": mi.price_cents})

        await db.commit()
        await _refresh_cart_snapshot(db, cart.id, result)

    if not added and not_found:
        return ActionResult("add_to_cart", False, error=f"Items not found: {', '.join(not_found)}")

    if not added and unavailable:
        # All items were unavailable — surface alternatives
        all_alts = []
        for u in unavailable:
            all_alts.extend(u.get("alternatives") or [])
        return ActionResult("add_to_cart", False, data={
            "unavailable": unavailable,
            "alternatives": all_alts[:4],
        }, error=f"{unavailable[0]['name']} is currently unavailable.")

    return ActionResult("add_to_cart", True, data={
        "added": added,
        "not_found": not_found,
        "unavailable": unavailable,
        "cart_total_cents": result.cart_total_cents,
    })


async def _tool_remove_from_cart(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    item_name = str(args.get("item_name") or "").strip()
    if not item_name:
        return ActionResult("remove_from_cart", False, error="item_name required.")

    if not ctx.cart_id:
        return ActionResult("remove_from_cart", False, error="Cart is empty.")

    async with AsyncSessionLocal() as db:
        mi = await _find_menu_item(db, ctx.restaurant_id, item_name)
        if not mi:
            return ActionResult("remove_from_cart", False, error=f"'{item_name}' not found in menu.")

        ci = (await db.execute(
            select(CartItem).where(CartItem.cart_id == ctx.cart_id, CartItem.item_id == mi.id)
        )).scalar_one_or_none()

        if not ci:
            return ActionResult("remove_from_cart", False, error=f"'{mi.name}' not in cart.")

        await db.delete(ci)
        await db.commit()
        await _refresh_cart_snapshot(db, ctx.cart_id, result)

    return ActionResult("remove_from_cart", True, data={"removed": mi.name, "cart_total_cents": result.cart_total_cents})


async def _tool_clear_cart(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    if not ctx.cart_id:
        return ActionResult("clear_cart", True, data={"cleared": True})

    async with AsyncSessionLocal() as db:
        items = (await db.execute(
            select(CartItem).where(CartItem.cart_id == ctx.cart_id)
        )).scalars().all()
        for ci in items:
            await db.delete(ci)
        await db.commit()

    result.updated_cart = []
    result.cart_total_cents = 0
    return ActionResult("clear_cart", True, data={"cleared": True})


async def _tool_view_cart(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    # Re-read from DB to return fresh state
    if ctx.cart_id:
        async with AsyncSessionLocal() as db:
            await _refresh_cart_snapshot(db, ctx.cart_id, result)
    return ActionResult("view_cart", True, data={
        "items": result.updated_cart or ctx.cart,
        "total_cents": result.cart_total_cents or ctx.cart_total_cents,
    })


async def _tool_update_customer(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    name = str(args.get("name") or "").strip()
    phone = str(args.get("phone") or "").strip()
    address = str(args.get("address") or "").strip()
    language = str(args.get("language") or "").strip()

    if not any([name, phone, address, language]):
        return ActionResult("update_customer", False, error="At least one field required.")

    # Basic phone validation
    if phone and not (phone.isdigit() and 8 <= len(phone) <= 15):
        return ActionResult("update_customer", False, error=f"Invalid phone number: '{phone}'")

    async with AsyncSessionLocal() as db:
        # Find or create channel
        ch = await _get_or_create_channel(db, ctx.chat_id)

        # Find or create customer
        customer = None
        if ch.customer_id:
            customer = (await db.execute(
                select(Customer).where(Customer.id == ch.customer_id)
            )).scalar_one_or_none()

        if not customer and phone:
            customer = (await db.execute(
                select(Customer).where(
                    Customer.restaurant_id == ctx.restaurant_id,
                    Customer.phone == phone,
                )
            )).scalar_one_or_none()

        if not customer:
            customer = Customer(
                restaurant_id=ctx.restaurant_id,
                name=name or ctx.user_name,
                phone=phone or "UNKNOWN",
                language_pref=language or "en",
            )
            db.add(customer)
            await db.flush()

        # Apply updates
        if name:     customer.name = name
        if phone:    customer.phone = phone
        if language: customer.language_pref = language

        # Link channel → customer
        if not ch.customer_id:
            ch.customer_id = customer.id

        await db.commit()

    saved = {k: v for k, v in {"name": name, "phone": phone, "address": address, "language": language}.items() if v}
    return ActionResult("update_customer", True, data={"saved": saved})


async def _tool_place_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    order_type = str(args.get("order_type") or "DELIVERY").upper()
    address = str(args.get("address") or "").strip()
    notes = str(args.get("notes") or "").strip()

    # ── Policy checks ──────────────────────────────────────────────────────
    if not ctx.is_open:
        return ActionResult("place_order", False, error="Restaurant is currently closed.")

    valid, msg = ctx.policy.validate_order_type(order_type)
    if not valid:
        return ActionResult("place_order", False, error=msg)

    if order_type == "DELIVERY" and not address:
        return ActionResult("place_order", False, error="Delivery address is required.")

    if order_type == "DELIVERY" and address:
        deliverable, reason = validate_delivery_address(address, ctx.policy)
        if not deliverable:
            return ActionResult("place_order", False,
                                error=reason, data={"radius_exceeded": True})

    if not ctx.customer_name or not ctx.customer_phone:
        return ActionResult("place_order", False, error="Customer name and phone are required before placing an order.")

    cart = result.updated_cart or ctx.cart
    if not cart:
        return ActionResult("place_order", False, error="Cart is empty. Add items first.")

    eta = ctx.policy.estimate_prep_time(ctx.kitchen_load)
    if eta < 0:
        return ActionResult("place_order", False, error="Kitchen is at capacity. Please try again in a few minutes.")

    if len(cart) > ctx.policy.max_items_per_order:
        return ActionResult("place_order", False,
                            error=f"Order exceeds maximum {ctx.policy.max_items_per_order} items.")

    async with AsyncSessionLocal() as db:
        # Re-read item prices from DB — never trust in-memory cart prices
        order_ref = f"DZK-{uuid.uuid4().hex[:6].upper()}"
        subtotal = 0
        validated_items = []

        for ci in cart:
            mi = (await db.execute(
                select(MenuItem).where(MenuItem.id == ci["item_id"])
            )).scalar_one_or_none()

            if not mi or not mi.available:
                return ActionResult(
                    "place_order", False,
                    error=f"'{ci['item_name']}' is no longer available. Please update your cart.",
                )
            price = mi.price_cents
            subtotal += price * ci["qty"]
            validated_items.append({
                "item_id": mi.id,
                "item_name_snapshot": mi.name,
                "qty": ci["qty"],
                "unit_price_cents": price,
            })

        # Delivery minimum check
        if order_type == "DELIVERY" and subtotal < ctx.policy.min_order_for_delivery_cents:
            min_inr = ctx.policy.min_order_for_delivery_cents // 100
            return ActionResult(
                "place_order", False,
                error=f"Minimum order for delivery is ₹{min_inr}. Current total: ₹{subtotal // 100}.",
            )

        # Find channel for FK
        ch = await _get_or_create_channel(db, ctx.chat_id)

        order = Order(
            restaurant_id=ctx.restaurant_id,
            order_ref=order_ref,
            customer_id=ctx.customer_id,
            channel_id=ch.id,
            order_type=order_type,
            status="CREATED",
            subtotal_cents=subtotal,
            total_cents=subtotal,
            notes=notes or None,
            idempotency_key=f"{ctx.customer_phone}-{order_ref}",
        )
        db.add(order)
        await db.flush()

        for item in validated_items:
            db.add(OrderItem(
                restaurant_id=ctx.restaurant_id,
                order_id=order.id,
                item_id=item["item_id"],
                item_name_snapshot=item["item_name_snapshot"],
                qty=item["qty"],
                unit_price_cents=item["unit_price_cents"],
                status="PENDING",
            ))

        # Mark cart as converted
        if ctx.cart_id:
            cart_obj = (await db.execute(select(Cart).where(Cart.id == ctx.cart_id))).scalar_one_or_none()
            if cart_obj:
                cart_obj.status = "CONVERTED"

        await db.commit()

        # Clear cart snapshot in result
        result.updated_cart = []
        result.cart_total_cents = 0
        result.order_ref = order_ref

    logger.info("Order placed: %s for customer %s", order_ref, ctx.customer_phone)

    # ── Update taste vector after order placement (Phase 1) ──────────────────
    if ctx.customer_id:
        try:
            from app.agent.memory_agent import update_taste_vector
            item_data = [
                {"item_name": ci["item_name"], "qty": ci["qty"],
                 "price_cents": ci["unit_price_cents"]}
                for ci in cart
            ]
            import asyncio as _asyncio
            _asyncio.create_task(update_taste_vector(ctx.customer_id, item_data))
        except Exception as _e:
            logger.debug("Taste vector update skipped: %s", _e)

    # ── Auto-chain: create Razorpay payment intent for DELIVERY/PICKUP orders ──
    payment_data = {}
    razorpay_configured = bool(settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET)
    if order_type in ("DELIVERY", "PICKUP") and razorpay_configured:
        try:
            payment_result = await _tool_create_payment_intent(
                {"order_ref": order_ref}, ctx, result
            )
            if payment_result.ok:
                payment_data = payment_result.data
        except Exception as e:
            logger.warning("Auto payment intent failed for %s: %s", order_ref, e)
    elif order_type in ("DELIVERY", "PICKUP") and not razorpay_configured:
        logger.info("Razorpay not configured — order %s confirmed as COD.", order_ref)

    return ActionResult("place_order", True, data={
        "order_ref": order_ref,
        "order_type": order_type,
        "subtotal_cents": subtotal,
        "eta_seconds": eta,
        "items": validated_items,
        "payment_required": bool(payment_data),
        "payment_url": payment_data.get("razorpay_order_id", ""),
    })


async def _tool_track_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    order_ref = str(args.get("order_ref") or "").strip().upper()

    # Try active orders from context first
    if not order_ref and ctx.active_orders:
        order_ref = ctx.active_orders[0]["order_ref"]

    if not order_ref:
        return ActionResult("track_order", False, error="Please provide an order reference (e.g. DZK-ABC123).")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()

        if not order:
            return ActionResult("track_order", False, error=f"Order {order_ref} not found.")

        info = {
            "order_ref": order.order_ref,
            "status": order.status,
            "total_cents": order.total_cents,
            "order_type": order.order_type,
            "created_at": order.created_at.isoformat() if order.created_at else "",
        }
        result.tracking_info = info

    return ActionResult("track_order", True, data=info)


async def _tool_cancel_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    order_ref = str(args.get("order_ref") or "").strip().upper()
    if not order_ref:
        return ActionResult("cancel_order", False, error="Order reference required.")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()

        if not order:
            return ActionResult("cancel_order", False, error=f"Order {order_ref} not found.")

        if not ctx.policy.can_cancel_order(order.status):
            return ActionResult(
                "cancel_order", False,
                error=f"Order {order_ref} is already {order.status} and cannot be cancelled. "
                      f"Call us directly to arrange changes.",
            )

        order.status = "CANCELLED"
        await db.commit()

        # Broadcast real-time event
        try:
            from app.realtime.events import order_status_changed
            from app.realtime.ws_manager import ws_manager
            evt = order_status_changed(order.id, order_ref, "CANCELLED")
            await ws_manager.broadcast(order.restaurant_id, evt.to_dict())
        except Exception as e:
            logger.warning("WS broadcast failed for cancel: %s", e)

    return ActionResult("cancel_order", True, data={"order_ref": order_ref, "status": "CANCELLED"})


async def _tool_make_reservation(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    date_str = str(args.get("date") or "").strip()
    time_str = str(args.get("time") or "").strip()
    guests = int(args.get("guests") or 0)
    special = str(args.get("special_request") or "").strip()

    # Validate
    if not date_str or not time_str:
        return ActionResult("make_reservation", False, error="Date and time are required.")
    if guests <= 0:
        return ActionResult("make_reservation", False, error="Number of guests must be at least 1.")
    if guests > ctx.policy.max_guests_per_reservation:
        return ActionResult(
            "make_reservation", False,
            error=f"Maximum {ctx.policy.max_guests_per_reservation} guests per reservation.",
        )
    if not ctx.customer_name or not ctx.customer_phone:
        return ActionResult("make_reservation", False, error="Customer name and phone required.")

    # Parse date to check it's in the future
    try:
        from datetime import date as date_cls
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if parsed_date < datetime.now(IST).date():
            return ActionResult("make_reservation", False, error="Reservation date cannot be in the past.")
    except ValueError:
        pass  # allow non-standard formats; DB stores as text

    async with AsyncSessionLocal() as db:
        # Ensure customer exists
        customer = None
        if ctx.customer_id:
            customer = (await db.execute(
                select(Customer).where(Customer.id == ctx.customer_id)
            )).scalar_one_or_none()

        if not customer:
            customer = (await db.execute(
                select(Customer).where(
                    Customer.restaurant_id == ctx.restaurant_id,
                    Customer.phone == ctx.customer_phone,
                )
            )).scalar_one_or_none()

        if not customer:
            customer = Customer(
                restaurant_id=ctx.restaurant_id,
                name=ctx.customer_name,
                phone=ctx.customer_phone,
            )
            db.add(customer)
            await db.flush()

        res_ref = f"RSV-{uuid.uuid4().hex[:6].upper()}"
        reservation = Reservation(
            restaurant_id=ctx.restaurant_id,
            reservation_ref=res_ref,
            customer_id=customer.id,
            date=parsed_date if "parsed_date" in dir() else date_str,
            time=time_str,
            guests=guests,
            special_request=special,
            status="CREATED",
        )
        db.add(reservation)
        await db.commit()
        result.reservation_ref = res_ref

    return ActionResult("make_reservation", True, data={
        "reservation_ref": res_ref,
        "date": date_str,
        "time": time_str,
        "guests": guests,
        "special_request": special,
    })


async def _tool_get_menu(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    filter_type = str(args.get("filter_type") or "All").strip()
    filter_category = str(args.get("filter_category") or "").strip().lower()

    items = ctx.menu_snapshot
    if filter_type and filter_type != "All":
        items = [i for i in items if (i.get("type") or "").upper() == filter_type.upper()]
    if filter_category:
        items = [i for i in items if filter_category in (i.get("description") or "").lower()
                 or filter_category in i["name"].lower()]

    result.menu_items = items
    return ActionResult("get_menu", True, data={"items": items, "count": len(items)})


async def _tool_get_kitchen_eta(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    eta = ctx.policy.estimate_prep_time(ctx.kitchen_load)
    result.eta_seconds = eta
    if eta < 0:
        return ActionResult("get_kitchen_eta", True, data={"message": "Kitchen is at capacity right now.", "eta_seconds": -1})
    return ActionResult("get_kitchen_eta", True, data={"eta_seconds": eta, "eta_minutes": eta // 60})


async def _tool_get_restaurant_info(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    from app.core.config import settings as s
    info = {
        "name": s.RESTAURANT_NAME,
        "tagline": s.RESTAURANT_TAGLINE,
        "timings": s.RESTAURANT_TIMINGS,
        "location": s.RESTAURANT_LOCATION,
        "delivery": s.RESTAURANT_DELIVERY,
        "cuisine": s.RESTAURANT_CUISINE,
        "is_open": ctx.is_open,
        "reservation": "Book directly through this chat",
    }
    result.info_payload = info
    return ActionResult("get_restaurant_info", True, data=info)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cart / order-type tools
# ══════════════════════════════════════════════════════════════════════════════

async def _tool_update_cart_item(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Change the quantity of an item already in the cart. qty=0 removes it."""
    item_name = str(args.get("item_name") or "").strip()
    qty = int(args.get("qty") or 0)
    if not item_name:
        return ActionResult("update_cart_item", False, error="item_name required.")
    if not ctx.cart_id:
        return ActionResult("update_cart_item", False, error="Cart is empty.")

    async with AsyncSessionLocal() as db:
        mi = await _find_menu_item(db, ctx.restaurant_id, item_name)
        if not mi:
            return ActionResult("update_cart_item", False, error=f"'{item_name}' not found in menu.")

        ci = (await db.execute(
            select(CartItem).where(CartItem.cart_id == ctx.cart_id, CartItem.item_id == mi.id)
        )).scalar_one_or_none()

        if not ci:
            return ActionResult("update_cart_item", False, error=f"'{mi.name}' is not in your cart.")

        if qty <= 0:
            await db.delete(ci)
        else:
            ci.qty = min(qty, ctx.policy.max_qty_per_item)
        await db.commit()
        await _refresh_cart_snapshot(db, ctx.cart_id, result)

    return ActionResult("update_cart_item", True, data={
        "item": mi.name, "qty": qty,
        "cart_total_cents": result.cart_total_cents,
    })


async def _tool_set_order_type(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Persist the chosen order type (DELIVERY/PICKUP/DINE_IN) to session."""
    order_type = str(args.get("order_type") or "").strip().upper()
    if order_type not in ("DELIVERY", "PICKUP", "DINE_IN"):
        return ActionResult("set_order_type", False,
                            error=f"Invalid order_type '{order_type}'. Use DELIVERY, PICKUP, or DINE_IN.")

    valid, msg = ctx.policy.validate_order_type(order_type)
    if not valid:
        return ActionResult("set_order_type", False, error=msg)

    await save_session_meta(ctx.chat_id, {"order_type": order_type})
    return ActionResult("set_order_type", True, data={"order_type": order_type})


async def _tool_set_delivery_address(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Save the customer's delivery address to session — validates delivery radius."""
    address = str(args.get("address") or "").strip()
    if not address or len(address) < 10:
        return ActionResult("set_delivery_address", False,
                            error="Please provide a complete delivery address (street, area, city).")

    deliverable, reason = validate_delivery_address(address, ctx.policy)
    if not deliverable:
        return ActionResult("set_delivery_address", False,
                            error=reason, data={"radius_exceeded": True})

    await save_session_meta(ctx.chat_id, {"delivery_address": address})
    return ActionResult("set_delivery_address", True, data={"address": address})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Payment tools
# ══════════════════════════════════════════════════════════════════════════════

async def _tool_create_payment_intent(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Create a Razorpay payment order for an existing order."""
    order_ref = str(args.get("order_ref") or ctx.pending_order_ref or "").strip().upper()
    provider = str(args.get("provider") or "razorpay").lower()

    if not order_ref:
        return ActionResult("create_payment_intent", False, error="order_ref required.")
    if provider != "razorpay":
        return ActionResult("create_payment_intent", False,
                            error=f"Provider '{provider}' not supported. Use 'razorpay'.")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()
        if not order:
            return ActionResult("create_payment_intent", False, error=f"Order {order_ref} not found.")

        # Idempotency: return existing pending payment if present
        existing_payment = (await db.execute(
            select(Payment).where(
                Payment.order_id == order.id,
                Payment.status == "CREATED",
            )
        )).scalar_one_or_none()

        if existing_payment and existing_payment.provider_order_id:
            await save_session_meta(ctx.chat_id, {
                "pending_payment_order_id": existing_payment.provider_order_id,
                "pending_order_ref": order_ref,
            })
            return ActionResult("create_payment_intent", True, data={
                "payment_id": existing_payment.id,
                "razorpay_order_id": existing_payment.provider_order_id,
                "amount_cents": existing_payment.amount_cents,
                "order_ref": order_ref,
                "key_id": "",
            })

        try:
            from app.payments.razorpay import create_payment_order
            rzp_order = await create_payment_order(
                amount_cents=order.total_cents,
                order_ref=order_ref,
                receipt=order_ref[:40],
            )
        except Exception as e:
            logger.error("Razorpay error for %s: %s", order_ref, e)
            return ActionResult("create_payment_intent", False,
                                error=f"Payment gateway error: {e}")

        from app.core.config import settings as s
        payment = Payment(
            restaurant_id=ctx.restaurant_id,
            order_id=order.id,
            provider="RAZORPAY",
            status="CREATED",
            amount_cents=order.total_cents,
            currency="INR",
            provider_order_id=rzp_order.get("id"),
        )
        db.add(payment)
        await db.commit()
        await db.refresh(payment)

        await save_session_meta(ctx.chat_id, {
            "pending_payment_order_id": rzp_order.get("id"),
            "pending_order_ref": order_ref,
        })

        return ActionResult("create_payment_intent", True, data={
            "payment_id": payment.id,
            "razorpay_order_id": rzp_order.get("id"),
            "amount_cents": order.total_cents,
            "order_ref": order_ref,
            "key_id": s.RAZORPAY_KEY_ID,
            "payment_required": True,
        })


async def _tool_check_payment_status(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Check payment status for an order — queries DB first, then Razorpay if needed."""
    order_ref = str(args.get("order_ref") or ctx.pending_order_ref or "").strip().upper()
    payment_id = str(args.get("payment_id") or "").strip()

    if not order_ref and not payment_id:
        return ActionResult("check_payment_status", False,
                            error="order_ref or payment_id required.")

    async with AsyncSessionLocal() as db:
        # Find payment by order ref
        payment = None
        if order_ref:
            order = (await db.execute(
                select(Order).where(Order.order_ref == order_ref)
            )).scalar_one_or_none()
            if order:
                payment = (await db.execute(
                    select(Payment).where(Payment.order_id == order.id)
                    .order_by(Payment.id.desc())
                )).scalars().first()

        if not payment:
            return ActionResult("check_payment_status", False,
                                error=f"No payment found for {order_ref or payment_id}.")

        # If not yet captured, poll Razorpay
        if payment.status not in ("CAPTURED", "AUTHORIZED") and payment.provider_payment_id:
            try:
                from app.payments.razorpay import fetch_payment
                rzp_data = await fetch_payment(payment.provider_payment_id)
                rzp_status = rzp_data.get("status", "").upper()
                if rzp_status == "CAPTURED":
                    payment.status = "CAPTURED"
                    await db.commit()
            except Exception as e:
                logger.warning("Razorpay poll failed: %s", e)

        return ActionResult("check_payment_status", True, data={
            "payment_id": payment.id,
            "status": payment.status,
            "amount_cents": payment.amount_cents,
            "order_ref": order_ref,
            "provider_order_id": payment.provider_order_id,
        })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Dine-in tools
# ══════════════════════════════════════════════════════════════════════════════

async def _tool_open_table_session(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Open a dine-in table session (waiter-initiated)."""
    table_id = int(args.get("table_id") or 0)
    guests = int(args.get("guests") or 1)
    waiter_user_id = args.get("waiter_user_id")

    if not table_id:
        return ActionResult("open_table_session", False, error="table_id required.")
    if guests < 1:
        return ActionResult("open_table_session", False, error="guests must be >= 1.")

    async with AsyncSessionLocal() as db:
        table = (await db.execute(
            select(DiningTable).where(
                DiningTable.id == table_id,
                DiningTable.restaurant_id == ctx.restaurant_id,
                DiningTable.active == True,
            )
        )).scalar_one_or_none()
        if not table:
            return ActionResult("open_table_session", False,
                                error=f"Table {table_id} not found or inactive.")

        # Check no open session already
        existing = (await db.execute(
            select(TableSession).where(
                TableSession.table_id == table_id,
                TableSession.status == "OPEN",
            )
        )).scalar_one_or_none()
        if existing:
            return ActionResult("open_table_session", False,
                                error=f"Table {table.name} already has an open session (#{existing.id}).")

        ts = TableSession(
            restaurant_id=ctx.restaurant_id,
            table_id=table_id,
            waiter_user_id=waiter_user_id,
            guests=guests,
            status="OPEN",
            opened_at=datetime.now(timezone.utc),
        )
        db.add(ts)
        await db.commit()
        await db.refresh(ts)

        try:
            from app.realtime.events import table_session_opened
            from app.realtime.ws_manager import ws_manager
            await ws_manager.broadcast(ctx.restaurant_id,
                                       table_session_opened(ts.id, table_id).to_dict())
        except Exception:
            pass

    return ActionResult("open_table_session", True, data={
        "session_id": ts.id,
        "table_id": table_id,
        "table_name": table.name,
        "guests": guests,
        "status": "OPEN",
    })


async def _tool_add_table_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Add items to a dine-in table session (creates order + fires to kitchen)."""
    session_id = int(args.get("table_session_id") or 0)
    items_requested = args.get("items") or []

    if not session_id:
        return ActionResult("add_table_order", False, error="table_session_id required.")
    if not items_requested:
        return ActionResult("add_table_order", False, error="No items specified.")

    async with AsyncSessionLocal() as db:
        ts = (await db.execute(
            select(TableSession).where(TableSession.id == session_id)
        )).scalar_one_or_none()
        if not ts or ts.status != "OPEN":
            return ActionResult("add_table_order", False,
                                error=f"Table session {session_id} not found or not open.")

        import uuid as _uuid
        order_ref = f"DZK-{_uuid.uuid4().hex[:6].upper()}"
        total = 0
        validated = []

        for req in items_requested:
            name = str(req.get("name") or "").strip()
            qty = max(1, int(req.get("qty") or 1))
            mi = await _find_menu_item(db, ctx.restaurant_id, name)
            if not mi or not mi.available:
                continue
            total += mi.price_cents * qty
            validated.append({"item_id": mi.id, "name": mi.name,
                               "qty": qty, "unit_price_cents": mi.price_cents})

        if not validated:
            return ActionResult("add_table_order", False,
                                error="None of the requested items were found or available.")

        order = Order(
            restaurant_id=ctx.restaurant_id,
            order_ref=order_ref,
            customer_id=ctx.customer_id,
            order_type="DINE_IN",
            status="ACCEPTED",
            subtotal_cents=total,
            total_cents=total,
        )
        db.add(order)
        await db.flush()

        for item in validated:
            db.add(OrderItem(
                restaurant_id=ctx.restaurant_id,
                order_id=order.id,
                item_id=item["item_id"],
                item_name_snapshot=item["name"],
                qty=item["qty"],
                unit_price_cents=item["unit_price_cents"],
                status="PENDING",
            ))

        db.add(TableSessionOrder(
            restaurant_id=ctx.restaurant_id,
            table_session_id=session_id,
            order_id=order.id,
        ))
        await db.commit()

    return ActionResult("add_table_order", True, data={
        "order_ref": order_ref,
        "session_id": session_id,
        "items": validated,
        "total_cents": total,
    })


async def _tool_close_table_session(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Close a dine-in table session."""
    session_id = int(args.get("table_session_id") or 0)
    if not session_id:
        return ActionResult("close_table_session", False, error="table_session_id required.")

    async with AsyncSessionLocal() as db:
        ts = (await db.execute(
            select(TableSession).where(TableSession.id == session_id)
        )).scalar_one_or_none()
        if not ts:
            return ActionResult("close_table_session", False,
                                error=f"Session {session_id} not found.")
        if ts.status == "CLOSED":
            return ActionResult("close_table_session", True,
                                data={"session_id": session_id, "status": "CLOSED"})

        ts.status = "CLOSED"
        ts.closed_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            from app.realtime.events import table_session_closed
            from app.realtime.ws_manager import ws_manager
            await ws_manager.broadcast(ctx.restaurant_id,
                                       table_session_closed(session_id, ts.table_id).to_dict())
        except Exception:
            pass

    return ActionResult("close_table_session", True,
                        data={"session_id": session_id, "status": "CLOSED"})


async def _tool_generate_invoice(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Generate invoice for a table session or order."""
    entity_id = int(args.get("entity_id") or 0)
    entity_type = str(args.get("entity_type") or "TABLE_SESSION").upper()

    if not entity_id:
        return ActionResult("generate_invoice", False, error="entity_id required.")

    async with AsyncSessionLocal() as db:
        subtotal = 0

        if entity_type == "TABLE_SESSION":
            links = (await db.execute(
                select(TableSessionOrder).where(
                    TableSessionOrder.table_session_id == entity_id
                )
            )).scalars().all()
            for link in links:
                order = (await db.execute(
                    select(Order).where(Order.id == link.order_id)
                )).scalar_one_or_none()
                if order:
                    subtotal += order.total_cents
        elif entity_type == "ORDER":
            order = (await db.execute(
                select(Order).where(Order.id == entity_id)
            )).scalar_one_or_none()
            if order:
                subtotal = order.total_cents

        import uuid as _uuid
        invoice = Invoice(
            restaurant_id=ctx.restaurant_id,
            invoice_no=f"INV-{_uuid.uuid4().hex[:6].upper()}",
            entity_type=entity_type,
            entity_id=entity_id,
            subtotal_cents=subtotal,
            tax_cents=0,
            total_cents=subtotal,
        )
        db.add(invoice)

        # Auto-close session
        if entity_type == "TABLE_SESSION":
            ts = (await db.execute(
                select(TableSession).where(TableSession.id == entity_id)
            )).scalar_one_or_none()
            if ts and ts.status == "OPEN":
                ts.status = "CLOSED"
                ts.closed_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(invoice)

    return ActionResult("generate_invoice", True, data={
        "invoice_no": invoice.invoice_no,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "total_cents": subtotal,
    })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Restaurant ops tools
# ══════════════════════════════════════════════════════════════════════════════

async def _tool_set_item_availability(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Toggle menu item availability (restaurant staff tool)."""
    item_id = int(args.get("item_id") or 0)
    available = bool(args.get("available", True))
    item_name = str(args.get("item_name") or "").strip()

    async with AsyncSessionLocal() as db:
        if item_id:
            mi = (await db.execute(
                select(MenuItem).where(MenuItem.id == item_id)
            )).scalar_one_or_none()
        else:
            mi = await _find_menu_item(db, ctx.restaurant_id, item_name)

        if not mi:
            return ActionResult("set_item_availability", False,
                                error=f"Menu item not found.")

        mi.available = available
        await db.commit()

        try:
            from app.realtime.events import menu_item_availability_changed
            from app.realtime.ws_manager import ws_manager
            await ws_manager.broadcast(ctx.restaurant_id,
                                       menu_item_availability_changed(mi.id, available).to_dict())
        except Exception:
            pass

    return ActionResult("set_item_availability", True, data={
        "item_id": mi.id, "item_name": mi.name, "available": available,
    })


async def _tool_update_stock(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Update stock quantity for a menu item. delta can be negative."""
    item_id = int(args.get("item_id") or 0)
    item_name = str(args.get("item_name") or "").strip()
    delta = int(args.get("delta") or 0)
    set_to = args.get("set_to")

    async with AsyncSessionLocal() as db:
        if item_id:
            mi = (await db.execute(
                select(MenuItem).where(MenuItem.id == item_id)
            )).scalar_one_or_none()
        else:
            mi = await _find_menu_item(db, ctx.restaurant_id, item_name)

        if not mi:
            return ActionResult("update_stock", False, error="Menu item not found.")

        if set_to is not None:
            mi.stock_qty = max(0, int(set_to))
        else:
            current = mi.stock_qty or 0
            mi.stock_qty = max(0, current + delta)

        # Auto mark unavailable if stock hits 0
        if mi.stock_qty == 0:
            mi.available = False

        await db.commit()

    return ActionResult("update_stock", True, data={
        "item_id": mi.id, "item_name": mi.name,
        "stock_qty": mi.stock_qty, "available": mi.available,
    })


async def _tool_update_order_status(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Update order status (kitchen / ops staff tool)."""
    order_ref = str(args.get("order_ref") or "").strip().upper()
    order_id = int(args.get("order_id") or 0)
    status = str(args.get("status") or "").strip().upper()

    VALID_STATUSES = {"ACCEPTED", "PREPARING", "READY", "OUT_FOR_DELIVERY", "DELIVERED",
                      "COMPLETED", "CANCELLED"}
    if status not in VALID_STATUSES:
        return ActionResult("update_order_status", False,
                            error=f"Invalid status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}")

    async with AsyncSessionLocal() as db:
        if order_ref:
            order = (await db.execute(
                select(Order).where(Order.order_ref == order_ref)
            )).scalar_one_or_none()
        elif order_id:
            order = (await db.execute(
                select(Order).where(Order.id == order_id)
            )).scalar_one_or_none()
        else:
            return ActionResult("update_order_status", False,
                                error="order_ref or order_id required.")

        if not order:
            return ActionResult("update_order_status", False,
                                error=f"Order not found.")

        order.status = status
        await db.commit()

        try:
            from app.realtime.events import order_status_changed
            from app.realtime.ws_manager import ws_manager
            evt = order_status_changed(order.id, order.order_ref, status)
            await ws_manager.broadcast(order.restaurant_id, evt.to_dict())
        except Exception:
            pass

    return ActionResult("update_order_status", True, data={
        "order_ref": order.order_ref, "status": status,
    })


async def _tool_assign_driver(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Assign a driver to a delivery order."""
    order_ref = str(args.get("order_ref") or "").strip().upper()
    driver_id = int(args.get("driver_id") or 0)

    if not order_ref or not driver_id:
        return ActionResult("assign_driver", False,
                            error="order_ref and driver_id are required.")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(Order.order_ref == order_ref)
        )).scalar_one_or_none()
        if not order:
            return ActionResult("assign_driver", False, error=f"Order {order_ref} not found.")

        driver = (await db.execute(
            select(Driver).where(Driver.id == driver_id, Driver.active == True)
        )).scalar_one_or_none()
        if not driver:
            return ActionResult("assign_driver", False,
                                error=f"Driver {driver_id} not found or inactive.")

        delivery = (await db.execute(
            select(Delivery).where(Delivery.order_id == order.id)
        )).scalar_one_or_none()

        if delivery:
            delivery.driver_id = driver_id
            delivery.status = "ASSIGNED"
            delivery.assigned_at = datetime.now(timezone.utc)
        else:
            delivery = Delivery(
                restaurant_id=ctx.restaurant_id,
                order_id=order.id,
                driver_id=driver_id,
                status="ASSIGNED",
                address_json={"address": ctx.delivery_address},
                customer_phone=ctx.customer_phone,
                assigned_at=datetime.now(timezone.utc),
            )
            db.add(delivery)

        await db.commit()

        try:
            from app.realtime.events import delivery_status_changed
            from app.realtime.ws_manager import ws_manager
            await ws_manager.broadcast(ctx.restaurant_id,
                delivery_status_changed(delivery.id, order.id, "ASSIGNED").to_dict())
        except Exception:
            pass

    return ActionResult("assign_driver", True, data={
        "order_ref": order_ref, "driver_id": driver_id, "status": "ASSIGNED",
    })


async def _tool_update_delivery_status(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Update delivery status (driver / ops tool)."""
    delivery_id = int(args.get("delivery_id") or 0)
    order_ref = str(args.get("order_ref") or "").strip().upper()
    status = str(args.get("status") or "").strip().upper()

    VALID = {"ASSIGNED", "PICKED_UP", "EN_ROUTE", "DELIVERED", "FAILED"}
    if status not in VALID:
        return ActionResult("update_delivery_status", False,
                            error=f"Invalid status. Use: {', '.join(sorted(VALID))}")

    async with AsyncSessionLocal() as db:
        if delivery_id:
            delivery = (await db.execute(
                select(Delivery).where(Delivery.id == delivery_id)
            )).scalar_one_or_none()
        elif order_ref:
            order = (await db.execute(
                select(Order).where(Order.order_ref == order_ref)
            )).scalar_one_or_none()
            delivery = (await db.execute(
                select(Delivery).where(Delivery.order_id == order.id)
            )).scalar_one_or_none() if order else None
        else:
            return ActionResult("update_delivery_status", False,
                                error="delivery_id or order_ref required.")

        if not delivery:
            return ActionResult("update_delivery_status", False, error="Delivery not found.")

        delivery.status = status
        if status == "PICKED_UP":
            delivery.picked_up_at = datetime.now(timezone.utc)
        elif status == "DELIVERED":
            delivery.delivered_at = datetime.now(timezone.utc)
            # Also mark order delivered
            order_obj = (await db.execute(
                select(Order).where(Order.id == delivery.order_id)
            )).scalar_one_or_none()
            if order_obj:
                order_obj.status = "DELIVERED"
        await db.commit()

    return ActionResult("update_delivery_status", True, data={
        "delivery_id": delivery.id, "status": status,
    })


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Search + item details tools
# ══════════════════════════════════════════════════════════════════════════════

async def _tool_search_menu(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """
    Full-text search on menu items: name, description, tags.
    Supports filter_type (Veg/Non-Veg/EGG/VEGAN) and filter_category.
    """
    query_str = str(args.get("query") or "").strip().lower()
    filter_type = str(args.get("filter_type") or "").strip().upper()
    filter_category = str(args.get("filter_category") or "").strip().lower()

    if not query_str and not filter_type and not filter_category:
        return ActionResult("search_menu", False, error="Provide a search query or filter.")

    # Search against in-memory menu snapshot (already loaded, avoids extra DB query)
    items = ctx.menu_snapshot
    hits = []
    for item in items:
        name_lc = item["name"].lower()
        desc_lc = (item.get("description") or "").lower()
        type_lc = (item.get("type") or "").lower()

        matches_query = (
            not query_str
            or query_str in name_lc
            or query_str in desc_lc
            or any(query_str in (t or "").lower() for t in [type_lc])
        )
        matches_type = (
            not filter_type
            or (item.get("type") or "").upper() == filter_type
        )
        matches_cat = (
            not filter_category
            or filter_category in desc_lc
            or filter_category in name_lc
        )
        if matches_query and matches_type and matches_cat:
            hits.append(item)

    result.menu_items = hits
    return ActionResult("search_menu", True, data={"items": hits, "count": len(hits),
                                                    "query": query_str})


async def _tool_get_item_details(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Fetch full item details including images and modifier groups."""
    item_id = int(args.get("item_id") or 0)
    item_name = str(args.get("item_name") or "").strip()

    async with AsyncSessionLocal() as db:
        if item_id:
            mi = (await db.execute(
                select(MenuItem).where(MenuItem.id == item_id)
            )).scalar_one_or_none()
        else:
            mi = await _find_menu_item(db, ctx.restaurant_id, item_name)

        if not mi:
            return ActionResult("get_item_details", False,
                                error=f"Item '{item_name or item_id}' not found.")

        # Images
        images = (await db.execute(
            select(MenuItemImage).where(MenuItemImage.item_id == mi.id)
            .order_by(MenuItemImage.sort_order)
        )).scalars().all()

        # Modifier groups
        mig_rows = (await db.execute(
            select(MenuItemModifierGroup).where(MenuItemModifierGroup.item_id == mi.id)
        )).scalars().all()

        modifier_groups = []
        for mig in mig_rows:
            group = (await db.execute(
                select(ModifierGroup).where(ModifierGroup.id == mig.group_id)
            )).scalar_one_or_none()
            if not group:
                continue
            modifiers = (await db.execute(
                select(Modifier).where(
                    Modifier.group_id == group.id,
                    Modifier.available == True,
                )
            )).scalars().all()
            modifier_groups.append({
                "id": group.id,
                "name": group.name,
                "min_select": group.min_select,
                "max_select": group.max_select,
                "modifiers": [
                    {"id": m.id, "name": m.name,
                     "price_cents": m.price_cents, "price": m.price_cents / 100}
                    for m in modifiers
                ],
            })

    detail = {
        "id": mi.id,
        "name": mi.name,
        "description": mi.description or "",
        "type": mi.type or "",
        "price_cents": mi.price_cents,
        "price": mi.price_cents / 100,
        "special_price": mi.special_price_cents / 100 if mi.special_price_cents else None,
        "available": mi.available,
        "stock_qty": mi.stock_qty,
        "prep_time_min": (mi.prep_time_sec or 900) // 60,
        "tags": mi.tags or [],
        "images": [{"url": img.url, "alt": img.alt_text or ""} for img in images],
        "modifier_groups": modifier_groups,
    }
    return ActionResult("get_item_details", True, data=detail)


# ── Tool registry ─────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {
    # Cart
    "add_to_cart":              _tool_add_to_cart,
    "update_cart_item":         _tool_update_cart_item,
    "remove_from_cart":         _tool_remove_from_cart,
    "clear_cart":               _tool_clear_cart,
    "view_cart":                _tool_view_cart,
    # Order flow
    "set_order_type":           _tool_set_order_type,
    "set_delivery_address":     _tool_set_delivery_address,
    "update_customer":          _tool_update_customer,
    "place_order":              _tool_place_order,
    "track_order":              _tool_track_order,
    "cancel_order":             _tool_cancel_order,
    # Payment
    "create_payment_intent":    _tool_create_payment_intent,
    "check_payment_status":     _tool_check_payment_status,
    # Reservation
    "make_reservation":         _tool_make_reservation,
    # Dine-in
    "open_table_session":       _tool_open_table_session,
    "add_table_order":          _tool_add_table_order,
    "close_table_session":      _tool_close_table_session,
    "generate_invoice":         _tool_generate_invoice,
    # Ops
    "set_item_availability":    _tool_set_item_availability,
    "update_stock":             _tool_update_stock,
    "update_order_status":      _tool_update_order_status,
    "assign_driver":            _tool_assign_driver,
    "update_delivery_status":   _tool_update_delivery_status,
    # Menu discovery
    "search_menu":              _tool_search_menu,
    "get_item_details":         _tool_get_item_details,
    "get_menu":                 _tool_get_menu,
    "get_kitchen_eta":          _tool_get_kitchen_eta,
    "get_restaurant_info":      _tool_get_restaurant_info,
}


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _get_or_create_channel(db: AsyncSession, chat_id: int) -> Channel:
    ch = (await db.execute(
        select(Channel).where(Channel.type == "TELEGRAM", Channel.external_id == str(chat_id))
    )).scalar_one_or_none()
    if not ch:
        ch = Channel(restaurant_id=1, type="TELEGRAM", external_id=str(chat_id))
        db.add(ch)
        await db.flush()
    return ch


async def _get_or_create_cart(db: AsyncSession, ctx: ContextSnapshot) -> Cart:
    """Return the open cart linked to the session, creating one if needed."""
    if ctx.cart_id:
        cart = (await db.execute(
            select(Cart).where(Cart.id == ctx.cart_id, Cart.status == "OPEN")
        )).scalar_one_or_none()
        if cart:
            return cart

    # Find or create channel + session
    ch = await _get_or_create_channel(db, ctx.chat_id)
    sess = (await db.execute(
        select(SessionModel).where(SessionModel.channel_id == ch.id)
    )).scalar_one_or_none()
    if not sess:
        sess = SessionModel(
            restaurant_id=ctx.restaurant_id,
            channel_id=ch.id,
            state="IDLE",
            history_json={"meta": {}, "turns": []},
        )
        db.add(sess)
        await db.flush()

    cart = Cart(restaurant_id=ctx.restaurant_id, customer_id=ctx.customer_id, status="OPEN")
    db.add(cart)
    await db.flush()
    sess.cart_id = cart.id
    return cart


async def _find_menu_item(db: AsyncSession, restaurant_id: int, name: str) -> Optional[MenuItem]:
    """Find menu item by exact name, then by case-insensitive contains."""
    mi = (await db.execute(
        select(MenuItem).where(
            MenuItem.restaurant_id == restaurant_id,
            MenuItem.name.ilike(name),
        )
    )).scalars().first()
    if not mi:
        mi = (await db.execute(
            select(MenuItem).where(
                MenuItem.restaurant_id == restaurant_id,
                MenuItem.name.ilike(f"%{name}%"),
            )
        )).scalars().first()
    return mi


async def _refresh_cart_snapshot(db: AsyncSession, cart_id: int, result: ExecutionResult) -> None:
    """Re-read cart from DB and update result snapshot."""
    items_result = await db.execute(
        select(CartItem).where(CartItem.cart_id == cart_id)
    )
    cart_items = []
    total = 0
    for ci in items_result.scalars().all():
        mi = (await db.execute(select(MenuItem).where(MenuItem.id == ci.item_id))).scalar_one_or_none()
        cart_items.append({
            "item_id": ci.item_id,
            "item_name": mi.name if mi else "Unknown",
            "qty": ci.qty,
            "unit_price_cents": ci.unit_price_cents,
            "available": bool(mi.available) if mi else False,
            "type": mi.type or "" if mi else "",
        })
        total += ci.unit_price_cents * ci.qty
    result.updated_cart = cart_items
    result.cart_total_cents = total
    result.cart_id = cart_id


# Import at bottom to avoid circular
try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    from datetime import timezone as _tz
    IST = _tz.utc
