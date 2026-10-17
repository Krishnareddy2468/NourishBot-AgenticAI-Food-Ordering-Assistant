"""
Executor — Stage 3 of the NourishBot Pipeline.

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
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
    Driver, Delivery, Invoice, ConfirmationChallenge, MealSchedule,
)
from app.agent.context_builder import ContextSnapshot, save_session_meta
from app.agent.policies import default_policy
from app.agent.persona import find_alternatives, validate_delivery_address
from app.agent.privacy_service import ensure_channel
from app.agent.order_confirmation import (
    build_snapshot,
    create_challenge,
    lock_valid_challenge,
    snapshot_hash,
)
from app.domain.order_state import (
    InvalidStateTransition,
    record_order_event,
    record_schedule_event,
    sync_terminal_schedule,
    transition_order,
    transition_schedule,
)
from app.domain.scheduling import add_durable_jobs

logger = logging.getLogger(__name__)

# The public chat planner may invoke only customer-facing tools. Keeping the
# larger registry for future authenticated staff agents must not make those
# operations reachable through prompt injection in Telegram or WhatsApp.
_CUSTOMER_TOOL_ALLOWLIST = {
    "search_menu", "get_menu", "get_item_details", "get_restaurant_info",
    "search_best_restaurants_on_maps", "find_drive_through_restaurants_on_route",
    "get_kitchen_eta", "add_to_cart", "update_cart_item", "remove_from_cart",
    "view_cart", "clear_cart", "set_order_type", "set_delivery_address",
    "update_customer", "set_schedule", "prepare_order_confirmation",
    "place_order", "track_order", "cancel_order",
}


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
            if tool not in _CUSTOMER_TOOL_ALLOWLIST:
                result.rejected.append(ActionResult(
                    tool=tool,
                    ok=False,
                    error="This operation is not available in customer chat.",
                ))
                continue
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
        ch = await _get_or_create_channel(db, ctx)

        # Find or create customer. A phone match is not proof of account
        # ownership, so an unlinked channel must never inherit an existing
        # customer merely because the caller supplied the same number.
        customer = None
        if ch.customer_id:
            customer = (await db.execute(
                select(Customer).where(
                    Customer.id == ch.customer_id,
                    Customer.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()

        if not customer and phone:
            existing_customer = (await db.execute(
                select(Customer).where(
                    Customer.restaurant_id == ctx.restaurant_id,
                    Customer.phone == phone,
                )
            )).scalar_one_or_none()
            if existing_customer:
                return ActionResult(
                    "update_customer",
                    False,
                    error=(
                        "That phone is already connected to a NourishBot profile. "
                        "Open an already linked channel, request a one-time link code, "
                        "and redeem it here."
                    ),
                )

        if not customer:
            if not phone:
                return ActionResult(
                    "update_customer",
                    False,
                    error="A phone number is required to create your customer profile.",
                )
            customer = Customer(
                restaurant_id=ctx.restaurant_id,
                name=name or ctx.user_name,
                phone=phone,
                language_pref=language or "en",
            )
            db.add(customer)
            await db.flush()

        # Apply updates
        if name:     customer.name = name
        if phone and phone != customer.phone:
            duplicate = (await db.execute(
                select(Customer.id).where(
                    Customer.restaurant_id == ctx.restaurant_id,
                    Customer.phone == phone,
                    Customer.id != customer.id,
                )
            )).scalar_one_or_none()
            if duplicate:
                return ActionResult(
                    "update_customer",
                    False,
                    error="That phone belongs to another profile and cannot be merged automatically.",
                )
            customer.phone = phone
        if language: customer.language_pref = language

        # Link channel → customer
        if not ch.customer_id:
            ch.customer_id = customer.id

        await db.commit()

    saved = {k: v for k, v in {"name": name, "phone": phone, "address": address, "language": language}.items() if v}
    return ActionResult("update_customer", True, data={"saved": saved})


async def _tool_prepare_order_confirmation(
    args: dict,
    ctx: ContextSnapshot,
    result: ExecutionResult,
) -> ActionResult:
    """Persist the exact summary that a later affirmative reply may confirm."""
    if not ctx.session_id:
        return ActionResult("prepare_order_confirmation", False, error="Your chat session is not ready. Please try once more.")
    if not ctx.customer_name or not ctx.customer_phone:
        return ActionResult("prepare_order_confirmation", False, error="Name and phone are required before confirmation.")
    cart = result.updated_cart or ctx.cart
    if not cart:
        return ActionResult("prepare_order_confirmation", False, error="Cart is empty. Add items first.")

    order_type = str(args.get("order_type") or ctx.order_type or "PICKUP").upper()
    address = str(args.get("address") or ctx.delivery_address or "").strip()
    scheduled_raw = str(args.get("scheduled_for") or ctx.scheduled_for or "").strip()
    if order_type not in ("PICKUP", "DRIVE_THRU", "DELIVERY"):
        return ActionResult("prepare_order_confirmation", False, error="Choose pickup, drive-through, or home delivery.")
    if order_type == "DELIVERY" and not address:
        return ActionResult("prepare_order_confirmation", False, error="Delivery address is required.")
    try:
        scheduled_for = datetime.fromisoformat(scheduled_raw.replace("Z", "+00:00"))
    except ValueError:
        return ActionResult("prepare_order_confirmation", False, error="Use a valid future meal time with timezone.")
    if scheduled_for.tzinfo is None:
        return ActionResult("prepare_order_confirmation", False, error="Meal time must include an explicit timezone.")
    valid_schedule, schedule_error = ctx.policy.validate_scheduled_for(scheduled_for, ctx.now)
    if not valid_schedule:
        return ActionResult("prepare_order_confirmation", False, error=schedule_error)
    if order_type == "DELIVERY":
        deliverable, reason = validate_delivery_address(address, ctx.policy)
        if not deliverable:
            return ActionResult(
                "prepare_order_confirmation",
                False,
                error=reason,
                data={"radius_exceeded": True},
            )

    async with AsyncSessionLocal() as db:
        verified_items = []
        for cart_item in cart:
            menu_item = (await db.execute(
                select(MenuItem).where(
                    MenuItem.id == cart_item["item_id"],
                    MenuItem.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()
            if not menu_item or not menu_item.available:
                return ActionResult("prepare_order_confirmation", False, error=f"{cart_item['item_name']} is no longer available.")
            verified_items.append({
                "item_id": menu_item.id,
                "item_name": menu_item.name,
                "qty": cart_item["qty"],
                "unit_price_cents": menu_item.price_cents,
            })

        snapshot = build_snapshot(ctx=ctx, args={
            **args,
            "order_type": order_type,
            "address": address,
            "scheduled_for": scheduled_raw,
        }, items=verified_items)
        challenge = await create_challenge(db, ctx=ctx, snapshot=snapshot)
        await db.commit()

    result.updated_cart = verified_items
    result.cart_total_cents = snapshot["total_cents"]
    return ActionResult("prepare_order_confirmation", True, data={
        "confirmation_id": challenge.id,
        "confirmation_required": True,
        "order_type": snapshot["order_type"],
        "delivery_address": snapshot["delivery_address"],
        "scheduled_for": snapshot["scheduled_for"],
        "provider": snapshot["provider"],
        "total_cents": snapshot["total_cents"],
        "items": snapshot["items"],
        "expires_at": challenge.expires_at.isoformat(),
    })


async def _tool_place_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    order_type = str(args.get("order_type") or ctx.order_type or "PICKUP").upper()
    address = str(args.get("address") or ctx.delivery_address or "").strip()
    notes = str(args.get("notes") or "").strip()
    provider = str(args.get("provider") or ctx.ordering_platform or "NOURISHBOT").strip().upper()
    scheduled_raw = str(args.get("scheduled_for") or ctx.scheduled_for or "").strip()
    confirmation_id = str(args.get("confirmation_id") or "").strip()

    if not confirmation_id or not ctx.session_id:
        return ActionResult(
            "place_order",
            False,
            error="Review the final meal summary and explicitly confirm it before ordering.",
        )

    # ── Policy checks ──────────────────────────────────────────────────────
    if not scheduled_raw:
        return ActionResult(
            "place_order",
            False,
            error="Choose a future meal date and time. Instant ordering is not available.",
        )
    try:
        scheduled_for = datetime.fromisoformat(scheduled_raw.replace("Z", "+00:00"))
    except ValueError:
        return ActionResult("place_order", False, error="Use a valid meal date and time with timezone.")
    valid_schedule, schedule_error = ctx.policy.validate_scheduled_for(scheduled_for, ctx.now)
    if not valid_schedule:
        return ActionResult("place_order", False, error=schedule_error)
    if scheduled_for.tzinfo is None:
        return ActionResult("place_order", False, error="Meal time must include an explicit timezone.")
    scheduled_for = scheduled_for.astimezone(timezone.utc)

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

    # This is a future order. Current live kitchen load must not reject a meal
    # scheduled for another service window; activation handles the live queue.
    eta = ctx.policy.base_prep_time_sec

    if len(cart) > ctx.policy.max_items_per_order:
        return ActionResult("place_order", False,
                            error=f"Order exceeds maximum {ctx.policy.max_items_per_order} items.")

    async with AsyncSessionLocal() as db:
        challenge_record = (await db.execute(
            select(ConfirmationChallenge).where(
                ConfirmationChallenge.id == confirmation_id,
                ConfirmationChallenge.restaurant_id == ctx.restaurant_id,
                ConfirmationChallenge.session_id == ctx.session_id,
            )
        )).scalar_one_or_none()
        if not challenge_record:
            return ActionResult("place_order", False, error="That confirmation is invalid. Please review the order again.")

        existing_order = (await db.execute(
            select(Order).where(
                Order.idempotency_key == challenge_record.idempotency_key,
                Order.restaurant_id == ctx.restaurant_id,
            )
        )).scalar_one_or_none()
        if existing_order:
            result.order_ref = existing_order.order_ref
            return ActionResult("place_order", True, data={
                "order_ref": existing_order.order_ref,
                "order_type": existing_order.order_type,
                "scheduled_for": existing_order.scheduled_for.isoformat() if existing_order.scheduled_for else None,
                "provider": existing_order.provider,
                "idempotent_replay": True,
                "tracking_token": existing_order.tracking_token,
            })

        challenge = await lock_valid_challenge(
            db,
            challenge_id=confirmation_id,
            restaurant_id=ctx.restaurant_id,
            session_id=ctx.session_id,
        )
        if not challenge:
            return ActionResult("place_order", False, error="That confirmation expired or was already used. Please review the order again.")

        # Re-read item prices from DB — never trust in-memory cart prices
        order_ref = f"NB-{uuid.uuid4().hex[:6].upper()}"
        subtotal = 0
        validated_items = []

        for ci in cart:
            mi = (await db.execute(
                select(MenuItem).where(MenuItem.id == ci["item_id"], MenuItem.restaurant_id == ctx.restaurant_id)
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

        current_snapshot = build_snapshot(ctx=ctx, args={
            **args,
            "order_type": order_type,
            "address": address,
            "scheduled_for": scheduled_for.isoformat(),
            "provider": provider,
        }, items=validated_items)
        if snapshot_hash(current_snapshot) != challenge.snapshot_hash:
            challenge.invalidated_at = datetime.now(timezone.utc)
            challenge.invalidated_by_channel_id = ctx.channel_id
            challenge.invalidation_reason = "SNAPSHOT_CHANGED"
            await db.commit()
            return ActionResult(
                "place_order",
                False,
                error="The cart, price, address, provider, or meal time changed. Please review and confirm the updated summary.",
            )

        # Delivery minimum check
        if order_type == "DELIVERY" and subtotal < ctx.policy.min_order_for_delivery_cents:
            min_inr = ctx.policy.min_order_for_delivery_cents // 100
            return ActionResult(
                "place_order", False,
                error=f"Minimum order for delivery is ₹{min_inr}. Current total: ₹{subtotal // 100}.",
            )

        # Find channel for FK
        ch = await _get_or_create_channel(db, ctx)

        order = Order(
            restaurant_id=ctx.restaurant_id,
            order_ref=order_ref,
            tracking_token=secrets.token_urlsafe(32),
            confirmation_id=challenge.id,
            customer_id=ctx.customer_id,
            channel_id=ch.id,
            order_type=order_type,
            status="CONFIRMED",
            provider=provider,
            schedule_state="SCHEDULED",
            provider_state="NOT_REQUIRED" if provider.upper() == "NOURISHBOT" else "PENDING",
            fulfillment_state="NOT_STARTED",
            source_channel=ctx.channel_type,
            scheduled_for=scheduled_for,
            desired_for=scheduled_for,
            ready_by=scheduled_for,
            submission_window_start=scheduled_for - timedelta(minutes=settings.SCHEDULE_ACTIVATION_LEAD_MINUTES),
            submission_window_end=scheduled_for,
            subtotal_cents=subtotal,
            total_cents=subtotal,
            notes=notes or None,
            idempotency_key=challenge.idempotency_key,
        )
        db.add(order)
        await db.flush()
        record_order_event(
            db,
            order,
            event_type="ORDER_CREATED",
            to_state="CONFIRMED",
            actor_type="CUSTOMER",
            actor_id=ctx.customer_id,
            source=ctx.channel_type,
            reason_code="EXPLICIT_CONFIRMATION",
            metadata={"confirmation_id": challenge.id},
        )

        if ctx.customer_id:
            meal_schedule = MealSchedule(
                restaurant_id=ctx.restaurant_id,
                customer_id=ctx.customer_id,
                order_id=order.id,
                title=f"Meal order {order_ref}",
                meal_period="UNSPECIFIED",
                state="SCHEDULED",
                desired_for=scheduled_for,
                ready_by=scheduled_for,
                submission_window_start=scheduled_for - timedelta(minutes=settings.SCHEDULE_ACTIVATION_LEAD_MINUTES),
                submission_window_end=scheduled_for,
                timezone=str(scheduled_for.tzinfo),
                source=ctx.channel_type,
            )
            db.add(meal_schedule)
            await db.flush()
            record_schedule_event(
                db,
                meal_schedule,
                event_type="SCHEDULE_CREATED",
                to_state="SCHEDULED",
                actor_type="CUSTOMER",
                actor_id=ctx.customer_id,
                source=ctx.channel_type,
                reason_code="ORDER_CONFIRMED",
            )
            add_durable_jobs(db, order, meal_schedule)

        if order_type == "DELIVERY":
            db.add(Delivery(
                restaurant_id=ctx.restaurant_id,
                order_id=order.id,
                status="PENDING_ASSIGNMENT",
                address_json={"address": address},
                customer_phone=ctx.customer_phone,
            ))

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
            cart_obj = (await db.execute(
                select(Cart).where(Cart.id == ctx.cart_id, Cart.restaurant_id == ctx.restaurant_id)
            )).scalar_one_or_none()
            if cart_obj:
                cart_obj.status = "CONVERTED"

        challenge.used_at = datetime.now(timezone.utc)
        challenge.confirmed_by_channel_id = ctx.channel_id
        challenge.confirmation_actor = "CUSTOMER"

        await db.commit()

        # Clear cart snapshot in result
        result.updated_cart = []
        result.cart_total_cents = 0
        result.order_ref = order_ref

    logger.info("Order placed: %s for customer_id=%s", order_ref, ctx.customer_id)

    # ── Auto-chain: create Razorpay payment intent for scheduled in-house orders ──
    payment_data = {}
    razorpay_configured = bool(settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET)
    if order_type in ("PICKUP", "DRIVE_THRU", "DELIVERY") and razorpay_configured:
        try:
            payment_result = await _tool_create_payment_intent(
                {"order_ref": order_ref}, ctx, result
            )
            if payment_result.ok:
                payment_data = payment_result.data
        except Exception as e:
            logger.warning("Auto payment intent failed for %s: %s", order_ref, e)
    elif order_type in ("PICKUP", "DRIVE_THRU", "DELIVERY") and not razorpay_configured:
        logger.info("Razorpay not configured — order %s confirmed as COD.", order_ref)

    return ActionResult("place_order", True, data={
        "order_ref": order_ref,
        "order_type": order_type,
        "subtotal_cents": subtotal,
        "eta_seconds": eta,
        "items": validated_items,
        "tracking_token": order.tracking_token,
        "payment_required": bool(payment_data),
        "payment_url": payment_data.get("razorpay_order_id", ""),
        "scheduled_for": scheduled_for.isoformat(),
        "provider": provider,
    })


async def _tool_set_schedule(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    """Validate and persist the future meal time before final confirmation."""
    scheduled_raw = str(args.get("scheduled_for") or "").strip()
    if not scheduled_raw:
        return ActionResult("set_schedule", False, error="Choose a meal date and time.")
    try:
        scheduled_for = datetime.fromisoformat(scheduled_raw.replace("Z", "+00:00"))
    except ValueError:
        return ActionResult("set_schedule", False, error="Use a valid meal date and time with timezone.")
    valid, error = ctx.policy.validate_scheduled_for(scheduled_for, ctx.now)
    if not valid:
        return ActionResult("set_schedule", False, error=error)
    if scheduled_for.tzinfo is None:
        return ActionResult("set_schedule", False, error="Meal time must include an explicit timezone.")
    normalized = scheduled_for.astimezone(timezone.utc).isoformat()
    await save_session_meta(
        ctx.chat_id,
        {"scheduled_for": normalized},
        channel_type=ctx.channel_type,
    )
    return ActionResult("set_schedule", True, data={"scheduled_for": normalized})


async def _tool_track_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    order_ref = str(args.get("order_ref") or "").strip().upper()

    # Try active orders from context first
    if not order_ref and ctx.active_orders:
        order_ref = ctx.active_orders[0]["order_ref"]

    if not order_ref:
        return ActionResult("track_order", False, error="Please provide an order reference (e.g. NB-ABC123).")
    if not ctx.customer_id:
        return ActionResult("track_order", False, error="A verified customer identity is required to view an order.")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(
                Order.order_ref == order_ref,
                Order.restaurant_id == ctx.restaurant_id,
                Order.customer_id == ctx.customer_id,
            )
        )).scalar_one_or_none()

        if not order:
            return ActionResult("track_order", False, error=f"Order {order_ref} not found.")

        info = {
            "order_ref": order.order_ref,
            "status": order.status,
            "total_cents": order.total_cents,
            "order_type": order.order_type,
            "created_at": order.created_at.isoformat() if order.created_at else "",
            "scheduled_for": order.scheduled_for.isoformat() if order.scheduled_for else "",
            "provider": order.provider,
        }
        result.tracking_info = info

    return ActionResult("track_order", True, data=info)


async def _tool_cancel_order(args: dict, ctx: ContextSnapshot, result: ExecutionResult) -> ActionResult:
    order_ref = str(args.get("order_ref") or "").strip().upper()
    if not order_ref:
        return ActionResult("cancel_order", False, error="Order reference required.")
    if not ctx.customer_id:
        return ActionResult("cancel_order", False, error="A verified customer identity is required to cancel an order.")

    async with AsyncSessionLocal() as db:
        order = (await db.execute(
            select(Order).where(
                Order.order_ref == order_ref,
                Order.restaurant_id == ctx.restaurant_id,
                Order.customer_id == ctx.customer_id,
            )
        )).scalar_one_or_none()

        if not order:
            return ActionResult("cancel_order", False, error=f"Order {order_ref} not found.")

        if not ctx.policy.can_cancel_order(order.status):
            return ActionResult(
                "cancel_order", False,
                error=f"Order {order_ref} is already {order.status} and cannot be cancelled. "
                      f"Call us directly to arrange changes.",
            )

        transition_order(
            db,
            order,
            "CANCELLED",
            actor_type="CUSTOMER",
            actor_id=ctx.customer_id,
            source=ctx.channel_type,
            reason_code="CUSTOMER_CANCELLED",
        )
        if order.schedule_state in {"SCHEDULED", "ACTIVATED"}:
            meal_schedule = (await db.execute(
                select(MealSchedule).where(MealSchedule.order_id == order.id)
            )).scalar_one_or_none()
            transition_schedule(
                db,
                order,
                "CANCELLED",
                actor_type="CUSTOMER",
                actor_id=ctx.customer_id,
                source=ctx.channel_type,
                reason_code="CUSTOMER_CANCELLED",
                meal_schedule=meal_schedule,
            )
        await db.commit()

        # Broadcast real-time event
        try:
            from app.realtime.events import order_status_changed
            from app.realtime.ws_manager import ws_manager
            evt = order_status_changed(order.id, order_ref, "CANCELLED", ctx.restaurant_id)
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
                select(Customer).where(
                    Customer.id == ctx.customer_id,
                    Customer.restaurant_id == ctx.restaurant_id,
                )
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


async def _tool_search_best_restaurants_on_maps(
    args: dict, ctx: ContextSnapshot, result: ExecutionResult
) -> ActionResult:
    from app.agent.maps_client import MapsServiceError, search_best_restaurants

    try:
        info = await search_best_restaurants(
            location=str(args.get("location") or ""),
            query=str(args.get("query") or "restaurants"),
            max_results=int(args.get("max_results") or 5),
        )
    except MapsServiceError as exc:
        return ActionResult("search_best_restaurants_on_maps", False, error=str(exc))
    result.info_payload = info
    return ActionResult("search_best_restaurants_on_maps", True, data=info)


async def _tool_find_drive_through_restaurants_on_route(
    args: dict, ctx: ContextSnapshot, result: ExecutionResult
) -> ActionResult:
    from app.agent.maps_client import MapsServiceError, find_drive_through_on_route

    try:
        info = await find_drive_through_on_route(
            origin=str(args.get("origin") or ""),
            destination=str(args.get("destination") or ""),
            cuisine=str(args.get("cuisine") or ""),
            depart_at=str(args.get("depart_at") or datetime.now(timezone.utc).isoformat()),
            max_results=int(args.get("max_results") or 5),
        )
    except MapsServiceError as exc:
        return ActionResult("find_drive_through_restaurants_on_route", False, error=str(exc))
    result.info_payload = info
    return ActionResult("find_drive_through_restaurants_on_route", True, data=info)


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
    """Persist the chosen fulfillment type to the conversation session."""
    order_type = str(args.get("order_type") or "").strip().upper()
    if order_type not in ("PICKUP", "DRIVE_THRU", "DELIVERY"):
        return ActionResult("set_order_type", False,
                            error=f"Invalid order_type '{order_type}'. Use PICKUP, DRIVE_THRU, or DELIVERY.")

    valid, msg = ctx.policy.validate_order_type(order_type)
    if not valid:
        return ActionResult("set_order_type", False, error=msg)

    await save_session_meta(ctx.chat_id, {"order_type": order_type}, channel_type=ctx.channel_type)
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

    await save_session_meta(ctx.chat_id, {"delivery_address": address}, channel_type=ctx.channel_type)
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
            select(Order).where(
                Order.order_ref == order_ref,
                Order.restaurant_id == ctx.restaurant_id,
                Order.customer_id == ctx.customer_id,
            )
        )).scalar_one_or_none()
        if not order:
            return ActionResult("create_payment_intent", False, error=f"Order {order_ref} not found.")

        # Idempotency: return existing pending payment if present
        existing_payment = (await db.execute(
            select(Payment).where(
                Payment.order_id == order.id,
                Payment.restaurant_id == ctx.restaurant_id,
                Payment.status == "CREATED",
            )
        )).scalar_one_or_none()

        if existing_payment and existing_payment.provider_order_id:
            await save_session_meta(ctx.chat_id, {
                "pending_payment_order_id": existing_payment.provider_order_id,
                "pending_order_ref": order_ref,
            }, channel_type=ctx.channel_type)
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
        }, channel_type=ctx.channel_type)

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
                select(Order).where(
                    Order.order_ref == order_ref,
                    Order.restaurant_id == ctx.restaurant_id,
                    Order.customer_id == ctx.customer_id,
                )
            )).scalar_one_or_none()
            if order:
                payment = (await db.execute(
                    select(Payment).where(
                        Payment.order_id == order.id,
                        Payment.restaurant_id == ctx.restaurant_id,
                    )
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
                TableSession.restaurant_id == ctx.restaurant_id,
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
                                       table_session_opened(ts.id, table_id, ctx.restaurant_id).to_dict())
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
            select(TableSession).where(
                TableSession.id == session_id,
                TableSession.restaurant_id == ctx.restaurant_id,
            )
        )).scalar_one_or_none()
        if not ts or ts.status != "OPEN":
            return ActionResult("add_table_order", False,
                                error=f"Table session {session_id} not found or not open.")

        import uuid as _uuid
        order_ref = f"NB-{_uuid.uuid4().hex[:6].upper()}"
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
        record_order_event(
            db,
            order,
            event_type="ORDER_CREATED",
            to_state="ACCEPTED",
            actor_type="STAFF",
            source="AGENT_TOOL",
            reason_code="DINE_IN_ORDER_CREATED",
        )

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
            select(TableSession).where(
                TableSession.id == session_id,
                TableSession.restaurant_id == ctx.restaurant_id,
            )
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
                                       table_session_closed(session_id, ts.table_id, ctx.restaurant_id).to_dict())
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
                    TableSessionOrder.table_session_id == entity_id,
                    TableSessionOrder.restaurant_id == ctx.restaurant_id,
                )
            )).scalars().all()
            for link in links:
                order = (await db.execute(
                    select(Order).where(
                        Order.id == link.order_id,
                        Order.restaurant_id == ctx.restaurant_id,
                    )
                )).scalar_one_or_none()
                if order:
                    subtotal += order.total_cents
        elif entity_type == "ORDER":
            order = (await db.execute(
                select(Order).where(
                    Order.id == entity_id,
                    Order.restaurant_id == ctx.restaurant_id,
                )
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
                select(TableSession).where(
                    TableSession.id == entity_id,
                    TableSession.restaurant_id == ctx.restaurant_id,
                )
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
                select(MenuItem).where(
                    MenuItem.id == item_id,
                    MenuItem.restaurant_id == ctx.restaurant_id,
                )
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
                                       menu_item_availability_changed(mi.id, available, ctx.restaurant_id).to_dict())
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
                select(MenuItem).where(
                    MenuItem.id == item_id,
                    MenuItem.restaurant_id == ctx.restaurant_id,
                )
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
                select(Order).where(
                    Order.order_ref == order_ref,
                    Order.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()
        elif order_id:
            order = (await db.execute(
                select(Order).where(
                    Order.id == order_id,
                    Order.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()
        else:
            return ActionResult("update_order_status", False,
                                error="order_ref or order_id required.")

        if not order:
            return ActionResult("update_order_status", False,
                                error=f"Order not found.")

        try:
            transition_order(
                db,
                order,
                status,
                actor_type="STAFF",
                source="AGENT_TOOL",
                reason_code="STAFF_STATE_UPDATE",
            )
        except InvalidStateTransition as exc:
            return ActionResult("update_order_status", False, error=str(exc))
        meal_schedule = (await db.execute(
            select(MealSchedule).where(MealSchedule.order_id == order.id)
        )).scalar_one_or_none()
        sync_terminal_schedule(
            db,
            order,
            meal_schedule=meal_schedule,
            actor_type="STAFF",
            source="AGENT_TOOL",
            reason_code="ORDER_TERMINAL_STATE_SYNC",
        )
        await db.commit()

        try:
            from app.realtime.events import order_status_changed
            from app.realtime.ws_manager import ws_manager
            evt = order_status_changed(order.id, order.order_ref, status, ctx.restaurant_id)
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
            select(Order).where(
                Order.order_ref == order_ref,
                Order.restaurant_id == ctx.restaurant_id,
            )
        )).scalar_one_or_none()
        if not order:
            return ActionResult("assign_driver", False, error=f"Order {order_ref} not found.")

        driver = (await db.execute(
            select(Driver).where(
                Driver.id == driver_id,
                Driver.restaurant_id == ctx.restaurant_id,
                Driver.active == True,
            )
        )).scalar_one_or_none()
        if not driver:
            return ActionResult("assign_driver", False,
                                error=f"Driver {driver_id} not found or inactive.")

        delivery = (await db.execute(
            select(Delivery).where(
                Delivery.order_id == order.id,
                Delivery.restaurant_id == ctx.restaurant_id,
            )
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
                delivery_status_changed(delivery.id, order.id, "ASSIGNED", ctx.restaurant_id).to_dict())
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
                select(Delivery).where(
                    Delivery.id == delivery_id,
                    Delivery.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()
        elif order_ref:
            order = (await db.execute(
                select(Order).where(
                    Order.order_ref == order_ref,
                    Order.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()
            delivery = (await db.execute(
                select(Delivery).where(
                    Delivery.order_id == order.id,
                    Delivery.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none() if order else None
        else:
            return ActionResult("update_delivery_status", False,
                                error="delivery_id or order_ref required.")

        if not delivery:
            return ActionResult("update_delivery_status", False, error="Delivery not found.")

        delivery.status = status
        order_obj = None
        if status in {"PICKED_UP", "EN_ROUTE", "DELIVERED", "FAILED"}:
            order_obj = (await db.execute(
                select(Order).where(
                    Order.id == delivery.order_id,
                    Order.restaurant_id == ctx.restaurant_id,
                )
            )).scalar_one_or_none()

        try:
            if status == "PICKED_UP":
                delivery.picked_up_at = datetime.now(timezone.utc)
                if order_obj and order_obj.status != "OUT_FOR_DELIVERY":
                    transition_order(
                        db, order_obj, "OUT_FOR_DELIVERY",
                        actor_type="STAFF", source="AGENT_TOOL",
                        reason_code="DRIVER_PICKED_UP",
                    )
            elif status == "EN_ROUTE":
                if order_obj and order_obj.status != "OUT_FOR_DELIVERY":
                    transition_order(
                        db, order_obj, "OUT_FOR_DELIVERY",
                        actor_type="STAFF", source="AGENT_TOOL",
                        reason_code="DELIVERY_EN_ROUTE",
                    )
            elif status == "DELIVERED":
                delivery.delivered_at = datetime.now(timezone.utc)
                if order_obj:
                    transition_order(
                        db, order_obj, "DELIVERED",
                        actor_type="STAFF", source="AGENT_TOOL",
                        reason_code="DELIVERY_COMPLETED",
                    )
            elif status == "FAILED" and order_obj:
                transition_order(
                    db, order_obj, "FAILED",
                    actor_type="STAFF", source="AGENT_TOOL",
                    reason_code="DELIVERY_FAILED",
                )
        except InvalidStateTransition as exc:
            return ActionResult("update_delivery_status", False, error=str(exc))

        if order_obj:
            meal_schedule = (await db.execute(
                select(MealSchedule).where(MealSchedule.order_id == order_obj.id)
            )).scalar_one_or_none()
            sync_terminal_schedule(
                db,
                order_obj,
                meal_schedule=meal_schedule,
                actor_type="STAFF",
                source="AGENT_TOOL",
                reason_code="DELIVERY_OUTCOME_SYNC",
            )
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
                select(MenuItem).where(
                    MenuItem.id == item_id,
                    MenuItem.restaurant_id == ctx.restaurant_id,
                )
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
    "set_schedule":             _tool_set_schedule,
    "set_delivery_address":     _tool_set_delivery_address,
    "update_customer":          _tool_update_customer,
    "prepare_order_confirmation": _tool_prepare_order_confirmation,
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
    "search_best_restaurants_on_maps": _tool_search_best_restaurants_on_maps,
    "find_drive_through_restaurants_on_route": _tool_find_drive_through_restaurants_on_route,
}


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _get_or_create_channel(db: AsyncSession, ctx: ContextSnapshot) -> Channel:
    return await ensure_channel(
        db,
        restaurant_id=ctx.restaurant_id,
        channel_type=ctx.channel_type,
        external_id=ctx.chat_id,
    )


async def _get_or_create_cart(db: AsyncSession, ctx: ContextSnapshot) -> Cart:
    """Return the open cart linked to the session, creating one if needed."""
    if ctx.cart_id:
        cart = (await db.execute(
            select(Cart).where(
                Cart.id == ctx.cart_id,
                Cart.restaurant_id == ctx.restaurant_id,
                Cart.status == "OPEN",
            )
        )).scalar_one_or_none()
        if cart:
            return cart

    # Find or create channel + session
    ch = await _get_or_create_channel(db, ctx)
    sess = (await db.execute(
        select(SessionModel).where(
            SessionModel.channel_id == ch.id,
            SessionModel.restaurant_id == ctx.restaurant_id,
        )
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
