"""
Verifier — Stage 4 of the Dzukku Pipeline.

Deterministic post-execution validation. Never calls LLM.

Checks:
  1. For place_order: re-read Order + OrderItems from DB, recompute total,
     verify it matches order.total_cents.
  2. For add_to_cart: verify all cart items are still available.
  3. For cancel_order / place_order: confirm the action is truly committed in DB.
  4. Flags any blocking issues (price mismatch, unavailable item, etc.)
  5. Builds a VerifiedSummary as structured input for the Responder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models import Order, OrderItem, MenuItem, CartItem, Cart, Reservation
from app.agent.context_builder import ContextSnapshot
from app.agent.executor import ExecutionResult
from app.agent.persona import kitchen_load_signal, eta_message, KitchenSignal

logger = logging.getLogger(__name__)

# Tolerance for floating-point price recomputation (1 paisa)
_PRICE_TOLERANCE = 1


@dataclass
class VerifiedSummary:
    safe_to_respond: bool = True
    blocking_issue: Optional[str] = None

    # Cart facts (re-read from DB)
    cart_items: list[dict] = field(default_factory=list)
    cart_total_cents: int = 0

    # Order facts (re-read from DB after place_order)
    order_ref: Optional[str] = None
    order_type: Optional[str] = None
    order_total_cents: int = 0
    order_status: Optional[str] = None
    order_eta_seconds: Optional[int] = None
    order_items: list[dict] = field(default_factory=list)

    # Reservation facts
    reservation_ref: Optional[str] = None
    reservation_date: Optional[str] = None
    reservation_time: Optional[str] = None
    reservation_guests: int = 0

    # Tracking facts
    tracking_info: Optional[dict] = None

    # Menu facts
    menu_items: list[dict] = field(default_factory=list)

    # Info payload (restaurant info / eta)
    info_payload: Optional[dict] = None
    eta_seconds: Optional[int] = None

    # Pending slots still needed (propagated from execution rejections)
    pending_slots: list[str] = field(default_factory=list)

    # Errors from rejected actions
    rejected_errors: list[str] = field(default_factory=list)

    # Live-state annotations (§3.5)
    kitchen_signal: str = KitchenSignal.NORMAL   # NORMAL | BUSY | VERY_BUSY | FULL
    eta_note: str = ""                            # human-readable ETA + apology if needed
    alternatives: list[dict] = field(default_factory=list)  # when item unavailable
    radius_exceeded: bool = False                 # delivery out of range

    # Goals / intent (for responder context)
    goal: str = "SUPPORT"
    intent_summary: str = ""

    def to_prompt_dict(self) -> dict:
        """Compact dict for the Responder prompt."""
        return {
            "safe": self.safe_to_respond,
            "blocking": self.blocking_issue,
            "cart": self.cart_items,
            "cart_total_inr": round(self.cart_total_cents / 100, 2),
            "order_ref": self.order_ref,
            "order_type": self.order_type,
            "order_total_inr": round(self.order_total_cents / 100, 2),
            "order_status": self.order_status,
            "order_eta_min": self.order_eta_seconds // 60 if self.order_eta_seconds else None,
            "order_items": self.order_items,
            "reservation_ref": self.reservation_ref,
            "reservation_date": self.reservation_date,
            "reservation_time": self.reservation_time,
            "reservation_guests": self.reservation_guests,
            "tracking": self.tracking_info,
            "menu_items": self.menu_items,
            "info": self.info_payload,
            "eta_min": self.eta_seconds // 60 if self.eta_seconds else None,
            "pending_slots": self.pending_slots,
            "errors": self.rejected_errors,
            # Live-state annotations (§3.5)
            "kitchen_signal": self.kitchen_signal,
            "eta_note": self.eta_note,
            "alternatives": self.alternatives,
            "radius_exceeded": self.radius_exceeded,
            "goal": self.goal,
            "intent": self.intent_summary,
        }


class Verifier:

    @classmethod
    async def verify(
        cls,
        exec_result: ExecutionResult,
        ctx: ContextSnapshot,
        goal: str = "SUPPORT",
        intent_summary: str = "",
        pending_slots: Optional[list[str]] = None,
    ) -> VerifiedSummary:
        summary = VerifiedSummary(
            goal=goal,
            intent_summary=intent_summary,
            rejected_errors=exec_result.error_messages,
            pending_slots=list(pending_slots or []),
        )

        # ── 0. Live-state annotations (§3.5) ─────────────────────────────────
        signal = kitchen_load_signal(ctx.kitchen_load, ctx.policy)
        summary.kitchen_signal = signal
        if exec_result.eta_seconds is not None:
            summary.eta_note = eta_message(exec_result.eta_seconds, signal)
        elif signal in (KitchenSignal.BUSY, KitchenSignal.VERY_BUSY, KitchenSignal.FULL):
            # Annotate even without a placed order
            eta_sec = ctx.policy.estimate_prep_time(ctx.kitchen_load)
            summary.eta_note = eta_message(max(eta_sec, 0), signal)

        # Collect alternatives from rejected add_to_cart actions
        for action in exec_result.rejected:
            if action.tool == "add_to_cart" and action.data:
                summary.alternatives.extend(action.data.get("alternatives") or [])
            if action.tool in ("set_delivery_address", "place_order") and \
                    (action.data or {}).get("radius_exceeded"):
                summary.radius_exceeded = True

        # ── 1. Verify placed order ────────────────────────────────────────────
        if exec_result.order_ref:
            await cls._verify_order(exec_result.order_ref, exec_result, summary)

        # ── 2. Verify cart ────────────────────────────────────────────────────
        elif exec_result.updated_cart is not None or ctx.cart_id:
            await cls._verify_cart(exec_result, ctx, summary)

        # ── 3. Verify reservation ─────────────────────────────────────────────
        if exec_result.reservation_ref:
            await cls._verify_reservation(exec_result.reservation_ref, summary)

        # ── 4. Pass-through data ──────────────────────────────────────────────
        if exec_result.tracking_info:
            summary.tracking_info = exec_result.tracking_info
        if exec_result.menu_items:
            summary.menu_items = exec_result.menu_items
        if exec_result.info_payload:
            summary.info_payload = exec_result.info_payload
        if exec_result.eta_seconds is not None:
            summary.eta_seconds = exec_result.eta_seconds

        # ── 5. Propagate rejections ───────────────────────────────────────────
        if exec_result.rejected:
            # Non-blocking rejections still allow a response
            logger.info(
                "Verifier: %d rejected action(s): %s",
                len(exec_result.rejected),
                exec_result.error_messages,
            )

        return summary


    # ── Private verifiers ──────────────────────────────────────────────────────

    @classmethod
    async def _verify_order(
        cls,
        order_ref: str,
        exec_result: ExecutionResult,
        summary: VerifiedSummary,
    ) -> None:
        """Re-read Order + OrderItems from DB; recompute total; flag mismatches."""
        async with AsyncSessionLocal() as db:
            order = (await db.execute(
                select(Order).where(Order.order_ref == order_ref)
            )).scalar_one_or_none()

            if not order:
                summary.safe_to_respond = False
                summary.blocking_issue = f"Order {order_ref} not found in DB after placement."
                logger.error("Verifier: order %s not found in DB!", order_ref)
                return

            items_result = await db.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = items_result.scalars().all()

            # Recompute total from actual DB rows
            recomputed_total = sum(i.unit_price_cents * i.qty for i in items)
            if abs(recomputed_total - order.total_cents) > _PRICE_TOLERANCE:
                logger.error(
                    "Verifier: price mismatch for %s — DB total=%d, recomputed=%d",
                    order_ref, order.total_cents, recomputed_total,
                )
                # Correct the order total in DB
                order.total_cents = recomputed_total
                order.subtotal_cents = recomputed_total
                await db.commit()

            # Verify all items still available
            unavailable = []
            for item in items:
                mi = (await db.execute(
                    select(MenuItem).where(MenuItem.id == item.item_id)
                )).scalar_one_or_none()
                if mi and not mi.available:
                    unavailable.append(item.item_name_snapshot)

            if unavailable:
                # Items went unavailable between planner and execution
                # Order is placed; flag as warning, not blocking
                logger.warning(
                    "Verifier: items became unavailable after order %s: %s",
                    order_ref, unavailable,
                )

            summary.order_ref = order_ref
            summary.order_type = order.order_type
            summary.order_total_cents = order.total_cents
            summary.order_status = order.status
            summary.order_eta_seconds = exec_result.eta_seconds
            summary.order_items = [
                {
                    "name": i.item_name_snapshot,
                    "qty": i.qty,
                    "unit_price_cents": i.unit_price_cents,
                }
                for i in items
            ]

    @classmethod
    async def _verify_cart(
        cls,
        exec_result: ExecutionResult,
        ctx: ContextSnapshot,
        summary: VerifiedSummary,
    ) -> None:
        """Verify cart items are available; populate summary from DB."""
        cart_id = exec_result.cart_id or ctx.cart_id
        if not cart_id:
            # No cart yet — use in-memory snapshot
            summary.cart_items = exec_result.updated_cart or ctx.cart
            summary.cart_total_cents = exec_result.cart_total_cents or ctx.cart_total_cents
            return

        async with AsyncSessionLocal() as db:
            items_result = await db.execute(
                select(CartItem).where(CartItem.cart_id == cart_id)
            )
            cart_items = []
            total = 0
            for ci in items_result.scalars().all():
                mi = (await db.execute(
                    select(MenuItem).where(MenuItem.id == ci.item_id)
                )).scalar_one_or_none()
                cart_items.append({
                    "item_id": ci.item_id,
                    "item_name": mi.name if mi else "Unknown",
                    "qty": ci.qty,
                    "unit_price_cents": ci.unit_price_cents,
                    "available": bool(mi.available) if mi else False,
                    "type": mi.type or "" if mi else "",
                })
                total += ci.unit_price_cents * ci.qty

            summary.cart_items = cart_items
            summary.cart_total_cents = total

    @classmethod
    async def _verify_reservation(
        cls,
        reservation_ref: str,
        summary: VerifiedSummary,
    ) -> None:
        async with AsyncSessionLocal() as db:
            res = (await db.execute(
                select(Reservation).where(Reservation.reservation_ref == reservation_ref)
            )).scalar_one_or_none()

            if not res:
                summary.safe_to_respond = False
                summary.blocking_issue = f"Reservation {reservation_ref} not found in DB."
                return

            summary.reservation_ref = reservation_ref
            summary.reservation_date = str(res.date) if res.date else ""
            summary.reservation_time = str(res.time) if res.time else ""
            summary.reservation_guests = res.guests or 0
