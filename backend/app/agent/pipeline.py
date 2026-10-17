"""
NourishBot Pipeline — Main Entry Point.

Replaces nourishbot_agent.py. Implements the 5-stage architecture:

  [1] ContextBuilder  — DB read, full snapshot
  [2] Planner (LLM)  — JSON plan: goal, slots, proposed_actions
  [3] Executor        — Deterministic tool runner, commits to DB
  [4] Verifier        — Re-reads DB, validates totals + availability
  [5] Responder (LLM) — Converts facts → friendly reply

Core rule: LLM never writes to DB. It proposes; deterministic code commits.
Loop bound: max settings.AGENT_MAX_ITERATIONS tool calls per turn.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from app.agent.context_builder import build_context, save_pipeline_turn, ContextSnapshot
from app.agent.planner import Planner, PlannerOutput
from app.agent.executor import Executor, ExecutionResult
from app.agent.verifier import Verifier, VerifiedSummary
from app.agent.responder import Responder
from app.agent.state_machine import StateMachine, BotState
from app.core.config import settings
from app.agent.order_confirmation import is_explicit_confirmation
from app.agent.privacy_service import handle_customer_account_command

logger = logging.getLogger(__name__)


async def process_message(
    message: str,
    chat_id: str | int,
    user_name: str = "",
    channel_type: str = "TELEGRAM",
    planning_context: str | None = None,
) -> str:
    """
    Full pipeline for one incoming Telegram or WhatsApp message.
    Returns the assistant reply string.
    """
    account_reply = await handle_customer_account_command(
        message,
        chat_id=chat_id,
        channel_type=channel_type,
    )
    if account_reply is not None:
        return account_reply

    # ── Stage 1: Context ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    ctx = await build_context(chat_id, user_name, channel_type=channel_type)
    logger.info(
        "Pipeline[chat=%s] ctx built — customer=%r cart=%d orders=%d kitchen_load=%d (stage=%s duration_ms=%d)",
        chat_id,
        ctx.customer_name or "new",
        len(ctx.cart),
        len(ctx.active_orders),
        ctx.kitchen_load,
        "context",
        int((time.monotonic() - t0) * 1000),
    )

    # ── Stage 2: Planner ──────────────────────────────────────────────────────
    # If we can deterministically extract slot values (phone, name, address),
    # inject them as a hint so the planner LLM doesn't have to guess.
    t1 = time.monotonic()
    slot_hint = _extract_slot_hint(message, ctx)
    if slot_hint:
        ctx.pending_slots = slot_hint.get("remaining_slots", ctx.pending_slots)

    planner_message = message
    if planning_context:
        planner_message += (
            "\n\nVERIFIED SAVED-PLAN CONTEXT (use as order details, not as final authorization):\n"
            f"{planning_context}"
        )
    plan = await Planner.plan(planner_message, ctx)

    # If we extracted safe slot values, prepend them so the executor saves
    # them before anything else.
    if slot_hint and slot_hint.get("actions"):
        plan.proposed_actions = slot_hint["actions"] + plan.proposed_actions
        plan.missing_slots = slot_hint.get("remaining_slots", plan.missing_slots)

    plan = _guard_order_plan(plan, ctx, message)
    logger.info(
        "Pipeline[chat=%s] plan — goal=%s missing=%s actions=%d confirm=%s (stage=%s duration_ms=%d)",
        chat_id, plan.goal, plan.missing_slots,
        len(plan.proposed_actions), plan.requires_confirmation,
        "planner",
        int((time.monotonic() - t1) * 1000),
    )

    # ── Short-circuit: only missing slots, no executable actions ─────────────
    if plan.missing_slots and not plan.proposed_actions:
        summary = VerifiedSummary(
            goal=plan.goal,
            intent_summary=plan.user_intent_summary,
            pending_slots=plan.missing_slots,
        )
        reply = await Responder.respond(summary, ctx, message)
        await _persist(
            chat_id,
            message,
            reply,
            plan,
            ctx,
            next_goal=plan.goal,
            next_slots=plan.missing_slots,
            upsell_count=ctx.upsell_count,
            next_state=ctx.current_state,
            order_type=ctx.order_type,
            delivery_address=ctx.delivery_address,
            pending_payment_order_id=ctx.pending_payment_order_id,
            pending_order_ref=ctx.pending_order_ref,
            scheduled_for=ctx.scheduled_for,
        )
        return reply

    # ── Stage 3: Executor + Stage 4: Verifier (loop up to max iterations) ────
    t2 = time.monotonic()
    exec_result = ExecutionResult()
    summary = VerifiedSummary()
    remaining_actions = list(plan.proposed_actions)
    iteration = 0

    while remaining_actions and iteration < settings.AGENT_MAX_ITERATIONS:
        # Batch the remaining actions (executor handles them sequentially)
        exec_result = await Executor.run(remaining_actions, ctx)
        remaining_actions = []  # all consumed in one Executor.run call

        summary = await Verifier.verify(
            exec_result,
            ctx,
            goal=plan.goal,
            intent_summary=plan.user_intent_summary,
            pending_slots=plan.missing_slots,
        )

        # If verifier signals unsafe, log and break — Responder will handle it
        if not summary.safe_to_respond:
            logger.error(
                "Pipeline[chat=%s] Verifier blocked: %s",
                chat_id, summary.blocking_issue,
            )
            break

        iteration += 1

    logger.info(
        "Pipeline[chat=%s] exec done — committed=%d rejected=%d order=%s (stage=%s duration_ms=%d)",
        chat_id,
        len(exec_result.committed),
        len(exec_result.rejected),
        exec_result.order_ref or "—",
        "executor+verifier",
        int((time.monotonic() - t2) * 1000),
    )

    # ── State machine transitions ─────────────────────────────────────────────
    next_state = ctx.current_state
    for action in exec_result.committed:
        next_state = StateMachine.next_from_tool(
            next_state, action.tool, action.data
        )
    logger.info(
        "Pipeline[chat=%s] state: %s → %s",
        chat_id, ctx.current_state.value, next_state.value,
    )

    # Update upsell count when order is placed
    upsell_count = ctx.upsell_count
    if exec_result.order_ref:
        upsell_count = min(upsell_count + 1, ctx.policy.max_upsells_per_session)

    # ── Stage 5: Responder ────────────────────────────────────────────────────
    t3 = time.monotonic()
    reply = await Responder.respond(summary, ctx, message)
    logger.info(
        "Pipeline[chat=%s] responder done (stage=%s duration_ms=%d)",
        chat_id, "responder", int((time.monotonic() - t3) * 1000),
    )

    # ── Persist turn + state ──────────────────────────────────────────────────
    next_goal = plan.goal if not StateMachine.is_terminal(next_state) else None
    next_slots = summary.pending_slots if not exec_result.order_ref else []

    # Carry forward order_type / delivery_address if collected this turn
    order_type = ctx.order_type
    delivery_address = ctx.delivery_address
    pending_payment_order_id = ctx.pending_payment_order_id
    pending_order_ref = ctx.pending_order_ref
    scheduled_for = ctx.scheduled_for
    pending_confirmation_id = ctx.pending_confirmation_id

    for action in exec_result.committed:
        if action.tool == "set_order_type":
            order_type = action.data.get("order_type", order_type)
        elif action.tool == "set_schedule":
            scheduled_for = action.data.get("scheduled_for", scheduled_for)
        elif action.tool == "prepare_order_confirmation":
            pending_confirmation_id = action.data.get("confirmation_id", pending_confirmation_id)
        elif action.tool == "place_order":
            pending_confirmation_id = None
            pending_order_ref = action.data.get("order_ref", pending_order_ref)
        elif action.tool == "set_delivery_address":
            delivery_address = action.data.get("address", delivery_address)
        elif action.tool == "create_payment_intent":
            pending_payment_order_id = action.data.get("razorpay_order_id", pending_payment_order_id)
            pending_order_ref = action.data.get("order_ref", pending_order_ref)
        elif action.tool in ("check_payment_status",) and \
                (action.data.get("status") or "").upper() in ("CAPTURED", "AUTHORIZED"):
            pending_payment_order_id = None  # clear after capture

    if next_slots and next_state == BotState.AWAITING_CONFIRMATION:
        next_state = BotState.COLLECTING_DETAILS

    await _persist(
        chat_id, message, reply, plan, ctx,
        next_goal=next_goal,
        next_slots=next_slots,
        upsell_count=upsell_count,
        next_state=next_state,
        order_type=order_type,
        delivery_address=delivery_address,
        pending_payment_order_id=pending_payment_order_id,
        pending_order_ref=pending_order_ref,
        scheduled_for=scheduled_for,
        pending_confirmation_id=pending_confirmation_id,
    )

    return reply


async def _persist(
    chat_id: str | int,
    user_message: str,
    reply: str,
    plan: PlannerOutput,
    ctx: ContextSnapshot,
    next_goal: Optional[str] = None,
    next_slots: Optional[list[str]] = None,
    upsell_count: int = 0,
    next_state: BotState = BotState.IDLE,
    order_type: Optional[str] = None,
    delivery_address: str = "",
    pending_payment_order_id: Optional[str] = None,
    pending_order_ref: Optional[str] = None,
    scheduled_for: Optional[str] = None,
    pending_confirmation_id: Optional[str] = None,
) -> None:
    try:
        await save_pipeline_turn(
            chat_id=chat_id,
            user_message=user_message,
            assistant_reply=reply,
            pending_goal=next_goal,
            pending_slots=next_slots or [],
            upsell_count=upsell_count,
            current_state=next_state.value,
            order_type=order_type,
            delivery_address=delivery_address or "",
            pending_payment_order_id=pending_payment_order_id,
            pending_order_ref=pending_order_ref,
            scheduled_for=scheduled_for,
            pending_confirmation_id=pending_confirmation_id,
            channel_type=ctx.channel_type,
        )
    except Exception as e:
        logger.error("Pipeline: failed to persist turn for chat=%s: %s", chat_id, e)


def _extract_slot_hint(message: str, ctx: ContextSnapshot) -> dict | None:
    """
    Lightweight slot extractor — does NOT bypass the planner.
    Finds phone/name/address in short replies and injects them as
    proposed actions before the LLM plan. The planner still runs and
    decides the goal; the responder LLM still composes the reply.
    """
    pending = list(ctx.pending_slots or [])
    if not pending:
        return None

    text = (message or "").strip()
    if not text:
        return None

    actions: list[dict] = []
    consumed: set[str] = set()

    # Phone extraction
    phone = _extract_phone(text)
    if "customer_phone" in pending and phone:
        actions.append({"tool": "update_customer", "args": {"phone": phone}})
        consumed.add("customer_phone")

    # Name heuristics (short text that looks like a name)
    if "customer_name" in pending and not phone:
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*", text)
        if 1 <= len(words) <= 4 and not _is_action_word(text):
            actions.append({"tool": "update_customer", "args": {"name": text.strip()}})
            consumed.add("customer_name")

    # Order type extraction
    if "order_type" in pending:
        order_type = _extract_order_type(text)
        if order_type:
            actions.append({"tool": "set_order_type", "args": {"order_type": order_type}})
            consumed.add("order_type")

    # Deterministic ISO-8601 schedule extraction; the planner handles natural
    # phrases such as "tomorrow at 1 PM" using the current timestamp context.
    if "scheduled_for" in pending:
        iso_match = re.search(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?",
            text,
        )
        if iso_match:
            actions.append({"tool": "set_schedule", "args": {"scheduled_for": iso_match.group(0)}})
            consumed.add("scheduled_for")

    # Address extraction (long text with commas, not a name)
    if "delivery_address" in pending:
        cleaned = re.sub(r"\+?\d[\d\s().-]{6,}\d", "", text).strip(" \t\r\n,;:-")
        if len(cleaned) >= 10 and not _is_action_word(cleaned):
            if not any(a.get("tool") == "set_order_type" for a in actions) and not ctx.order_type:
                actions.append({"tool": "set_order_type", "args": {"order_type": "DELIVERY"}})
                consumed.add("order_type")
            actions.append({"tool": "set_delivery_address", "args": {"address": cleaned}})
            consumed.add("delivery_address")

    if not actions:
        return None

    remaining = [slot for slot in pending if slot not in consumed]
    return {"actions": actions, "remaining_slots": remaining}


def _is_action_word(text: str) -> bool:
    lowered = (text or "").strip().lower()
    blocked = {
        "order", "place", "confirm", "yes", "no", "delivery", "pickup",
        "drive", "through", "thru", "curbside",
        "dine", "menu", "cart", "phone", "number", "cancel", "reserve",
        "book", "help", "reset", "info", "specials",
    }
    return any(w in blocked for w in re.findall(r"\w+", lowered))


def _guard_order_plan(plan: PlannerOutput, ctx: ContextSnapshot, message: str = "") -> PlannerOutput:
    """
    Deterministic safety pass over the LLM plan.

    The planner may correctly add an item to cart but still ask for order
    confirmation too early. This keeps DB writes agentic while ensuring
    required slots are collected before `place_order`.
    """
    blocked_tools = {
        "create_payment_intent",
        "check_payment_status",
        "make_reservation",
    }
    proposed_types = {
        str((proposed.get("args") or {}).get("order_type") or "").upper()
        for proposed in (plan.proposed_actions or [])
        if proposed.get("tool") in ("set_order_type", "place_order")
    }
    allowed_types = ("PICKUP", "DRIVE_THRU", "DELIVERY")
    preferred_type = next(
        (order_type for order_type in ("DELIVERY", "DRIVE_THRU", "PICKUP") if order_type in proposed_types),
        ctx.order_type if ctx.order_type in allowed_types else "PICKUP",
    )
    actions = []
    for proposed in list(plan.proposed_actions or []):
        if proposed.get("tool") in blocked_tools:
            continue
        action = {**proposed, "args": dict(proposed.get("args") or {})}
        if action.get("tool") in ("set_order_type", "place_order"):
            requested_type = str(action["args"].get("order_type") or preferred_type).upper()
            action["args"]["order_type"] = (
                requested_type if requested_type in allowed_types else "PICKUP"
            )
            if action["args"]["order_type"] != "DELIVERY":
                action["args"].pop("address", None)
        elif action.get("tool") == "update_customer":
            action["args"].pop("address", None)
        actions.append(action)

    safe_missing = [
        slot for slot in (plan.missing_slots or [])
        if slot not in ("address", "order_type")
    ]
    action_tools = [a.get("tool") for a in actions]
    has_order_intent = (
        plan.goal in ("ORDER_ONLINE", "TAKEAWAY") and bool(ctx.cart)
    ) or any(
        t in {
            "add_to_cart",
            "update_cart_item",
            "remove_from_cart",
            "view_cart",
            "set_order_type",
            "place_order",
        }
        for t in action_tools
    )
    if not has_order_intent:
        return PlannerOutput(
            goal=plan.goal,
            missing_slots=safe_missing,
            constraints=plan.constraints,
            proposed_actions=actions,
            user_intent_summary=plan.user_intent_summary,
            requires_confirmation=plan.requires_confirmation,
        )

    missing = safe_missing

    def add_missing(slot: str) -> None:
        if slot not in missing:
            missing.append(slot)

    customer_action_args = {}
    for action in actions:
        if action.get("tool") == "update_customer":
            customer_action_args.update(action.get("args") or {})

    if not ctx.customer_name and not customer_action_args.get("name"):
        add_missing("customer_name")
    if not ctx.customer_phone and not customer_action_args.get("phone"):
        add_missing("customer_phone")

    schedule_value = ctx.scheduled_for
    for action in actions:
        if action.get("tool") == "set_schedule":
            schedule_value = (action.get("args") or {}).get("scheduled_for") or schedule_value
        elif action.get("tool") == "place_order":
            schedule_value = (action.get("args") or {}).get("scheduled_for") or schedule_value
    if not schedule_value:
        add_missing("scheduled_for")

    address_value = ctx.delivery_address
    for action in actions:
        if action.get("tool") == "set_delivery_address":
            address_value = (action.get("args") or {}).get("address") or address_value
        elif action.get("tool") == "place_order":
            address_value = (action.get("args") or {}).get("address") or address_value

    if preferred_type == "DELIVERY" and not address_value:
        add_missing("delivery_address")
    elif preferred_type == "DELIVERY":
        missing = [slot for slot in missing if slot != "delivery_address"]

    # Only home delivery needs an address; clear stale address slots for collection orders.
    missing = [
        slot for slot in missing
        if slot not in ("address", "order_type")
        and not (slot == "delivery_address" and preferred_type != "DELIVERY")
    ]

    if missing:
        actions = [a for a in actions if a.get("tool") != "place_order"]
    else:
        has_confirmation = (
            ctx.current_state == BotState.AWAITING_CONFIRMATION
            and bool(ctx.pending_confirmation_id)
            and bool(ctx.confirmation_snapshot)
            and is_explicit_confirmation(message)
        )
        place_actions = [a for a in actions if a.get("tool") == "place_order"]
        if has_confirmation and not place_actions:
            snapshot = dict(ctx.confirmation_snapshot or {})
            actions.append({
                "tool": "place_order",
                "args": {
                    "order_type": snapshot.get("order_type"),
                    "address": snapshot.get("delivery_address"),
                    "scheduled_for": snapshot.get("scheduled_for"),
                    "provider": snapshot.get("provider"),
                    "confirmation_id": ctx.pending_confirmation_id,
                },
            })
        for action in actions:
            if action.get("tool") == "place_order":
                if has_confirmation:
                    snapshot = dict(ctx.confirmation_snapshot or {})
                    action["args"]["order_type"] = snapshot.get("order_type")
                    action["args"]["scheduled_for"] = snapshot.get("scheduled_for")
                    action["args"]["provider"] = snapshot.get("provider")
                    if snapshot.get("order_type") == "DELIVERY":
                        action["args"]["address"] = snapshot.get("delivery_address")
                    else:
                        action["args"].pop("address", None)
                    action["args"]["confirmation_id"] = ctx.pending_confirmation_id
                else:
                    action["args"]["scheduled_for"] = schedule_value
                    if preferred_type == "DELIVERY":
                        action["args"]["address"] = address_value
                    action["tool"] = "prepare_order_confirmation"

    return PlannerOutput(
        goal=plan.goal if plan.goal in ("ORDER_ONLINE", "TAKEAWAY") else "ORDER_ONLINE",
        missing_slots=missing,
        constraints=plan.constraints,
        proposed_actions=actions,
        user_intent_summary=plan.user_intent_summary,
        requires_confirmation=not missing and not is_explicit_confirmation(message),
    )


def _extract_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    if 8 <= len(digits) <= 15:
        return digits
    return ""


def _extract_order_type(text: str) -> str:
    lowered = (text or "").strip().lower()
    if any(term in lowered for term in ("drive through", "drive-through", "drive thru", "drive-thru", "curbside")):
        return "DRIVE_THRU"
    if any(term in lowered for term in ("home delivery", "deliver to", "delivery", "delivered")) \
            or re.search(r"\bdeliver(?:ed|ing)?\b", lowered):
        return "DELIVERY"
    if "pickup" in lowered or "pick up" in lowered or "takeaway" in lowered or "take away" in lowered:
        return "PICKUP"
    return ""
