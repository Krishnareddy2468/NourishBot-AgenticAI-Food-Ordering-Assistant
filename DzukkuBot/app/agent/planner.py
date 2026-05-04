"""
Planner — Stage 2 of the Dzukku Pipeline.

One LLM call (Gemini, JSON mode) that reads the ContextSnapshot +
user message and outputs a structured PlannerOutput:

  {
    "goal":                "ORDER_ONLINE" | "DINE_IN" | "TAKEAWAY" |
                           "RESERVATION" | "TRACK_ORDER" | "CANCEL_ORDER" |
                           "SUPPORT" | "MENU_BROWSE",
    "missing_slots":       ["phone", "address", ...],
    "constraints":         {"veg": true, "budget_max": 500, "allergy": ""},
    "proposed_actions":    [{"tool": "...", "args": {...}}, ...],
    "user_intent_summary": "wants 2x Biryani delivered to Kondapur",
    "requires_confirmation": false
  }

The LLM ONLY plans. It does not write to the DB.
Max proposed_actions = settings.AGENT_MAX_ITERATIONS (6).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import google.generativeai as genai

from app.agent.context_builder import ContextSnapshot
from app.agent.state_machine import BotState
from app.core.config import settings

logger = logging.getLogger(__name__)

# Goal constants
GOALS = {
    "ORDER_ONLINE",
    "DINE_IN",
    "TAKEAWAY",
    "RESERVATION",
    "TRACK_ORDER",
    "CANCEL_ORDER",
    "SUPPORT",
    "MENU_BROWSE",
}

# Full tool catalogue exposed to the planner
TOOL_CATALOGUE = """
AVAILABLE TOOLS (executor runs these — never fake results):

── MENU DISCOVERY ───────────────────────────────────────────────────────────
  search_menu         args: {query, filter_type?: Veg|Non-Veg|EGG|VEGAN, filter_category?}
  get_menu            args: {filter_type?: All|Veg|Non-Veg, filter_category?}
  get_item_details    args: {item_id?, item_name?}
  get_restaurant_info args: {}
  get_kitchen_eta     args: {}

── CART ─────────────────────────────────────────────────────────────────────
  add_to_cart         args: {items: [{name, qty}]}
  update_cart_item    args: {item_name, qty}          (qty=0 to remove)
  remove_from_cart    args: {item_name}
  view_cart           args: {}
  clear_cart          args: {}

── ORDER FLOW ───────────────────────────────────────────────────────────────
  set_order_type      args: {order_type: DELIVERY|PICKUP|DINE_IN}
  set_delivery_address args: {address}               (required for DELIVERY)
  update_customer     args: {name?, phone?, address?, language?}
  place_order         args: {order_type: DELIVERY|PICKUP|DINE_IN, address?, notes?}
  track_order         args: {order_ref}
  cancel_order        args: {order_ref}

── PAYMENT ──────────────────────────────────────────────────────────────────
  create_payment_intent args: {order_ref, provider?: razorpay}
  check_payment_status  args: {order_ref?, payment_id?}

── RESERVATION ──────────────────────────────────────────────────────────────
  make_reservation    args: {date, time, guests, special_request?}

── DINE-IN (waiter / staff) ─────────────────────────────────────────────────
  open_table_session  args: {table_id, guests, waiter_user_id?}
  add_table_order     args: {table_session_id, items: [{name, qty}]}
  close_table_session args: {table_session_id}
  generate_invoice    args: {entity_id, entity_type: TABLE_SESSION|ORDER}

── RESTAURANT OPS (staff only) ──────────────────────────────────────────────
  set_item_availability args: {item_id?, item_name?, available: bool}
  update_stock          args: {item_id?, item_name?, delta?, set_to?}
  update_order_status   args: {order_ref?, order_id?, status}
  assign_driver         args: {order_ref, driver_id}
  update_delivery_status args: {delivery_id?, order_ref?, status}
"""

# Slots required per goal
REQUIRED_SLOTS: dict[str, list[str]] = {
    "ORDER_ONLINE":  ["customer_name", "customer_phone"],
    "DINE_IN":       ["customer_name", "customer_phone"],
    "TAKEAWAY":      ["customer_name", "customer_phone"],
    "RESERVATION":   ["customer_name", "customer_phone", "date", "time", "guests"],
    "TRACK_ORDER":   ["order_ref"],
    "CANCEL_ORDER":  ["order_ref"],
    "SUPPORT":       [],
    "MENU_BROWSE":   [],
}


@dataclass
class PlannerOutput:
    goal: str = "SUPPORT"
    missing_slots: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    proposed_actions: list[dict] = field(default_factory=list)
    user_intent_summary: str = ""
    requires_confirmation: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "PlannerOutput":
        actions = [a for a in (d.get("proposed_actions") or []) if isinstance(a, dict)]
        actions = actions[:settings.AGENT_MAX_ITERATIONS]  # hard cap
        return cls(
            goal=d.get("goal", "SUPPORT") if d.get("goal") in GOALS else "SUPPORT",
            missing_slots=list(d.get("missing_slots") or []),
            constraints=dict(d.get("constraints") or {}),
            proposed_actions=actions,
            user_intent_summary=str(d.get("user_intent_summary") or ""),
            requires_confirmation=bool(d.get("requires_confirmation", False)),
        )

    @classmethod
    def fallback(cls, reason: str = "") -> "PlannerOutput":
        return cls(goal="SUPPORT", user_intent_summary=reason or "planner_fallback")


# ── Planner prompt ────────────────────────────────────────────────────────────

def _build_planner_prompt(message: str, ctx: ContextSnapshot) -> str:
    menu_lines = [
        f"  {i['name']} ({i['type']}) ₹{int(i['price'])}"
        for i in ctx.menu_snapshot[:40]
    ]
    menu_text = "\n".join(menu_lines) or "  (no items available)"
    active_order_text = ctx.active_order_summary

    slots_known = []
    if ctx.customer_name:       slots_known.append("customer_name")
    if ctx.customer_phone:      slots_known.append("customer_phone")
    if ctx.order_type:          slots_known.append(f"order_type={ctx.order_type}")
    if ctx.delivery_address:    slots_known.append("delivery_address")

    # State-specific instruction injected into rules
    state_hint = _state_hint(ctx.current_state, ctx)

    return f"""You are the PLANNER for Dzukku Restaurant's AI agent.

=== YOUR JOB ===
Read the customer message + context. Output ONE JSON object describing what to do.
You NEVER write to the database. You ONLY plan.

=== CONTEXT ===
Bot state     : {ctx.current_state.value}
Time          : {ctx.time_of_day} ({ctx.now.strftime('%I:%M %p')})
Restaurant    : {"OPEN" if ctx.is_open else "CLOSED"}
Customer name : {ctx.customer_name or "(unknown)"}
Customer phone: {ctx.customer_phone or "(unknown)"}
Order type    : {ctx.order_type or "(not set)"}
Delivery addr : {ctx.delivery_address or "(not set)"}
Slots known   : {', '.join(slots_known) or 'none'}
Cart          :
{ctx.cart_summary}
Active orders : {active_order_text}
Pending payment: {ctx.pending_order_ref or "none"}
Kitchen load  : {ctx.kitchen_load} orders in prep

=== AVAILABLE MENU ===
{menu_text}

{TOOL_CATALOGUE}

=== OUTPUT FORMAT (strict JSON, no extra keys) ===
{{
  "goal": "<ORDER_ONLINE|DINE_IN|TAKEAWAY|RESERVATION|TRACK_ORDER|CANCEL_ORDER|SUPPORT|MENU_BROWSE>",
  "missing_slots": ["list of slot names still needed"],
  "constraints": {{"veg": true/false, "budget_max": 0, "allergy": ""}},
  "proposed_actions": [
    {{"tool": "<tool_name>", "args": {{...}}}}
  ],
  "user_intent_summary": "one line summary",
  "requires_confirmation": true/false
}}

=== RULES ===
1. proposed_actions max {settings.AGENT_MAX_ITERATIONS} entries.
2. If cart is empty and customer wants to order → include add_to_cart first.
3. Missing name/phone → add to missing_slots, DO NOT call place_order.
4. DELIVERY orders also require delivery_address — add to missing_slots if absent.
5. place_order ONLY when: cart non-empty + all required slots known + requires_confirmation=true + customer explicitly confirmed ("yes", "place it", "confirm").
6. cancel_order ONLY on explicit cancel request with order ref.
7. MENU_BROWSE/SUPPORT → proposed_actions may be empty or just search_menu/get_menu.
8. requires_confirmation = true whenever place_order or cancel_order is in proposed_actions.
9. Never fabricate item names or prices — only use menu items listed above.
10. If bot state is AWAITING_PAYMENT → suggest check_payment_status or create_payment_intent.
{state_hint}

=== CONVERSATION HISTORY (last turns) ===
{_format_history(ctx.last_turns)}

=== CUSTOMER MESSAGE ===
{message}

Respond with ONLY the JSON object, no markdown, no explanation."""


def _build_minimal_prompt(message: str, ctx: ContextSnapshot) -> str:
    """Stripped-down prompt used as retry when the full prompt causes parse errors."""
    cart_text = ctx.cart_summary or "(empty)"
    known = []
    if ctx.customer_name:  known.append(f"name={ctx.customer_name}")
    if ctx.customer_phone: known.append(f"phone={ctx.customer_phone}")
    return f"""You are a restaurant order assistant planner. Output ONLY a JSON object.

Customer message: {message}
Cart: {cart_text}
Known: {', '.join(known) or 'nothing'}
Restaurant: {"OPEN" if ctx.is_open else "CLOSED"}

Output this exact JSON (fill in values):
{{"goal":"ORDER_ONLINE","missing_slots":[],"constraints":{{}},"proposed_actions":[{{"tool":"get_menu","args":{{}}}}],"user_intent_summary":"","requires_confirmation":false}}

Valid goals: ORDER_ONLINE DINE_IN TAKEAWAY RESERVATION TRACK_ORDER CANCEL_ORDER SUPPORT MENU_BROWSE
Return ONLY the JSON, nothing else."""


def _state_hint(state: BotState, ctx: ContextSnapshot) -> str:
    """Inject state-specific rule as an extra hint line."""
    hints = {
        BotState.AWAITING_PAYMENT:
            "11. State=AWAITING_PAYMENT: primary action should be check_payment_status or create_payment_intent.",
        BotState.BUILDING_CART:
            "11. State=BUILDING_CART: after adding items, suggest set_order_type if not set.",
        BotState.COLLECTING_DETAILS:
            f"11. State=COLLECTING_DETAILS: collect missing slots ({', '.join(ctx.pending_slots)}) before place_order.",
        BotState.AWAITING_CONFIRMATION:
            "11. State=AWAITING_CONFIRMATION: if customer confirms → place_order; if they change order → update cart.",
        BotState.ORDER_PLACED:
            "11. State=ORDER_PLACED: order is placed; use track_order for status queries.",
    }
    return hints.get(state, "")


def _format_history(turns: list[dict]) -> str:
    if not turns:
        return "(none)"
    return "\n".join(
        f"{'User' if t['role'] == 'user' else 'Bot'}: {t['content']}"
        for t in turns
    )


# ── Planner class ─────────────────────────────────────────────────────────────

class Planner:
    _model: Any = None

    @classmethod
    def _get_model(cls) -> Any:
        if cls._model is None:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            cls._model = genai.GenerativeModel(
                model_name=settings.GEMINI_PRIMARY,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
        return cls._model

    @classmethod
    async def plan(cls, message: str, ctx: ContextSnapshot) -> PlannerOutput:
        """Call Gemini in JSON mode and return a validated PlannerOutput."""
        import asyncio
        loop = asyncio.get_running_loop()

        for attempt in range(2):
            prompt = (
                _build_planner_prompt(message, ctx)
                if attempt == 0
                else _build_minimal_prompt(message, ctx)
            )
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda p=prompt: cls._get_model().generate_content(p),
                )
                raw = (response.text or "").strip()
                # Strip markdown code fences if model adds them
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                data = json.loads(raw)
                out = PlannerOutput.from_dict(data)
                logger.info(
                    "Planner[chat=%s] goal=%s actions=%d intent=%r",
                    ctx.chat_id, out.goal, len(out.proposed_actions), out.user_intent_summary,
                )
                return out
            except json.JSONDecodeError as e:
                logger.warning(
                    "Planner JSON parse error attempt=%d (chat=%s): %s | raw=%r",
                    attempt + 1, ctx.chat_id, e, (raw or "")[:120],
                )
                if attempt == 1:
                    return PlannerOutput.fallback(f"json_parse_error: {e}")
            except Exception as e:
                logger.error("Planner call failed (chat=%s): %s", ctx.chat_id, e, exc_info=True)
                return PlannerOutput.fallback(str(e))

        return PlannerOutput.fallback("all_attempts_failed")
