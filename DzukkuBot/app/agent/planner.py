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

# Compact tool catalogue for the planner
TOOL_CATALOGUE = """
AVAILABLE TOOLS:
  search_menu         {query, filter_type?, filter_category?}
  get_menu            {filter_type?, filter_category?}
  get_item_details    {item_id?, item_name?}
  get_restaurant_info {}
  get_kitchen_eta     {}
  add_to_cart         {items: [{name, qty}]}
  update_cart_item    {item_name, qty}
  remove_from_cart    {item_name}
  view_cart           {}
  clear_cart          {}
  set_order_type      {order_type: DELIVERY|PICKUP|DINE_IN}
  set_delivery_address {address}
  update_customer     {name?, phone?, address?, language?}
  place_order         {order_type, address?, notes?}
  track_order         {order_ref}
  cancel_order        {order_ref}
  create_payment_intent {order_ref, provider?}
  check_payment_status {order_ref?, payment_id?}
  make_reservation    {date, time, guests, special_request?}
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

    # State-specific instruction injected into rules
    state_hint = _state_hint(ctx.current_state, ctx)

    return f"""You are the PLANNER for Dzukku Restaurant. Read the context + message. Output ONE JSON object.

CONTEXT:
State: {ctx.current_state.value} | Time: {ctx.time_of_day} ({ctx.now.strftime('%I:%M %p')}) | Open: {ctx.is_open}
Customer: {ctx.customer_name or "new"} | Phone: {ctx.customer_phone or "—"} | Order type: {ctx.order_type or "—"} | Address: {ctx.delivery_address or "—"}
Cart: {ctx.cart_summary}
Orders: {active_order_text} | Kitchen: {ctx.kitchen_load} orders

MENU (top 40):
{menu_text}

TOOLS:
{TOOL_CATALOGUE}

OUTPUT JSON:
{{"goal":"ORDER_ONLINE|DINE_IN|TAKEAWAY|RESERVATION|TRACK_ORDER|CANCEL_ORDER|SUPPORT|MENU_BROWSE","missing_slots":[],"constraints":{{}},"proposed_actions":[{{"tool":"...","args":{{...}}}}],"user_intent_summary":"...","requires_confirmation":false}}

RULES:
1. Max {settings.AGENT_MAX_ITERATIONS} proposed_actions.
2. Missing name/phone → add to missing_slots, don't place_order.
3. DELIVERY needs address → add to missing_slots if absent.
4. place_order ONLY when: cart non-empty + all slots known + customer confirmed.
5. Never fabricate items or prices.
{state_hint}

HISTORY:
{_format_history(ctx.last_turns)}

MESSAGE: {message}

JSON only:"""


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
                    max_output_tokens=2048,
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
