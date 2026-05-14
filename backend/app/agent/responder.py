"""
Responder — Stage 5 of the Dzukku Pipeline.

Second (and last) LLM call per turn. Converts the deterministic
VerifiedSummary into a friendly, brand-consistent reply.

Responsibilities:
  - Turn structured facts into natural language.
  - Mirror customer's language (EN / TE+EN / HI+EN code-mix).
  - Always end with a CTA or question.
  - Never invent facts — only narrate what VerifiedSummary contains.
  - Enforce max-upsell policy (1 gentle upsell per conversation).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.agent.context_builder import ContextSnapshot
from app.agent.verifier import VerifiedSummary
from app.agent.persona import (
    detect_language, tone_for_context, KitchenSignal,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class Responder:
    _client: OpenAI | None = None

    @classmethod
    def _get_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return cls._client

    @classmethod
    async def respond(
        cls,
        summary: VerifiedSummary,
        ctx: ContextSnapshot,
        original_message: str = "",
    ) -> str:
        # ── Detect / update language from this message ────────────────────────
        detected_lang = detect_language(original_message, ctx.last_turns)
        if detected_lang != "en":
            ctx.language = detected_lang  # override for this turn's reply

        # Let the LLM handle everything — off-topic, missing slots, etc.
        # The prompt gives it all the context it needs to compose a natural reply.
        prompt = _build_responder_prompt(summary, ctx, original_message)
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            client = cls._get_client()

            for attempt in range(2):
                response = await loop.run_in_executor(
                    None,
                    lambda p=prompt: client.chat.completions.create(
                        model=settings.OPENAI_PRIMARY,
                        messages=[{"role": "user", "content": p}],
                        temperature=0.7,
                        max_tokens=2048,
                    ),
                )
                text = (response.choices[0].message.content or "").strip()

                # Check finish reason for truncation
                finish = response.choices[0].finish_reason or ""
                if finish in ("length", "content_filter"):
                    logger.warning(
                        "Responder truncated/blocked (chat=%s) finish=%s len=%d tail=%r",
                        ctx.chat_id, finish, len(text), text[-60:] if text else "",
                    )
                    if attempt == 0:
                        prompt = _build_minimal_responder_prompt(summary, ctx, original_message)
                        continue

                # If text looks cut off (ends mid-word, no punctuation), try once more
                if text and attempt == 0 and not text[-1:] in ".!?😊😉😄🔥🎉📅\"'" and len(text) > 20:
                    prompt = _build_minimal_responder_prompt(summary, ctx, original_message)
                    continue

                if text:
                    return text
                break

            return _fallback_response(summary, ctx)
        except Exception as e:
            logger.error("Responder failed (chat=%s): %s", ctx.chat_id, e, exc_info=True)
            return _fallback_response(summary, ctx)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_responder_prompt(
    summary: VerifiedSummary,
    ctx: ContextSnapshot,
    original_message: str,
) -> str:
    facts = json.dumps(summary.to_prompt_dict(), ensure_ascii=False, indent=2)
    name = ctx.customer_name.split()[0] if ctx.customer_name else (ctx.user_name or "there")
    lang = ctx.language

    upsell_allowed = ctx.upsell_count < ctx.policy.max_upsells_per_session
    upsell_note = (
        "You MAY include ONE gentle upsell if natural (e.g. 'A Lassi pairs great with that!')."
        if upsell_allowed else
        "Do NOT upsell — limit reached."
    )

    tone = tone_for_context(ctx.time_of_day, ctx.current_state.value)

    # Live-state annotations to inject
    live_state_notes = []
    if summary.kitchen_signal in (KitchenSignal.VERY_BUSY, KitchenSignal.FULL):
        live_state_notes.append(
            f"KITCHEN SIGNAL = {summary.kitchen_signal}: "
            f"Use this ETA note verbatim if quoting prep time: \"{summary.eta_note}\""
        )
    if summary.alternatives:
        alt_names = ", ".join(a["name"] for a in summary.alternatives[:3])
        live_state_notes.append(
            f"ALTERNATIVES AVAILABLE: Customer's item was unavailable. "
            f"Suggest these alternatives naturally: {alt_names}"
        )
    if summary.radius_exceeded:
        live_state_notes.append(
            "RADIUS EXCEEDED: Delivery not available to this address. "
            "Offer pickup or dine-in as alternatives — be apologetic but helpful."
        )

    live_block = (
        "\n=== LIVE STATE NOTES (must influence reply) ===\n"
        + "\n".join(f"  • {n}" for n in live_state_notes)
        if live_state_notes else ""
    )

    return f"""You are Dzukku — restaurant assistant for Dzukku Restaurant ("Where every bite hits different ❤️").

=== PERSONA ===
Name       : Dzukku
Tone       : {tone}
Language   : {lang} — mirror the customer's exact register (EN / TE+EN code-mix / HI+EN code-mix)
Address as : {name}
{upsell_note}

=== RULES ===
1. NEVER invent items, prices, order refs, or ETAs — only use facts below.
2. ALWAYS end with a CTA or question.
3. Keep it concise — 2-4 lines for chat; longer only for menu or order summaries.
4. For off-topic messages (not about food/orders/reservations), deflect with warmth and humor.
5. If pending_slots has contact details missing, ask for them naturally in one question.
6. If kitchen_signal is VERY_BUSY or FULL, mention the eta_note.
{live_block}

=== VERIFIED FACTS ===
{facts}

=== CUSTOMER'S MESSAGE ===
{original_message}

Write ONLY the reply — no labels, no preamble, no JSON."""


def _build_minimal_responder_prompt(
    summary: VerifiedSummary,
    ctx: ContextSnapshot,
    original_message: str,
) -> str:
    """Stripped-down prompt used as retry when the full prompt causes truncation."""
    facts = summary.to_prompt_dict()
    name = ctx.customer_name.split()[0] if ctx.customer_name else (ctx.user_name or "there")

    # Build a concise fact summary instead of full JSON dump
    fact_lines = []
    if facts.get("order_ref"):
        fact_lines.append(f"Order ref: {facts['order_ref']}, total: ₹{facts.get('order_total_inr','?')}, type: {facts.get('order_type','?')}")
    if facts.get("cart"):
        fact_lines.append(f"Cart: {', '.join(c.get('item_name','?') for c in facts['cart'])}, total: ₹{facts.get('cart_total_inr','?')}")
    if facts.get("menu_items"):
        fact_lines.append(f"Menu items ({len(facts['menu_items'])}): " + ", ".join(f"{m.get('name','?')} ₹{m.get('price','?')}" for m in facts['menu_items'][:8]))
    if facts.get("pending_slots"):
        fact_lines.append(f"Missing: {', '.join(facts['pending_slots'])}")
    if facts.get("errors"):
        fact_lines.append(f"Errors: {'; '.join(facts['errors'])}")
    if facts.get("reservation_ref"):
        fact_lines.append(f"Reservation: {facts['reservation_ref']} on {facts.get('reservation_date','?')} at {facts.get('reservation_time','?')} for {facts.get('reservation_guests','?')} guests")
    if not fact_lines:
        fact_lines.append("No specific facts — general assistance needed.")

    return f"""You are Dzukku, a warm restaurant assistant. Reply in 2-3 lines max. End with a question.

Customer: {name} | Message: {original_message}
Facts: {" | ".join(fact_lines)}

Reply:"""


# ── Fallback (no LLM) ─────────────────────────────────────────────────────────

def _fallback_response(summary: VerifiedSummary, ctx: ContextSnapshot) -> str:
    name = ctx.customer_name.split()[0] if ctx.customer_name else "there"

    if summary.blocking_issue:
        return f"Sorry {name}, something went wrong: {summary.blocking_issue} Please try again."

    if summary.rejected_errors:
        return f"Hmm, {summary.rejected_errors[0]} Want to try something else?"

    if summary.order_ref:
        total = round(summary.order_total_cents / 100, 2)
        return (
            f"Order confirmed! 🎉\n"
            f"Ref: {summary.order_ref}\n"
            f"Total: ₹{total:.0f}\n"
            f"ETA: ~{summary.order_eta_seconds // 60 if summary.order_eta_seconds else 20} mins\n"
            f"Anything else, {name}?"
        )

    if summary.reservation_ref:
        return (
            f"Reservation confirmed! 📅\n"
            f"Ref: {summary.reservation_ref}\n"
            f"Date: {summary.reservation_date} at {summary.reservation_time}\n"
            f"Guests: {summary.reservation_guests}\n"
            f"See you then! 😊"
        )

    if summary.pending_slots:
        # Ask for the first missing slot in a natural way
        slot = summary.pending_slots[0]
        labels = {"customer_name": "name", "customer_phone": "phone number",
                  "delivery_address": "delivery address", "order_type": "order type",
                  "date": "preferred date", "time": "preferred time", "guests": "number of guests"}
        label = labels.get(slot, slot)
        return f"Could you share your {label}, {name}? 😊"

    if summary.alternatives:
        alt_names = " / ".join(a["name"] for a in summary.alternatives[:3])
        return f"That item's not available right now 😔 — how about {alt_names} instead?"

    if summary.radius_exceeded:
        return (
            f"Sorry {name}, we don't deliver to that area yet 🏙️ "
            f"— but you're welcome to pick up or book a table with us!"
        )

    return f"Got it, {name}! What would you like — menu, order, or table? 😊"
