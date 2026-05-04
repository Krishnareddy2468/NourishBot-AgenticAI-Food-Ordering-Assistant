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

import google.generativeai as genai

from app.agent.context_builder import ContextSnapshot
from app.agent.verifier import VerifiedSummary
from app.agent.persona import (
    detect_language, is_off_topic, OFF_TOPIC_REPLY,
    tone_for_context, slot_question, get_cta, KitchenSignal,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class Responder:
    _model: Any = None

    @classmethod
    def _get_model(cls) -> Any:
        if cls._model is None:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            cls._model = genai.GenerativeModel(
                model_name=settings.GEMINI_PRIMARY,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=400,
                ),
            )
        return cls._model

    @classmethod
    async def respond(
        cls,
        summary: VerifiedSummary,
        ctx: ContextSnapshot,
        original_message: str = "",
    ) -> str:
        # ── Food-first boundary (§3.5) ────────────────────────────────────────
        if is_off_topic(original_message):
            return OFF_TOPIC_REPLY

        # ── Detect / update language from this message ────────────────────────
        detected_lang = detect_language(original_message, ctx.last_turns)
        if detected_lang != "en":
            ctx.language = detected_lang  # override for this turn's reply

        prompt = _build_responder_prompt(summary, ctx, original_message)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: cls._get_model().generate_content(prompt),
            )
            text = (response.text or "").strip()
            if not text:
                return _fallback_response(summary, ctx)
            return text
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

    # Language-aware CTA suggestions
    cta_confirm    = get_cta("confirm", lang)
    cta_order_more = get_cta("order_more", lang)
    cta_help       = get_cta("help", lang)

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
Boundary   : Food, orders, reservations ONLY. Deflect anything else with humour.
{upsell_note}

=== HARD RULES ===
1. NEVER invent items, prices, order refs, or ETAs — only use facts below.
2. ALWAYS end with a CTA or question (examples: "{cta_confirm}" / "{cta_order_more}" / "{cta_help}").
3. 2–4 lines normally; full list only for menu/order summaries.
4. Ask for only ONE missing slot at a time.
5. If kitchen_signal is VERY_BUSY or FULL → include the eta_note verbatim.
6. If alternatives present → suggest them naturally, not as a list dump.
7. If radius_exceeded → apologise, offer pickup/dine-in — never just say "no".
{live_block}

=== VERIFIED FACTS ===
{facts}

=== RESPONSE GUIDELINES BY SCENARIO ===
• order_ref set           → Confirm order: ref, items, total, ETA, type. End upbeat.
• errors non-empty        → Explain gently. Offer the fix. Include alternatives if present.
• pending_slots non-empty → Show any current cart briefly, then ask for the FIRST missing slot only. Do NOT ask for order confirmation yet.
• cart non-empty, no order → Show cart summary. Ask "{cta_confirm}" only when pending_slots is empty.
• reservation_ref set     → Confirm date, time, guests, ref.
• tracking set            → Status update conversationally.
• menu_items non-empty    → Scannable list (name · type · price). Max 8 items shown inline.
• blocking set            → Explain blocker, suggest next step.

=== CUSTOMER'S ORIGINAL MESSAGE ===
{original_message}

Write ONLY the reply — no labels, no preamble, no JSON."""


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
        return slot_question(summary.pending_slots[0], ctx.language)

    if summary.alternatives:
        alt_names = " / ".join(a["name"] for a in summary.alternatives[:3])
        return f"That item's not available right now 😔 — how about {alt_names} instead?"

    if summary.radius_exceeded:
        return (
            f"Sorry {name}, we don't deliver to that area yet 🏙️ "
            f"— but you're welcome to pick up or book a table with us!"
        )

    return get_cta("help", ctx.language) or \
           f"Got it, {name}! What would you like — menu, order, or table? 😊"
