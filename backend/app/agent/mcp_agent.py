"""
LangGraph MCP Agent — scheduled provider ordering over chat.
============================================================

    Telegram message
        -> deterministic guards (greeting, reset, simple intents)
        -> LangGraph ReAct agent
            - Google Gemini chat model (bound to MCP tools)
            - Tools come from langchain-mcp-adapters (Zomato + Swiggy)
            - System prompt enforces no-hallucination + concise replies
        -> Final assistant text returned to Telegram

The agent itself is per-message stateless; the rolling conversation history
is replayed into the prompt the same way as the NourishBot orchestrator already
does, so the same PostgreSQL session table works without schema changes.

If MCP is not enabled or the tools fail to load, get_mcp_response() returns
None and the channel reports that the selected provider is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.db.crud import get_session, save_session
from app.db.session import AsyncSessionLocal
from app.db.models import ProviderAttempt
from sqlalchemy import select
from app.agent.order_confirmation import is_explicit_confirmation
from app.providers.base import CapabilityState, ProviderMode, ProviderPolicy
from app.providers.redaction import content_hash, redact_text, redact_value
from app.providers.registry import (
    current_policy,
    is_cancel_tool,
    is_status_tool,
    is_transaction_tool as registry_is_transaction_tool,
    sync_tool_inventory,
    tool_allowed,
)
from app.providers.service import persist_tool_receipts, provider_handoff_url, reconcile_attempt

logger = logging.getLogger(__name__)

def _is_transaction_tool(name: str) -> bool:
    return registry_is_transaction_tool(name)


async def _start_provider_attempt(
    *,
    platform: str,
    channel_type: str,
    chat_id: str | int,
    fingerprint_source: str,
    policy: ProviderPolicy,
) -> tuple[ProviderAttempt | None, str | None]:
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    idempotency_key = hashlib.sha256(
        f"{platform}|{channel_type}|{chat_id}|{fingerprint}".encode("utf-8")
    ).hexdigest()
    async with AsyncSessionLocal() as db:
        existing = (await db.execute(
            select(ProviderAttempt).where(
                ProviderAttempt.provider == platform.upper(),
                ProviderAttempt.idempotency_key == idempotency_key,
            )
        )).scalar_one_or_none()
        if existing:
            if existing.status == "CONFIRMED":
                return existing, "confirmed"
            if existing.status == "UNKNOWN":
                return existing, "reconcile_required"
            if existing.status == "SUBMITTING":
                return existing, "pending"
            existing.attempt_count += 1
            existing.status = "SUBMITTING"
            existing.last_error = None
            existing.reconciliation_required = False
            existing.next_action = None
            existing.submitted_at = datetime.now(timezone.utc)
            existing.completed_at = None
            await db.commit()
            return existing, None

        attempt = ProviderAttempt(
            id=str(uuid.uuid4()),
            restaurant_id=settings.DEFAULT_RESTAURANT_ID,
            provider=platform.upper(),
            channel_type=channel_type.upper(),
            external_chat_id=str(chat_id),
            operation="CONFIRM_ORDER",
            mode=ProviderMode.TRANSACTIONAL.value,
            capability_version_id=policy.version_id,
            tool_schema_hash=policy.tool_schema_hash,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            status="SUBMITTING",
            submitted_at=datetime.now(timezone.utc),
        )
        db.add(attempt)
        await db.commit()
        return attempt, None


async def _finish_provider_attempt(
    attempt_id: str | None,
    *,
    status: str,
    provider_order_id: str | None = None,
    error: str | None = None,
) -> None:
    if not attempt_id:
        return
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id)
        )).scalar_one_or_none()
        if not attempt:
            return
        attempt.status = status
        attempt.provider_order_id = provider_order_id or attempt.provider_order_id
        if provider_order_id:
            receipt = {"provider_order_id": provider_order_id, "status": status}
            attempt.receipt_json = receipt
            attempt.response_hash = content_hash(receipt)
        attempt.last_error = redact_text(error or "")[:500] or None
        attempt.reconciliation_required = status == "UNKNOWN"
        attempt.next_action = "RECONCILE_BEFORE_RETRY" if status == "UNKNOWN" else None
        if status in {"CONFIRMED", "FAILED", "CANCELLED"}:
            attempt.completed_at = datetime.now(timezone.utc)
        await db.commit()


# ── Provider Error Classification ──────────────────────────────────────────

class ProviderError:
    """Classifies provider errors for user-friendly fallback handling."""
    
    # Error categories
    RATE_LIMIT = "rate_limit"
    AUTH_FAILED = "auth_failed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_REQUEST = "invalid_request"
    NETWORK_TIMEOUT = "network_timeout"
    UNKNOWN = "unknown"
    
    @staticmethod
    def classify(error: Exception) -> tuple[str, str]:
        """
        Classify an exception and return (category, user_message).
        
        Returns:
            (category, fallback_text) where category is one of the above,
            and fallback_text is safe to show the user.
        """
        error_str = str(error).lower()
        exc_name = error.__class__.__name__
        
        # Rate limiting
        if any(x in error_str for x in ("429", "rate limit", "too many", "quota")):
            return (ProviderError.RATE_LIMIT,
                    "The ordering service is busy right now. Please try again in a moment.")
        
        # Auth/token issues
        if any(x in error_str for x in ("401", "403", "unauthorized", "forbidden", "invalid token")):
            return (ProviderError.AUTH_FAILED,
                    "Authentication with the ordering service failed. Please try again shortly.")
        
        # Service down
        if any(x in error_str for x in ("503", "502", "service unavailable", "bad gateway")):
            return (ProviderError.SERVICE_UNAVAILABLE,
                    "The ordering service is temporarily down. We'll try again soon.")
        
        # Timeout
        if any(x in error_str for x in ("timeout", "timed out", "asyncio.timeouterror")):
            return (ProviderError.NETWORK_TIMEOUT,
                    "The request took too long. Please try again.")
        
        # Bad request (our fault)
        if any(x in error_str for x in ("400", "invalid", "malformed", "bad request")):
            return (ProviderError.INVALID_REQUEST,
                    "There was an issue with your request. Please try rephrasing.")
        
        # Unknown
        return (ProviderError.UNKNOWN,
                "Something went wrong with the ordering service. Please try again.")


def _message_content_text(content: Any) -> str:
    """Normalize plain or structured message content into readable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


_ADDRESS_OPTION_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = _JSON_BLOCK_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_last_mcp_context(history: list[dict[str, Any]]) -> str:
    for turn in reversed(history):
        if turn.get("role") == "mcp_context":
            return turn.get("content") or ""
    return ""


def _extract_address_options_from_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = _extract_last_mcp_context(history)
    options: list[dict[str, str]] = []
    for line in context.splitlines():
        match = re.match(r"^\s*(\d+)\.\s+address_id=([^\s]+)\s+label=(.+?)\s*$", line)
        if match:
            options.append({
                "index": match.group(1),
                "address_id": match.group(2).strip(),
                "label": match.group(3).strip(),
            })
    return options


def _extract_restaurants_from_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = _extract_last_mcp_context(history)
    options: list[dict[str, str]] = []
    for line in context.splitlines():
        match = re.match(
            r"^\s*restaurant:\s+res_id=([^\s]+)\s+name=(.+?)\s+rating=(.+?)\s+eta=(.+?)\s*$",
            line,
        )
        if match:
            options.append({
                "res_id": match.group(1).strip(),
                "name": match.group(2).strip(),
                "rating": match.group(3).strip(),
                "eta": match.group(4).strip(),
            })
    return options


def _normalize_followup_selection(user_message: str, history: list[dict[str, Any]]) -> str:
    text = (user_message or "").strip()
    lowered = text.lower()

    for option in _extract_address_options_from_history(history):
        label = option["label"].lower()
        if label in lowered or lowered == option["index"]:
            return (
                f'I choose delivery address "{option["label"]}" with address_id {option["address_id"]}. '
                "Use this exact address_id for all further Zomato tool calls and do not invent another one."
            )

    for restaurant in _extract_restaurants_from_history(history):
        name = restaurant["name"].lower()
        if name in lowered:
            return (
                f'I choose restaurant "{restaurant["name"]}" with res_id {restaurant["res_id"]}. '
                "Reuse this exact res_id for menu and cart steps. If menu lookup fails, search this same restaurant name "
                "again with the current valid address_id before suggesting a different restaurant."
            )

    return text


def _build_context_from_messages(out_messages: list[Any]) -> str | None:
    address_lines: list[str] = []
    restaurant_lines: list[str] = []

    for message in out_messages:
        payload = _extract_json_object(_message_content_text(getattr(message, "content", "")))
        if not payload:
            continue

        addresses = payload.get("addresses")
        if isinstance(addresses, list):
            address_lines = ["Saved addresses from latest platform lookup:"]
            for idx, address in enumerate(addresses, start=1):
                address_id = str(address.get("address_id") or "").strip()
                label = str(address.get("location_name") or "").strip()
                if address_id and label:
                    address_lines.append(f"{idx}. address_id={address_id} label={label}")

        results = payload.get("results")
        if isinstance(results, list):
            lines = ["Restaurants from latest platform search:"]
            for result in results[:12]:
                res_id = str(result.get("res_id") or "").strip()
                name = str(result.get("name") or "").strip()
                rating = str(result.get("rating") or "").strip()
                eta = str(result.get("eta") or "").strip()
                if res_id and name:
                    lines.append(f"restaurant: res_id={res_id} name={name} rating={rating or '-'} eta={eta or '-'}")
            if len(lines) > 1:
                restaurant_lines = lines

    combined = [*address_lines, *restaurant_lines]
    if not combined:
        return None
    combined.append("When the user chooses one of these saved addresses or restaurants later, reuse the exact IDs above.")
    return "\n".join(combined)


def _is_hidden_context_turn(turn: dict[str, Any]) -> bool:
    return turn.get("role") in {"mcp_context", "mcp_action"}


def _visible_history(history: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Return user/assistant turns only; hidden MCP context is injected separately."""
    visible = [turn for turn in history if not _is_hidden_context_turn(turn)]
    return visible[-limit:]


# ── Lazy imports (guarded so the bot still boots without LangGraph) ──────────

_LC_IMPORTS_DONE = False

def _ensure_imports() -> None:
    global _LC_IMPORTS_DONE, ChatGoogleGenerativeAI, HumanMessage, SystemMessage, AIMessage  # noqa: PLW0603
    global create_react_agent  # noqa: PLW0603
    if _LC_IMPORTS_DONE:
        return
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
        from langchain_core.messages import (
            HumanMessage as _HumanMessage,
            SystemMessage as _SystemMessage,
            AIMessage as _AIMessage,
        )
        from langgraph.prebuilt import create_react_agent as _create_react_agent
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "LangGraph MCP agent stack not installed. Run: pip install -r requirements.txt"
        ) from e
    ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI
    HumanMessage       = _HumanMessage
    SystemMessage      = _SystemMessage
    AIMessage          = _AIMessage
    create_react_agent = _create_react_agent
    _LC_IMPORTS_DONE = True


def _build_llm_with_fallbacks(api_key: str) -> Any:
    """
    Build a Gemini LLM with a two-tier fallback chain.

    LangChain's .with_fallbacks() automatically retries the next model when
    the previous one raises any exception (rate-limit, server error, etc.)
    """
    def _llm(model: str) -> Any:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.4,
            max_output_tokens=settings.GEMINI_MAX_TOKENS,
        )

    primary   = _llm(settings.GEMINI_PRIMARY)
    fallback1 = _llm(settings.GEMINI_FALLBACK)
    fallback2 = _llm(settings.GEMINI_FALLBACK_2)
    return primary.with_fallbacks([fallback1, fallback2])


ChatGoogleGenerativeAI = None  # type: ignore[assignment]
HumanMessage           = None  # type: ignore[assignment]
SystemMessage          = None  # type: ignore[assignment]
AIMessage              = None  # type: ignore[assignment]
create_react_agent     = None  # type: ignore[assignment]


# ── Agent build / cache ──────────────────────────────────────────────────────

_agent_cache: dict[str, Any] = {}
_tool_names_cache: dict[str, list[str]] = {}
_policy_cache: dict[str, ProviderPolicy] = {}
_build_lock: asyncio.Lock | None = None  # lazy-init: see _get_build_lock


def _get_build_lock() -> asyncio.Lock:
    """Lazily create the lock the first time async code uses it.
    Avoids 'no current event loop' errors on Python 3.9 at module-load time."""
    global _build_lock  # noqa: PLW0603
    if _build_lock is None:
        _build_lock = asyncio.Lock()
    return _build_lock


def _system_prompt(platform: str, user_name: str, policy: ProviderPolicy) -> str:
    plat_label = platform or "external delivery platform"
    if (
        settings.provider_transactions_enabled(platform)
        and policy.mode == ProviderMode.TRANSACTIONAL
        and policy.allows_transactional_mode
    ):
        provider_mode = (
            "MODE C — VERIFIED TRANSACTIONAL: mutation tools are exposed only on an explicit "
            "confirmation turn and must return a verifiable receipt."
        )
    elif policy.mode == ProviderMode.HANDOFF:
        provider_mode = (
            "MODE B — USER-CONTROLLED HANDOFF: discovery is available, but checkout occurs only "
            "on the provider's own surface. Never claim NourishBot placed the order."
        )
    else:
        provider_mode = (
            "MODE A — DISCOVERY ONLY: no checkout or provider mutation tool is available. "
            "Never claim an order was placed."
        )
    now        = datetime.now()
    hour       = now.hour
    time_label = (
        "morning"    if 6  <= hour < 11 else
        "lunch time" if 11 <= hour < 15 else
        "snack time" if 15 <= hour < 18 else
        "dinner time"if 18 <= hour < 23 else
        "late night"
    )
    return f"""You are NourishBot — a focused scheduled-meal assistant using the {plat_label} MCP server.

CURRENT PROVIDER MODE: {provider_mode}

╔══════════════════════════════════════════════════════════════════════════╗
║                          CRITICAL SAFETY RULES                           ║
║               These rules are NON-NEGOTIABLE and auditable.               ║
╚══════════════════════════════════════════════════════════════════════════╝

RULE 0: SCHEDULED MEALS ONLY
- Instant ordering, ASAP checkout, and immediate delivery are prohibited.
- Collect an explicit future meal date, local time, and timezone before any transactional call.
- Only call a provider tool that natively accepts a scheduled/pre-order/fulfillment time.
- If the connected MCP server has no scheduling-capable checkout tool, do not place the order.
  Explain that browsing is available but scheduled checkout is unsupported by this provider.
- Never simulate scheduling by placing an immediate order early.

RULE 1: ZERO HALLUCINATION
- ALL restaurant names, menu items, prices, ratings, ETAs, and order IDs must come from MCP tools.
- You CANNOT invent, assume, or guess any data.
- If a tool returns empty results or an error, report it plainly and ask the user for clarification.
- If you can't find what the user wants, say: "I couldn't find that on {plat_label}. Can you describe it differently?"

RULE 2: LOCATION AND FULFILLMENT
- First establish HOME DELIVERY or DRIVE_THROUGH/PICKUP.
- For home delivery, never search the provider before confirming a delivery address.
- When a home-delivery user gives a pincode/area: call get_saved_addresses_for_user, then validate.
- ONLY use address_ids returned by the {plat_label} API. NEVER fabricate or guess IDs.
- If a tool rejects an address_id, immediately re-fetch fresh addresses: "Let me refresh your saved addresses..."
- Confirm a home-delivery address before proceeding: "Just confirming — delivering to [ADDRESS]?"
- For drive-through discovery, collect the user's route origin and destination, then call
  find_drive_through_restaurants_on_route. Do not require a delivery address.

RULE 3: TOOL DISCOVERY BEFORE QUOTING
- Before mentioning ANY restaurant: call get_restaurants_for_keyword to find its res_id.
- Before quoting ANY menu item or price: call get_menu_items_listing with the correct res_id.
- Before confirming ANY price/item: verify it in the tool response.

RULE 4: ORDER CONFIRMATION GATES
- Before placing ANY order: call get_cart_summary or equivalent to show user exact items + price.
- User MUST give explicit confirmation like "yes", "place it", or "go ahead".
- If user says "no", "cancel", or "back" — immediately abort. Do not retry.
- Checkout/payment calls require PRIOR explicit user confirmation.
- The summary must show the scheduled meal time and timezone before confirmation.

RULE 5: ERROR RECOVERY WITHOUT SPINNING
- If a tool call fails (invalid address, menu not found, search empty), do NOT retry with guesses.
- Diagnose and ask the user: "Seems like [RESTAURANT] isn't available at your address. Try a different one?"
- If user's area is not serviceable, suggest alternatives from successful prior searches.
- After 2 failed attempts at the same thing, escalate: "I'm having trouble. Could you try a different search term?"

RULE 6: AUDIT TRAIL
- Every tool call is logged with its arguments and results.
- Your responses should be clear about what you did: "I searched for [X] and found [RESULTS]."
- If a tool fails, explain the failure to the user (in user-friendly terms).

RULE 7: RESPONSE DISCIPLINE
- Keep responses SHORT: 2–4 lines for conversational turns.
- When listing menus/orders, format clearly with prices.
- Never apologize excessively; be honest and forward-looking.
- Do NOT repeat the same failed search. Change strategy or ask for clarification.

RULE 8: MCP TOOL USAGE DISCIPLINE
- For every non-greeting ordering/search/menu/cart turn, call the relevant MCP tool before answering.
- Do not answer restaurant availability, menu items, item prices, cart contents, delivery ETA, or checkout status from memory.
- If saved IDs are provided in "SAVED MCP CONTEXT", reuse those exact address_id/res_id values instead of asking again.
- If the user chooses by number or restaurant name and the saved context contains an ID, call the next tool with that ID.

RULE 9: MAPS DISCOVERY BOUNDARY
- Use search_best_restaurants_on_maps to discover well-rated restaurants near a place.
- Use find_drive_through_restaurants_on_route for route-side drive-through discovery.
- Maps results are discovery leads, not proof of provider availability, menu, price, future opening,
  or drive-through support. After the user chooses one, search that exact restaurant on {plat_label}.
- A route travel-time estimate is not a food-ready estimate. Never promise food will be ready
  at arrival without verified preparation-time evidence from the selected provider.
- Never send a Google place_id to a {plat_label} tool as a provider restaurant ID.
- Before drive-through checkout, the provider tool must explicitly support scheduled pickup,
  drive-through, or curbside fulfillment. If it does not, stop before checkout and explain why.

CONTEXT & CONSTRAINTS
- User: {user_name or "there"}
- Time: {time_label} ({now.strftime('%I:%M %p')})
- Current date: {now.strftime('%Y-%m-%d')}
- Platform: {plat_label} (tools only — no external links or fallback URLs)
- Session state is tracked to prevent order duplicates and unauthorized actions

WORKFLOW SUMMARY
1. Greet or acknowledge the user's intent
2. Collect and validate the future meal date, time, and timezone
3. For HOME DELIVERY: validate the delivery address. For DRIVE_THROUGH: collect route origin + destination.
4. Discover restaurants with Maps when useful; for drive-through, search along the driving route.
5. Search the chosen restaurant on {plat_label} and verify provider-side fulfillment support.
6. Fetch the provider menu, build the cart, and show the exact summary and scheduled time.
7. Ask for explicit confirmation.
8. On confirmation, use only a provider tool that supports both scheduling and the chosen fulfillment.
9. Return the provider's verified order ID and scheduled time.

WHEN UNSURE
- Ask the user, don't guess.
- Call a tool to get fresh data, don't use stale context.
- Confirm important actions (address, order amount, checkout) with the user.
- If stuck, say: "I can help, but I need [SPECIFIC INFO]. Can you tell me...?"
"""


def _is_greeting(text: str) -> bool:
    t = (text or "").strip().lower().rstrip("!.?,…")
    if not t:
        return False
    words = {
        "hi", "hii", "hello", "hey", "yo", "namaste", "namaskar",
        "hola", "salaam", "good morning", "good afternoon", "good evening",
    }
    return t in words or t.split()[0] in words


def _is_reset_intent(text: str) -> bool:
    """Detect if user is requesting a reset/restart."""
    t = (text or "").strip().lower()
    reset_words = {"reset", "start over", "restart", "begin", "new order", "forget"}
    return any(word in t for word in reset_words)


def _is_confirmation_intent(text: str) -> bool:
    """Detect if user is confirming an action (yes/ok/go ahead)."""
    return is_explicit_confirmation(text)


def _is_rejection_intent(text: str) -> bool:
    """Detect if user is rejecting/canceling."""
    t = (text or "").strip().lower().rstrip("!.?,…")
    reject_words = {"no", "nope", "cancel", "stop", "back", "nevermind", "never mind"}
    return t in reject_words or any(word in t for word in reject_words)


def _extract_address_intent(text: str) -> str | None:
    """Extract if user is trying to set an address (pincode, area name, etc)."""
    t = (text or "").strip().lower()
    # Patterns: area/pincode mentions (5 digits, common area names)
    import re
    # Match 5-6 digit zip codes
    if re.search(r'\b\d{5,6}\b', t):
        return "pincode"
    # Match common area indicators
    if any(word in t for word in ["near", "around", "at", "in ", "location", "address", "area"]):
        return "area_description"
    return None


def _scheduled_safe_tool(tool: Any) -> bool:
    """Block transactional MCP tools unless their schema exposes a schedule field."""
    name = (getattr(tool, "name", "") or "").lower()
    transactional = (
        "checkout",
        "place_order",
        "create_order",
        "submit_order",
        "confirm_order",
        "purchase",
    )
    if not any(marker in name for marker in transactional):
        return True

    schema = getattr(tool, "args_schema", None)
    schema_text = ""
    try:
        if schema is not None and hasattr(schema, "model_json_schema"):
            schema_text = str(schema.model_json_schema()).lower()
        elif schema is not None and hasattr(schema, "schema"):
            schema_text = str(schema.schema()).lower()
    except Exception:
        schema_text = ""

    scheduling_fields = (
        "scheduled_for",
        "scheduled_time",
        "schedule_time",
        "delivery_time",
        "fulfillment_time",
        "preorder_time",
    )
    return any(field in schema_text for field in scheduling_fields)


async def _build_agent(platform: str, allow_transaction: bool = False):
    """Build (or fetch cached) LangGraph React agent for the given platform."""
    async with _get_build_lock():
        _ensure_imports()

        # Lazy import — keeps mcp_clients out of import path when MCP off.
        from app.agent.mcp_clients import get_mcp_tools_async

        provider_tools = await get_mcp_tools_async(platform)
        if not provider_tools:
            logger.warning("MCP agent: no tools available for platform=%s; agent will not be built.", platform)
            return None

        async with AsyncSessionLocal() as db:
            policy = await sync_tool_inventory(db, platform, provider_tools)
        _policy_cache[platform] = policy
        transaction_turn = allow_transaction and policy.allows_transactional_mode
        cache_key = f"{platform}:v{policy.version}:{'transactional' if transaction_turn else policy.mode.value.lower()}"
        if cache_key in _agent_cache:
            return _agent_cache[cache_key]

        from app.agent.maps_client import build_maps_tools
        maps_tools = build_maps_tools()

        # Optional: filter tools by platform prefix so a Zomato-mode chat
        # only sees Zomato tools. This keeps the model focused.
        prefix = "zomato" if platform == "Zomato" else "swiggy"
        filtered = [t for t in provider_tools if (getattr(t, "name", "") or "").lower().startswith(prefix)]
        # If our naming convention doesn't match (langchain-mcp-adapters
        # may name tools <server>__<tool> or similar), fall back to all tools.
        candidate_provider_tools = filtered or provider_tools
        bound_provider_tools = [
            tool
            for tool in candidate_provider_tools
            if _scheduled_safe_tool(tool)
            and tool_allowed(policy, tool, transaction_turn=transaction_turn)
        ]
        bound_tools = [*bound_provider_tools, *maps_tools]
        blocked = len(candidate_provider_tools) - len(bound_provider_tools)
        if blocked:
            logger.warning(
                "MCP[%s]: blocked %d tool(s) by scheduling/capability policy",
                platform,
                blocked,
            )
        _tool_names_cache[cache_key] = [getattr(t, "name", str(t)) for t in bound_tools]
        _tool_names_cache[platform] = _tool_names_cache[cache_key]

        api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        llm = _build_llm_with_fallbacks(api_key)

        agent = create_react_agent(llm, bound_tools)
        _agent_cache[cache_key] = agent
        logger.info(
            "MCP agent built for platform=%s with %d/%d tools.",
            platform, len(bound_tools), len(provider_tools) + len(maps_tools),
        )
        return agent


# ── Public API used by the Telegram bot ──────────────────────────────────────

async def get_mcp_response(
    user_message: str,
    chat_id: str | int,
    user_name: str,
    platform: str,
    channel_type: str = "TELEGRAM",
    planning_context: str | None = None,
) -> str | None:
    """
    Run a single agent turn for a Telegram or WhatsApp chat. Returns the assistant text
    on success, or None when MCP is unavailable so the caller can fall back
    to the legacy redirect-link / NourishBot-bot flow.
    """
    async with AsyncSessionLocal() as db:
        policy = await current_policy(db, platform)
    confirmation_turn = _is_confirmation_intent(user_message)
    if policy.state == CapabilityState.DISABLED:
        return f"{platform} is currently disabled. Please choose another provider."
    if policy.mode == ProviderMode.HANDOFF and confirmation_turn:
        url = provider_handoff_url(platform)
        if not url:
            return f"{platform} handoff is not configured right now. No order was placed."
        return (
            f"{platform} is in Mode B (user-controlled handoff). Complete and verify the scheduled "
            f"checkout on the provider's own page: {url}"
        )
    if not settings.MCP_ENABLED:
        if policy.mode == ProviderMode.HANDOFF:
            url = provider_handoff_url(platform)
            return f"{platform} is available through Mode B handoff: {url}" if url else None
        return None
    try:
        agent = await _build_agent(
            platform,
            allow_transaction=(
                settings.provider_transactions_enabled(platform)
                and confirmation_turn
            ),
        )
    except Exception as e:
        logger.error("MCP agent: build failed: %s", redact_text(str(e)))
        return None

    if agent is None:
        return None

    policy = _policy_cache.get(platform, policy)
    if confirmation_turn and not (
        settings.provider_transactions_enabled(platform) and policy.allows_transactional_mode
    ):
        if policy.mode == ProviderMode.HANDOFF:
            url = provider_handoff_url(platform)
            return (
                f"{platform} is in Mode B (user-controlled handoff). Complete checkout here: {url}"
                if url
                else f"{platform} handoff is not configured. No order was placed."
            )
        return (
            f"{platform} is in Mode A (discovery only). Scheduled transactional ordering is not "
            "verified, so I did not submit an order."
        )

    # Cheap deterministic guard for greetings — skip a tool call.
    if _is_greeting(user_message):
        plat_label = platform
        reply = (
            f"Hey {user_name or 'there'}! 👋 *{plat_label}* is in {policy.mode.value.title()} mode. "
            "Tell me what cuisine you're craving or share your area / pin code."
        )
        session = await get_session(chat_id, channel_type)
        new_history = list(session.get("history") or [])
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": reply})
        await save_session(chat_id, {
            "history": _trim_mcp_history(new_history),
            "ordering_platform": platform,
        }, channel_type=channel_type)
        return reply

    _ensure_imports()

    # Replay rolling conversation history (last 8 turns) so the model has
    # continuity, just like the NourishBot orchestrator does.
    session = await get_session(chat_id, channel_type)
    full_history = list(session.get("history") or [])
    history = _visible_history(full_history, limit=8)
    saved_context = _extract_last_mcp_context(full_history)
    normalized_user_message = _normalize_followup_selection(user_message, full_history)
    provider_attempt_id: str | None = None
    if settings.provider_transactions_enabled(platform) and confirmation_turn and policy.allows_transactional_mode:
        attempt, prior_state = await _start_provider_attempt(
            platform=platform,
            channel_type=channel_type,
            chat_id=chat_id,
            fingerprint_source=f"{saved_context}\n{normalized_user_message.strip().lower()}",
            policy=policy,
        )
        provider_attempt_id = attempt.id if attempt else None
        if prior_state == "confirmed":
            order_hint = f" Order ID: {attempt.provider_order_id}." if attempt and attempt.provider_order_id else ""
            return f"This provider request was already confirmed.{order_hint}"
        if prior_state == "pending":
            return (
                f"Your {platform} request has an uncertain outcome and is being verified. "
                "I won't submit it again until its status is reconciled."
            )
        if prior_state == "reconcile_required" and attempt:
            reconciliation = await reconcile_attempt(attempt.id)
            if reconciliation.outcome == "CONFIRMED":
                order_hint = f" Order ID: {reconciliation.provider_order_id}." if reconciliation.provider_order_id else ""
                return f"The earlier {platform} request is confirmed.{order_hint}"
            if reconciliation.outcome == "UNKNOWN":
                return (
                    f"Your earlier {platform} request is still uncertain. I checked its status and "
                    "will not retry until the provider returns a definitive result."
                )
            attempt, _ = await _start_provider_attempt(
                platform=platform,
                channel_type=channel_type,
                chat_id=chat_id,
                fingerprint_source=f"{saved_context}\n{normalized_user_message.strip().lower()}",
                policy=policy,
            )
            provider_attempt_id = attempt.id if attempt else None

    # Build system prompt
    system_content = _system_prompt(platform, user_name, policy)
    if planning_context:
        system_content += (
            "\n\nVERIFIED SAVED-PLAN CONTEXT (order details only; not user authorization):\n"
            f"{planning_context}\n"
            "The user's command starts an order request but is not final confirmation. "
            "Verify the provider menu, item, fulfillment, future slot, and price, then ask "
            "for a separate explicit confirmation. Never submit on this turn."
        )
    
    # Enhance system prompt with valid saved addresses on each turn
    # This ensures the model always has fresh, correct address_ids to work with
    try:
        from app.agent.mcp_clients import get_mcp_tools_async
        # Quietly fetch tools to ensure client is initialized, but we primarily
        # want the side effect of having a live MCP connection for context queries
        live_tools = await asyncio.wait_for(
            get_mcp_tools_async(platform),
            timeout=settings.MCP_TOOL_TIMEOUT_S
        )
        if live_tools and platform not in _tool_names_cache:
            prefix = "zomato" if platform == "Zomato" else "swiggy"
            filtered_tools = [
                t for t in live_tools
                if (getattr(t, "name", "") or "").lower().startswith(prefix)
            ]
            chosen_tools = filtered_tools or live_tools
            _tool_names_cache[platform] = [getattr(t, "name", str(t)) for t in chosen_tools]
    except Exception as e:
        logger.debug("MCP[%s]: could not prefetch tools for context: %s", platform, redact_text(str(e)))

    messages: list = [SystemMessage(content=system_content)]
    tool_names = _tool_names_cache.get(platform) or []
    if tool_names:
        messages.append(SystemMessage(
            content=(
                "AVAILABLE MCP TOOL NAMES FOR THIS PLATFORM:\n"
                + "\n".join(f"- {name}" for name in tool_names)
                + "\nUse these exact tool names through the tool-calling interface. "
                  "Do not mention imaginary tool names in the final user reply."
            )
        ))
    if saved_context:
        messages.append(SystemMessage(content=f"SAVED MCP CONTEXT FROM PREVIOUS TOOL RESULTS:\n{saved_context}"))
    for turn in history:
        role    = turn.get("role")
        content = turn.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=normalized_user_message))

    logger.debug(
        "MCP[%s] turn start — chat=%s history_turns=%d has_saved_context=%s",
        platform, chat_id, len(history), bool(saved_context),
    )

    result = None
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages}),
            timeout=settings.MCP_TOOL_TIMEOUT_S * 2,
        )
    except asyncio.TimeoutError:
        await _finish_provider_attempt(
            provider_attempt_id,
            status="UNKNOWN",
            error="Provider timed out before a verifiable receipt was returned",
        )
        logger.warning("MCP agent: timeout for chat_id=%s platform=%s", chat_id, platform)
        return (
            "Hmm, that took too long on the "
            f"{platform} side 🐢 — could you try again in a moment?"
        )
    except Exception as e:
        err_str = str(e).lower()
        # ── AbortError handling ────────────────────────────────────────────
        # When mcp-remote (Node.js subprocess) finishes a tool call and the
        # SSE connection closes, undici HTTP client raises AbortError during
        # cleanup. This is expected noise — the tool call actually succeeded,
        # and the response was already delivered to the agent.
        # The error occurs asynchronously during subprocess shutdown; catching
        # it here prevents it from crashing the message handler.
        #
        # Note: The subprocess still prints "Error from remote server: DOMException
        # [AbortError]" to stderr — this is from Node.js and can't be suppressed
        # via Python. It's harmless and doesn't affect functionality.
        #
        # Real errors (connection failure, timeout, etc.) are caught separately.
        if "abort" in err_str or "aborted" in err_str:
            logger.debug(
                "MCP[%s] subprocess AbortError (expected cleanup) chat=%s — tool call likely succeeded.",
                platform, chat_id,
            )
            await _finish_provider_attempt(
                provider_attempt_id,
                status="UNKNOWN",
                error="Provider connection ended before a verifiable receipt was returned",
            )
            return (
                f"I couldn't verify whether {platform} accepted the request. "
                "I have not retried it, to avoid a duplicate order. Its status needs reconciliation."
            )
        else:
            await _finish_provider_attempt(provider_attempt_id, status="FAILED", error=str(e))
            logger.error("MCP[%s] invoke failed chat=%s: %s", platform, chat_id, redact_text(str(e)))
            from app.agent.mcp_clients import reset_platform_cache
            await reset_platform_cache(platform)
            for cache in (_agent_cache, _tool_names_cache):
                for key in list(cache):
                    if key == platform or key.startswith(f"{platform}:"):
                        cache.pop(key, None)
            return None

    if result is None:
        return "Got it — anything else you'd like me to do on " + platform + "?"

    # ── Debug: log every message in the result ────────────────────────────────
    out_messages = result.get("messages") if isinstance(result, dict) else None
    detected_address_error = False
    tool_call_count = 0
    transaction_called = False
    provider_order_id: str | None = None
    tool_receipt_records: list[dict[str, Any]] = []
    pending_tool_records: dict[str, dict[str, Any]] = {}
    pending_tool_order: list[dict[str, Any]] = []
    
    if out_messages:
        for i, m in enumerate(out_messages):
            mtype   = getattr(m, "type", m.__class__.__name__)
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    logger.debug(
                        "MCP[%s] msg[%d] TOOL_CALL — tool=%s arg_keys=%s",
                        platform, i, tc_name, sorted(tc_args.keys()) if isinstance(tc_args, dict) else [],
                    )
                    tool_call_count += 1
                    transaction_called = transaction_called or _is_transaction_tool(str(tc_name))
                    tc_id = (
                        tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    )
                    operation = (
                        "TRANSACTION"
                        if _is_transaction_tool(str(tc_name))
                        else "CANCEL_ORDER"
                        if is_cancel_tool(str(tc_name))
                        else "GET_ORDER_STATUS"
                        if is_status_tool(str(tc_name))
                        else "DISCOVERY"
                    )
                    receipt_record = {
                        "operation": operation,
                        "tool_name": str(tc_name),
                        "request": tc_args if isinstance(tc_args, dict) else {"value": str(tc_args)},
                        "response": {},
                        "outcome": "NO_RESPONSE",
                    }
                    tool_receipt_records.append(receipt_record)
                    pending_tool_order.append(receipt_record)
                    if tc_id:
                        pending_tool_records[str(tc_id)] = receipt_record
                    # Extra focus on address/location args
                    if isinstance(tc_args, dict):
                        for k, v in tc_args.items():
                            if any(x in k.lower() for x in ("address", "location", "area", "lat", "lng", "pincode", "city")):
                                logger.info(
                                    "MCP[%s] sensitive location argument redacted — tool=%s field=%s",
                                    platform, tc_name, k,
                                )
            elif mtype in ("tool", "ToolMessage"):
                content_str = str(content)[:500]
                payload = _extract_json_object(str(content))
                if payload:
                    provider_order_id = str(
                        payload.get("order_id")
                        or payload.get("orderId")
                        or payload.get("provider_order_id")
                        or provider_order_id
                        or ""
                    ).strip() or None
                tool_call_id = getattr(m, "tool_call_id", None)
                receipt_record = pending_tool_records.get(str(tool_call_id)) if tool_call_id else None
                if receipt_record is None:
                    receipt_record = next(
                        (record for record in pending_tool_order if record["outcome"] == "NO_RESPONSE"),
                        None,
                    )
                if receipt_record is not None:
                    receipt_record["response"] = payload or {"text": redact_text(str(content))[:2000]}
                    receipt_record["outcome"] = "RETURNED"
                    receipt_record["provider_order_id"] = provider_order_id
                logger.debug(
                    "MCP[%s] msg[%d] TOOL_RESULT — chars=%d",
                    platform, i, len(str(content)),
                )
                # Detect common errors that indicate bad address IDs or restaurant IDs
                if "invalid_address" in content_str.lower() or "invalid" in content_str.lower() or \
                   "menu_not_found" in content_str.lower() or "no menu" in content_str.lower() or \
                   '"results":[]' in content_str:
                    detected_address_error = True
                    logger.warning(
                        "MCP[%s] msg[%d] detected address/menu error: %s",
                        platform, i, redact_text(content_str[:100]),
                    )
            elif mtype in ("ai", "AIMessage", "assistant") and content:
                logger.debug(
                    "MCP[%s] msg[%d] AI — chars=%d",
                    platform, i, len(str(content)),
                )
        
        if detected_address_error:
            logger.warning(
                "MCP[%s] detected address/menu error in conversation — agent should recover by fetching fresh data",
                platform,
            )

    # ── Extract final assistant text ──────────────────────────────────────────
    final_text = ""
    if out_messages:
        for m in reversed(out_messages):
            content = getattr(m, "content", None)
            mtype   = getattr(m, "type", "") or m.__class__.__name__
            if mtype.lower() in ("ai", "aimessage", "assistant"):
                normalized = _message_content_text(content)
                if normalized:
                    final_text = normalized
                    break
    if not final_text and out_messages:
        last = out_messages[-1]
        final_text = _message_content_text(getattr(last, "content", ""))

    if not final_text:
        final_text = "Got it — anything else you'd like me to do on " + platform + "?"

    if provider_attempt_id:
        if transaction_called and provider_order_id:
            await _finish_provider_attempt(
                provider_attempt_id,
                status="CONFIRMED",
                provider_order_id=provider_order_id,
            )
        elif transaction_called:
            await _finish_provider_attempt(
                provider_attempt_id,
                status="UNKNOWN",
                error="Transactional tool returned no verifiable provider order id",
            )
            final_text = (
                f"I couldn't verify whether {platform} accepted the request. "
                "I won't retry it until the provider status is reconciled, so a duplicate is not created."
            )
        else:
            await _finish_provider_attempt(
                provider_attempt_id,
                status="FAILED",
                error="Confirmation turn did not invoke a transactional provider tool",
            )

    try:
        await persist_tool_receipts(
            provider=platform,
            mode=policy.mode.value,
            channel_type=channel_type,
            external_chat_id=chat_id,
            records=tool_receipt_records,
            attempt_id=provider_attempt_id,
            capability_version_id=policy.version_id,
            tool_schema_hash=policy.tool_schema_hash,
        )
    except Exception as exc:
        # A confirmed transaction already has its minimal receipt on the
        # ProviderAttempt. Do not turn a successful provider call into a retry.
        logger.error(
            "MCP[%s] receipt persistence failed chat=%s: %s",
            platform,
            chat_id,
            redact_text(str(exc)),
        )

    context_update = _build_context_from_messages(out_messages or [])

    logger.debug(
        "MCP[%s] reply chat=%s tool_calls=%d context_update=%s chars=%d",
        platform, chat_id, tool_call_count, bool(context_update), len(final_text),
    )

    # Persist this turn into rolling history (cap last 16)
    new_history = list(full_history)
    new_history.append({"role": "user",      "content": user_message})
    new_history.append({"role": "assistant", "content": final_text})
    if context_update:
        new_history.append({"role": "mcp_context", "content": context_update})
    new_history = _trim_mcp_history(new_history)
    await save_session(chat_id, {
        "history":           new_history,
        "ordering_platform": platform,
    }, channel_type=channel_type)

    return final_text


def _trim_mcp_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep the chat history small without losing the latest hidden MCP context.
    Visible user/assistant turns are capped, while the newest mcp_context turn
    is preserved so follow-up selections can reuse exact address_id/res_id data.
    """
    latest_context: dict[str, Any] | None = None
    for turn in reversed(history):
        if turn.get("role") == "mcp_context":
            latest_context = turn
            break

    visible = [turn for turn in history if not _is_hidden_context_turn(turn)][-16:]
    if latest_context:
        visible.append(latest_context)
    return visible
