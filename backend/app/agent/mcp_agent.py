"""
LangGraph MCP Agent — Zomato + Swiggy live ordering inside Telegram.
====================================================================

Mirrors the architecture of the previous Zomatobot LangGraph backend:

    Telegram message
        -> deterministic guards (greeting, reset, simple intents)
        -> LangGraph ReAct agent
            - OpenAI GPT-4o chat model (bound to MCP tools)
            - Tools come from langchain-mcp-adapters (Zomato + Swiggy)
            - System prompt enforces no-hallucination + concise replies
        -> Final assistant text returned to Telegram

The agent itself is per-message stateless; the rolling conversation history
is replayed into the prompt the same way as the Dzukku orchestrator already
does, so the same PostgreSQL session table works without schema changes.

If MCP is not enabled or the tools fail to load, get_mcp_response() returns
None and Telegram falls back to the existing Dzukku orchestrator + redirect
links.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.db.crud import get_session, save_session

logger = logging.getLogger(__name__)


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
    global _LC_IMPORTS_DONE, ChatOpenAI, HumanMessage, SystemMessage, AIMessage  # noqa: PLW0603
    global create_react_agent  # noqa: PLW0603
    if _LC_IMPORTS_DONE:
        return
    try:
        from langchain_openai import ChatOpenAI as _ChatOpenAI
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
    ChatOpenAI         = _ChatOpenAI
    HumanMessage       = _HumanMessage
    SystemMessage      = _SystemMessage
    AIMessage          = _AIMessage
    create_react_agent = _create_react_agent
    _LC_IMPORTS_DONE = True


def _build_llm_with_fallbacks(api_key: str) -> Any:
    """
    Build an OpenAI LLM with a 2-tier fallback chain:
      gpt-4o       (primary — fast, capable)
        └─ gpt-4o-mini   (fallback 1 — cheaper, stable)
             └─ gpt-3.5-turbo  (fallback 2 — legacy reliable)

    LangChain's .with_fallbacks() automatically retries the next model when
    the previous one raises any exception (rate-limit, server error, etc.)
    """
    def _llm(model: str) -> Any:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.4,
            max_tokens=settings.OPENAI_MAX_TOKENS,
        )

    primary   = _llm(settings.OPENAI_PRIMARY)
    fallback1 = _llm(settings.OPENAI_FALLBACK)
    fallback2 = _llm(settings.OPENAI_FALLBACK_2)
    return primary.with_fallbacks([fallback1, fallback2])


ChatOpenAI = None  # type: ignore[assignment]
HumanMessage           = None  # type: ignore[assignment]
SystemMessage          = None  # type: ignore[assignment]
AIMessage              = None  # type: ignore[assignment]
create_react_agent     = None  # type: ignore[assignment]


# ── Agent build / cache ──────────────────────────────────────────────────────

_agent_cache: dict[str, Any] = {}
_tool_names_cache: dict[str, list[str]] = {}
_build_lock: asyncio.Lock | None = None  # lazy-init: see _get_build_lock


def _get_build_lock() -> asyncio.Lock:
    """Lazily create the lock the first time async code uses it.
    Avoids 'no current event loop' errors on Python 3.9 at module-load time."""
    global _build_lock  # noqa: PLW0603
    if _build_lock is None:
        _build_lock = asyncio.Lock()
    return _build_lock


def _system_prompt(platform: str, user_name: str) -> str:
    plat_label = {"Zomato": "Zomato", "Swiggy": "Swiggy"}.get(platform, "external delivery platform")
    now        = datetime.now()
    hour       = now.hour
    time_label = (
        "morning"    if 6  <= hour < 11 else
        "lunch time" if 11 <= hour < 15 else
        "snack time" if 15 <= hour < 18 else
        "dinner time"if 18 <= hour < 23 else
        "late night"
    )
    return f"""You are Dzukku — a warm, witty restaurant assistant helping a Telegram user place a real order on {plat_label} via the {plat_label} MCP server.

╔══════════════════════════════════════════════════════════════════════════╗
║                          CRITICAL SAFETY RULES                           ║
║               These rules are NON-NEGOTIABLE and auditable.               ║
╚══════════════════════════════════════════════════════════════════════════╝

RULE 1: ZERO HALLUCINATION
- ALL restaurant names, menu items, prices, ratings, ETAs, and order IDs must come from MCP tools.
- You CANNOT invent, assume, or guess any data.
- If a tool returns empty results or an error, report it plainly and ask the user for clarification.
- If you can't find what the user wants, say: "I couldn't find that on {plat_label}. Can you describe it differently?"

RULE 2: ADDRESS VALIDATION (BEFORE ANYTHING ELSE)
- NEVER proceed to search restaurants without confirming a delivery address.
- When user gives a pincode/area: immediately call get_saved_addresses_for_user THEN validate.
- ONLY use address_ids returned by the {plat_label} API. NEVER fabricate or guess IDs.
- If a tool rejects an address_id, immediately re-fetch fresh addresses: "Let me refresh your saved addresses..."
- Confirm address with user before proceeding: "Just confirming — delivering to [ADDRESS]?"

RULE 3: TOOL DISCOVERY BEFORE QUOTING
- Before mentioning ANY restaurant: call get_restaurants_for_keyword to find its res_id.
- Before quoting ANY menu item or price: call get_menu_items_listing with the correct res_id.
- Before confirming ANY price/item: verify it in the tool response.

RULE 4: ORDER CONFIRMATION GATES
- Before placing ANY order: call get_cart_summary or equivalent to show user exact items + price.
- User MUST give explicit confirmation like "yes", "place it", or "go ahead".
- If user says "no", "cancel", or "back" — immediately abort. Do not retry.
- Checkout/payment calls require PRIOR explicit user confirmation.

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

CONTEXT & CONSTRAINTS
- User: {user_name or "there"}
- Time: {time_label} ({now.strftime('%I:%M %p')})
- Platform: {plat_label} (tools only — no external links or fallback URLs)
- Session state is tracked to prevent order duplicates and unauthorized actions

WORKFLOW SUMMARY
1. Greet or acknowledge the user's intent
2. Validate/set delivery address (call get_saved_addresses_for_user if needed)
3. Search for restaurants matching user's taste/budget (use get_restaurants_for_keyword)
4. For chosen restaurant: fetch menu (get_menu_items_listing)
5. Build cart with selected items
6. Show order summary and request confirmation
7. On explicit confirmation: place order via checkout tool
8. Return order ID and delivery ETA

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
    t = (text or "").strip().lower().rstrip("!.?,…")
    confirm_words = {"yes", "yep", "ok", "okay", "sure", "go", "go ahead", "place it", "confirm", "agreed"}
    return t in confirm_words or any(word in t for word in confirm_words)


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


async def _build_agent(platform: str):
    """Build (or fetch cached) LangGraph React agent for the given platform."""
    cache_key = platform
    if cache_key in _agent_cache:
        return _agent_cache[cache_key]

    async with _get_build_lock():
        if cache_key in _agent_cache:
            return _agent_cache[cache_key]

        _ensure_imports()

        # Lazy import — keeps mcp_clients out of import path when MCP off.
        from app.agent.mcp_clients import get_mcp_tools_async

        tools = await get_mcp_tools_async(platform)
        if not tools:
            logger.warning("MCP agent: no tools available for platform=%s; agent will not be built.", platform)
            _agent_cache[cache_key] = None
            return None

        # Optional: filter tools by platform prefix so a Zomato-mode chat
        # only sees Zomato tools. This keeps the model focused.
        prefix = "zomato" if platform == "Zomato" else "swiggy"
        filtered = [t for t in tools if (getattr(t, "name", "") or "").lower().startswith(prefix)]
        # If our naming convention doesn't match (langchain-mcp-adapters
        # may name tools <server>__<tool> or similar), fall back to all tools.
        bound_tools = filtered or tools
        _tool_names_cache[cache_key] = [getattr(t, "name", str(t)) for t in bound_tools]

        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        llm = _build_llm_with_fallbacks(api_key)

        agent = create_react_agent(llm, bound_tools)
        _agent_cache[cache_key] = agent
        logger.info(
            "MCP agent built for platform=%s with %d/%d tools.",
            platform, len(bound_tools), len(tools),
        )
        return agent


# ── Public API used by the Telegram bot ──────────────────────────────────────

async def get_mcp_response(
    user_message: str,
    chat_id: int,
    user_name: str,
    platform: str,
) -> str | None:
    """
    Run a single agent turn for a Telegram chat. Returns the assistant text
    on success, or None when MCP is unavailable so the caller can fall back
    to the legacy redirect-link / Dzukku-bot flow.
    """
    if not settings.MCP_ENABLED:
        return None
    if platform not in ("Zomato", "Swiggy"):
        return None

    try:
        agent = await _build_agent(platform)
    except Exception as e:
        logger.error("MCP agent: build failed: %s", e, exc_info=True)
        return None

    if agent is None:
        return None

    # Cheap deterministic guard for greetings — skip a tool call.
    if _is_greeting(user_message):
        plat_label = "Zomato" if platform == "Zomato" else "Swiggy"
        reply = (
            f"Hey {user_name or 'there'}! 👋 I'm connected to *{plat_label}* now. "
            "Tell me what cuisine you're craving or share your area / pin code."
        )
        session = await get_session(chat_id)
        new_history = list(session.get("history") or [])
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": reply})
        await save_session(chat_id, {
            "history": _trim_mcp_history(new_history),
            "ordering_platform": platform,
        })
        return reply

    _ensure_imports()

    # Replay rolling conversation history (last 8 turns) so the model has
    # continuity, just like the Dzukku orchestrator does.
    session = await get_session(chat_id)
    full_history = list(session.get("history") or [])
    history = _visible_history(full_history, limit=8)
    saved_context = _extract_last_mcp_context(full_history)
    normalized_user_message = _normalize_followup_selection(user_message, full_history)

    # Build system prompt
    system_content = _system_prompt(platform, user_name)
    
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
        logger.debug("MCP[%s]: could not prefetch tools for context: %s", platform, e)

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
        "MCP[%s] turn start — chat=%s user=%r normalized=%r history_turns=%d has_saved_context=%s",
        platform, chat_id, user_message[:120], normalized_user_message[:160], len(history), bool(saved_context),
    )

    result = None
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": messages}),
            timeout=settings.MCP_TOOL_TIMEOUT_S * 2,
        )
    except asyncio.TimeoutError:
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
            # Return a graceful message since we can't get the agent response
            return f"Got it — I've sent your request to {platform}. You'll see the confirmation shortly."
        else:
            logger.error("MCP[%s] invoke failed chat=%s: %s", platform, chat_id, e, exc_info=True)
            from app.agent.mcp_clients import reset_platform_cache
            await reset_platform_cache(platform)
            _agent_cache.pop(platform, None)
            _tool_names_cache.pop(platform, None)
            return None

    if result is None:
        return "Got it — anything else you'd like me to do on " + platform + "?"

    # ── Debug: log every message in the result ────────────────────────────────
    out_messages = result.get("messages") if isinstance(result, dict) else None
    detected_address_error = False
    tool_call_count = 0
    
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
                        "MCP[%s] msg[%d] TOOL_CALL — tool=%s args=%s",
                        platform, i, tc_name, str(tc_args)[:200],
                    )
                    tool_call_count += 1
                    # Extra focus on address/location args
                    if isinstance(tc_args, dict):
                        for k, v in tc_args.items():
                            if any(x in k.lower() for x in ("address", "location", "area", "lat", "lng", "pincode", "city")):
                                logger.info(
                                    "MCP[%s] ADDRESS ARG — tool=%s field=%s value=%r",
                                    platform, tc_name, k, v,
                                )
            elif mtype in ("tool", "ToolMessage"):
                content_str = str(content)[:500]
                logger.debug(
                    "MCP[%s] msg[%d] TOOL_RESULT — content=%s",
                    platform, i, content_str,
                )
                # Detect common errors that indicate bad address IDs or restaurant IDs
                if "invalid_address" in content_str.lower() or "invalid" in content_str.lower() or \
                   "menu_not_found" in content_str.lower() or "no menu" in content_str.lower() or \
                   '"results":[]' in content_str:
                    detected_address_error = True
                    logger.warning(
                        "MCP[%s] msg[%d] detected address/menu error: %s",
                        platform, i, content_str[:100],
                    )
            elif mtype in ("ai", "AIMessage", "assistant") and content:
                logger.debug(
                    "MCP[%s] msg[%d] AI — content=%s",
                    platform, i, str(content)[:200],
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

    context_update = _build_context_from_messages(out_messages or [])

    logger.debug(
        "MCP[%s] reply chat=%s tool_calls=%d context_update=%s: %r",
        platform, chat_id, tool_call_count, bool(context_update), final_text[:120],
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
    })

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
