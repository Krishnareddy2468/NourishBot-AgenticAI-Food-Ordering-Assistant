"""
LangGraph MCP Agent — Zomato + Swiggy live ordering inside Telegram.
====================================================================

Mirrors the architecture of the previous Zomatobot LangGraph backend:

    Telegram message
        -> deterministic guards (greeting, reset, simple intents)
        -> LangGraph ReAct agent
            - Gemini 2.5 Flash chat model (bound to MCP tools)
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
import logging
import os
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.db.crud import get_session, save_session

logger = logging.getLogger(__name__)


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
    HumanMessage           = _HumanMessage
    SystemMessage          = _SystemMessage
    AIMessage              = _AIMessage
    create_react_agent     = _create_react_agent
    _patch_gemini_schema_converter()
    _LC_IMPORTS_DONE = True


def _build_llm_with_fallbacks(api_key: str) -> Any:
    """
    Build a Gemini LLM with a 2-tier fallback chain:
      gemini-2.5-flash  (primary — fast, capable)
        └─ gemini-2.0-flash  (fallback 1 — stable, less traffic)
             └─ gemini-1.5-flash  (fallback 2 — old reliable)

    LangChain's .with_fallbacks() automatically retries the next model when
    the previous one raises any exception (503 high-demand, 429 rate-limit, etc.)
    The Google SDK already does its own exponential retry on transient errors;
    this chain kicks in only when all SDK retries are exhausted.
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


def _patch_gemini_schema_converter() -> None:
    """
    Monkey-patch langchain_google_genai._dict_to_genai_schema so that:
      1. enum values that are integers become strings (Gemini rejects int enums)
      2. additionalProperties keys are stripped (Gemini doesn't support them)

    This is a one-time patch at import time. It covers ALL recursive calls
    because the original function looks up _dict_to_genai_schema in its own
    module globals — which now point to the patched version.
    """
    try:
        import langchain_google_genai._function_utils as _fu
        _orig = _fu._dict_to_genai_schema

        def _sanitize(d: Any) -> Any:
            if not isinstance(d, dict):
                return d
            out: dict = {}
            for k, v in d.items():
                if k == "additionalProperties":
                    continue  # not supported by Gemini
                if k == "enum" and isinstance(v, list):
                    out[k] = [str(i) for i in v]  # integers → strings
                elif isinstance(v, dict):
                    out[k] = _sanitize(v)
                elif isinstance(v, list):
                    out[k] = [_sanitize(i) if isinstance(i, dict) else i for i in v]
                else:
                    out[k] = v
            # Gemini only allows enum on STRING type — force type if enum present
            if "enum" in out and out.get("type") != "string":
                out["type"] = "string"
            return out

        def _patched(schema_dict: Any, *args: Any, **kwargs: Any) -> Any:
            return _orig(_sanitize(schema_dict), *args, **kwargs)

        _fu._dict_to_genai_schema = _patched
        logger.debug("Patched Gemini schema converter: int enums → strings.")
    except Exception as e:
        logger.warning("Could not patch Gemini schema converter: %s", e)


ChatGoogleGenerativeAI = None  # type: ignore[assignment]
HumanMessage           = None  # type: ignore[assignment]
SystemMessage          = None  # type: ignore[assignment]
AIMessage              = None  # type: ignore[assignment]
create_react_agent     = None  # type: ignore[assignment]


# ── Agent build / cache ──────────────────────────────────────────────────────

_agent_cache: dict[str, Any] = {}
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

CRITICAL RULES — read carefully:
- All restaurant data, menu items, prices, availability, and order placement happen through MCP tools. NEVER fabricate restaurants, dishes, prices, or order IDs.
- Always call a search/discovery tool BEFORE quoting any restaurant or dish.
- Always call a menu / item-detail tool BEFORE quoting prices or item descriptions.
- Always call the cart / order-summary tool to show the user what they're paying for BEFORE checkout.
- Confirm the order summary with the user (single explicit "yes / place it" / "go ahead") BEFORE calling any place-order or checkout tool.
- If a tool returns an empty list or an error, say so plainly and suggest a next step. Do not invent data.
- Keep replies short — 2 to 4 lines for chat, longer only when listing menus or order summaries.

CONTEXT
- Telegram user first name: {user_name or "there"}
- Time of day: {time_label} ({now.strftime('%I:%M %p')})
- Active platform: {plat_label}

DELIVERY ADDRESS / LOCATION
- If the platform's tools require a delivery location and one is not yet set on the session, ask the user for their area / pin code. Do not proceed to ordering until a location is established.

UPSELLING / CHIT-CHAT
- One gentle upsell per conversation (e.g. "want a beverage with that?"). Never spam.
- If the user goes off-topic, redirect politely back to ordering.

WHEN UNSURE
- Pick the smallest tool call that moves the conversation forward and let the user steer.
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

        api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
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
        return (
            f"Hey {user_name or 'there'}! 👋 I'm connected to *{plat_label}* now. "
            "Tell me what cuisine you're craving or share your area / pin code."
        )

    _ensure_imports()

    # Replay rolling conversation history (last 8 turns) so the model has
    # continuity, just like the Dzukku orchestrator does.
    session = await get_session(chat_id)
    history = (session.get("history") or [])[-8:]

    messages: list = [SystemMessage(content=_system_prompt(platform, user_name))]
    for turn in history:
        role    = turn.get("role")
        content = turn.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

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
        logger.error("MCP agent: invoke failed: %s", e, exc_info=True)
        return None  # let Telegram fall back to the redirect link

    # Extract final assistant text
    final_text = ""
    out_messages = result.get("messages") if isinstance(result, dict) else None
    if out_messages:
        for m in reversed(out_messages):
            content = getattr(m, "content", None)
            if isinstance(content, str) and content.strip() and getattr(m, "type", "") in ("ai", "AIMessage", "assistant"):
                final_text = content.strip()
                break
    if not final_text and out_messages:
        last = out_messages[-1]
        final_text = (getattr(last, "content", "") or "").strip()

    if not final_text:
        final_text = "Got it — anything else you'd like me to do on " + platform + "?"

    # Persist this turn into rolling history (cap last 16)
    new_history = list(session.get("history") or [])
    new_history.append({"role": "user",      "content": user_message})
    new_history.append({"role": "assistant", "content": final_text})
    new_history = new_history[-16:]
    await save_session(chat_id, {
        "history":           new_history,
        "ordering_platform": platform,
    })

    return final_text
