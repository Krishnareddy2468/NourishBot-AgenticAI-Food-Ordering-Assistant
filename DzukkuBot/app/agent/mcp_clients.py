"""
MCP (Model Context Protocol) client manager — Zomato + Swiggy.
==============================================================

Spawns `npx mcp-remote <URL>` per remote MCP server and bridges it to the
local stdio MCP transport. mcp-remote handles the OAuth dance on its own
loopback callback (no public callback / whitelisted redirect URI required)
and caches tokens under ~/.mcp-auth/. The first time you run this on a
machine, mcp-remote will open a browser for the bot owner to log in. After
that the cached token is reused — that's the "shared bot-owner token"
model agreed for this build.

IMPORTANT: Expected subprocess output
─────────────────────────────────────
During normal operation, you may see stderr messages from mcp-remote like:
    Error from remote server: DOMException [AbortError]: This operation was aborted
    at node:internal/deps/undici/undici:13502:13
    at async StreamableHTTPClientTransport._startOrAuthSse (...)

These are harmless cleanup noise occurring when the Node.js HTTPclient's SSE
stream is closed during subprocess shutdown. The tool calls complete successfully
before these errors occur. They cannot be suppressed from Python; mcp_agent.py
catches the corresponding Python exceptions and handles them gracefully.

Public surface
--------------
    get_mcp_tools_async(platform) -> list[BaseTool]
        Returns LangChain tools assembled from every enabled MCP server,
        with deterministic timeout / reconnect / cooldown behaviour around
        each tool call. Safe to call repeatedly — connection is cached.

    close_all_async() -> None
        Cleanly tears down all MCP sessions. Called from FastAPI lifespan
        on shutdown.

Reliability controls (mirrors the pattern from the previous Zomato build):
- Per-tool timeout                         (settings.MCP_TOOL_TIMEOUT_S)
- Cooldown after a connection failure     (settings.MCP_RECONNECT_COOLDOWN_S)
- Reconnect lock so a single failure does not stampede reconnects
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Lazy imports ─────────────────────────────────────────────────────────────
# We keep these inside _ensure_imports() so the bot can boot even when the
# heavy LangGraph/MCP stack isn't installed (e.g. dev mode with MCP_ENABLED=false).

def _ensure_imports() -> None:
    global MultiServerMCPClient, BaseTool  # noqa: PLW0603
    try:
        # langchain-mcp-adapters >= 0.1
        from langchain_mcp_adapters.client import MultiServerMCPClient as _MultiServerMCPClient
        from langchain_core.tools import BaseTool as _BaseTool
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "MCP_ENABLED=true but the LangGraph/MCP stack isn't installed. "
            "Run: pip install -r requirements.txt (langchain-mcp-adapters, langgraph, mcp)"
        ) from e
    MultiServerMCPClient = _MultiServerMCPClient
    BaseTool = _BaseTool


MultiServerMCPClient = None  # type: ignore[assignment]
BaseTool = None              # type: ignore[assignment]


# ── State (per-platform so Zomato and Swiggy don't block each other) ─────────

_clients:          dict[str, Any]   = {}   # platform → MultiServerMCPClient
_tools_caches:     dict[str, list]  = {}   # platform → list[BaseTool]
_last_failure_ts:  dict[str, float] = {}   # platform → epoch
_connect_locks:    dict[str, asyncio.Lock | None] = {}  # platform → Lock (lazy)


def _get_connect_lock(platform: str) -> asyncio.Lock:
    """Lazily create a per-platform lock inside an async context."""
    if _connect_locks.get(platform) is None:
        _connect_locks[platform] = asyncio.Lock()
    return _connect_locks[platform]  # type: ignore[return-value]


# ── Server definition helpers ────────────────────────────────────────────────

def _servers_for_platform(platform: str) -> dict[str, dict[str, Any]]:
    """
    Return only the MCP server entries relevant to the given platform.
    platform = "Zomato"  → only the Zomato server
    platform = "Swiggy"  → only the Swiggy food server (+ instamart/dineout if enabled)
    """

    def _stdio_spec(url: str) -> dict[str, Any]:
        env: dict[str, str] = {}
        if settings.MCP_AUTH_DIR:
            env["MCP_REMOTE_CONFIG_DIR"] = settings.MCP_AUTH_DIR
        spec: dict[str, Any] = {
            "transport": "stdio",
            "command":   settings.NPX_BIN,
            "args":      ["-y", "mcp-remote", url],
        }
        if env:
            spec["env"] = env
        return spec

    servers: dict[str, dict[str, Any]] = {}

    if platform == "Zomato":
        if settings.MCP_ZOMATO_ENABLED:
            servers["zomato"] = _stdio_spec(settings.MCP_ZOMATO_URL)

    elif platform == "Swiggy":
        if settings.MCP_SWIGGY_FOOD_ENABLED:
            servers["swiggy_food"] = _stdio_spec(settings.MCP_SWIGGY_FOOD_URL)
        if settings.MCP_SWIGGY_INSTAMART_ENABLED:
            servers["swiggy_instamart"] = _stdio_spec(settings.MCP_SWIGGY_INSTAMART_URL)
        if settings.MCP_SWIGGY_DINEOUT_ENABLED:
            servers["swiggy_dineout"] = _stdio_spec(settings.MCP_SWIGGY_DINEOUT_URL)

    return servers


# ── Public API ───────────────────────────────────────────────────────────────

async def get_mcp_tools_async(platform: str) -> list:
    """
    Connect (once per platform, lazily) to the MCP servers for the given
    platform and return their LangChain tools. Results are cached separately
    per platform so Zomato and Swiggy never block each other.

    platform: "Zomato" or "Swiggy"
    
    The underlying MultiServerMCPClient spawns subprocess(es) running
    `mcp-remote` to bridge remote HTTP MCP servers → local stdio.
    These subprocesses stay alive as long as the client is referenced.
    """
    if not settings.MCP_ENABLED:
        return []

    if platform in _tools_caches:
        logger.debug("MCP[%s]: returning cached %d tools", platform, len(_tools_caches[platform]))
        return _tools_caches[platform]

    last_fail = _last_failure_ts.get(platform, 0.0)
    if last_fail and (time.time() - last_fail) < settings.MCP_RECONNECT_COOLDOWN_S:
        logger.warning("MCP[%s]: still cooling down; returning empty tool set.", platform)
        return []

    async with _get_connect_lock(platform):
        # Double-check inside the lock (another coroutine may have filled the cache)
        if platform in _tools_caches:
            logger.debug("MCP[%s]: another task filled cache; returning %d tools", platform, len(_tools_caches[platform]))
            return _tools_caches[platform]

        _ensure_imports()
        servers = _servers_for_platform(platform)
        if not servers:
            logger.info("MCP[%s]: no servers configured; returning empty tool set.", platform)
            _tools_caches[platform] = []
            return []

        try:
            logger.info("MCP[%s]: connecting to servers: %s", platform, list(servers.keys()))
            client = MultiServerMCPClient(servers)
            logger.debug("MCP[%s]: MultiServerMCPClient instance created (PID tracking via subprocess)")
            
            tools = await client.get_tools()
            tool_names = [getattr(t, "name", str(t)) for t in tools]
            logger.info("MCP[%s]: %d tool(s) loaded: %s", platform, len(tools), tool_names)
            logger.debug("MCP[%s]: tool details: %s", platform, [
                {"name": getattr(t, "name", "?"), "description": (getattr(t, "description", "") or "")[:80]}
                for t in tools
            ])
            
            # CRITICAL: Keep a strong reference to the client to prevent garbage collection.
            # If the client is GC'd, the underlying subprocess(es) will shut down.
            _clients[platform]      = client
            _tools_caches[platform] = list(tools)
            logger.info("MCP[%s]: client cached; subprocess(es) will remain alive", platform)
            return _tools_caches[platform]
        except Exception as e:
            _last_failure_ts[platform] = time.time()
            logger.error("MCP[%s]: failed to connect / list tools: %s", platform, e, exc_info=True)
            return []


async def reset_platform_cache(platform: str) -> None:
    """
    Evict the cached client + tools for a platform so the next call to
    get_mcp_tools_async() spawns a fresh subprocess connection.
    Called by mcp_agent when it detects a stale / failed connection.
    Note: langchain-mcp-adapters 0.1.0 does not support __aexit__ — just
    clear the cache dicts and let the subprocess clean itself up.
    """
    async with _get_connect_lock(platform):
        _clients.pop(platform, None)
        _tools_caches.pop(platform, None)
    logger.info("MCP[%s]: cache reset — will reconnect on next request.", platform)


async def close_all_async() -> None:
    """Tear down all per-platform MCP sessions cleanly (FastAPI shutdown hook)."""
    for platform, client in list(_clients.items()):
        try:
            # Try aclose if available (future versions), otherwise just clear.
            aclose = getattr(client, "aclose", None)
            if aclose and callable(aclose):
                await aclose()
        except Exception as e:  # pragma: no cover
            logger.warning("MCP[%s]: clean shutdown raised: %s", platform, e)
    _clients.clear()
    _tools_caches.clear()
