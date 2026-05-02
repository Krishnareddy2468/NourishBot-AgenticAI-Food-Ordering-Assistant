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
    """
    if not settings.MCP_ENABLED:
        return []

    if platform in _tools_caches:
        return _tools_caches[platform]

    last_fail = _last_failure_ts.get(platform, 0.0)
    if last_fail and (time.time() - last_fail) < settings.MCP_RECONNECT_COOLDOWN_S:
        logger.warning("MCP[%s]: still cooling down; returning empty tool set.", platform)
        return []

    async with _get_connect_lock(platform):
        if platform in _tools_caches:
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
            tools = await client.get_tools()
            logger.info("MCP[%s]: %d tool(s) loaded.", platform, len(tools))
            _clients[platform]      = client
            _tools_caches[platform] = list(tools)
            return _tools_caches[platform]
        except Exception as e:
            _last_failure_ts[platform] = time.time()
            logger.error("MCP[%s]: failed to connect / list tools: %s", platform, e, exc_info=True)
            return []


async def close_all_async() -> None:
    """Tear down all per-platform MCP sessions cleanly (FastAPI shutdown hook)."""
    for platform, client in list(_clients.items()):
        try:
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:  # pragma: no cover
            logger.warning("MCP[%s]: clean shutdown raised: %s", platform, e)
    _clients.clear()
    _tools_caches.clear()
