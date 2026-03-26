import asyncio
import logging
import time
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from app.config import app_config

logger = logging.getLogger(__name__)

MCP_CONNECT_TIMEOUT = app_config.mcp_connect_timeout
MCP_TOOL_TIMEOUT = app_config.mcp_tool_timeout
MCP_RETRY_COOLDOWN = app_config.mcp_retry_cooldown

class ZomatoMCP:
    def __init__(self):
        self._session = None
        self._exit_stack = AsyncExitStack()
        self._tools = []
        self._connect_failed_at: float = 0   # epoch time of last failed connect
        self._connect_lock = asyncio.Lock()

    async def _reset_connection(self, reason: str):
        logger.warning("Resetting Zomato MCP connection: %s", reason)
        self._session = None
        self._tools = []
        try:
            await self._exit_stack.aclose()
        except Exception:
            pass
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        if self._session:
            return
        async with self._connect_lock:
            if self._session:
                return

            # Don't hammer a broken server — wait for cooldown before retrying
            if self._connect_failed_at and (time.time() - self._connect_failed_at) < MCP_RETRY_COOLDOWN:
                remaining = int(MCP_RETRY_COOLDOWN - (time.time() - self._connect_failed_at))
                logger.warning("Skipping MCP connect — in cooldown for %ds more", remaining)
                return

            logger.info("Connecting to Zomato MCP Server (timeout=%ds)...", MCP_CONNECT_TIMEOUT)
            server_params = StdioServerParameters(
                command=app_config.mcp_command,
                args=app_config.mcp_args,
            )

            try:
                next_exit_stack = AsyncExitStack()

                async def _do_connect():
                    stdio_transport = await next_exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read, write = stdio_transport
                    session = await next_exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    tools_response = await session.list_tools()
                    return session, tools_response.tools

                session, tools = await asyncio.wait_for(_do_connect(), timeout=MCP_CONNECT_TIMEOUT)
                self._session = session
                self._tools = tools
                self._exit_stack = next_exit_stack

                tool_names = [t.name for t in self._tools]
                logger.info("Zomato MCP connected — %d tools available: %s", len(self._tools), tool_names)
                for t in self._tools:
                    logger.debug("Tool schema: %s — params: %s", t.name, t.inputSchema)
                self._connect_failed_at = 0
            except asyncio.TimeoutError:
                logger.error("Zomato MCP connect TIMED OUT after %ds (OAuth discovery hung)", MCP_CONNECT_TIMEOUT)
                self._connect_failed_at = time.time()
                await self._reset_connection("connect timeout")
            except Exception as e:
                logger.error("Zomato MCP connect FAILED: %s", e)
                self._connect_failed_at = time.time()
                await self._reset_connection(f"connect failure: {e}")

    def _is_recoverable_tool_error(self, error: Exception) -> bool:
        message = str(error).lower()
        recoverable_markers = (
            "fetch failed",
            "connect timeout",
            "connection closed",
            "closedresourceerror",
            "broken pipe",
            "eof",
            "socket",
        )
        return any(marker in message for marker in recoverable_markers)

    async def get_tools(self):
        if not self._session:
            await self.connect()
        return self._tools

    def get_tool_names(self) -> list:
        """Return a list of available tool names for validation."""
        return [t.name for t in self._tools]

    async def call_tool(self, name: str, arguments: dict, _allow_retry: bool = True):
        if not self._session:
            await self.connect()
        if not self._session:
            return [f"MCP unavailable — cannot call {name}"]
        # Validate tool name before calling
        available = self.get_tool_names()
        if available and name not in available:
            logger.warning("Tool '%s' not found. Available tools: %s", name, available)
            return [f"Tool '{name}' not available. Available: {', '.join(available)}"]
        logger.info("Calling Zomato MCP Tool: %s %s", name, arguments)
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments=arguments),
                timeout=MCP_TOOL_TIMEOUT,
            )
            if result.content:
                texts = [c.text for c in result.content]
                logger.info("Tool %s returned %d content chunks, first 300 chars: %s",
                            name, len(texts), texts[0][:300] if texts else "")
                return texts
            return ["Tool returned no output."]
        except asyncio.TimeoutError:
            logger.error("Tool %s TIMED OUT after %ds", name, MCP_TOOL_TIMEOUT)
            await self._reset_connection(f"tool timeout while calling {name}")
            if _allow_retry:
                logger.info("Retrying tool %s once after timeout", name)
                return await self.call_tool(name, arguments, _allow_retry=False)
            return [f"Tool {name} timed out — Zomato server is slow. Try a more specific search."]
        except Exception as e:
            logger.error("Tool %s error: %s", name, e)
            if self._is_recoverable_tool_error(e):
                await self._reset_connection(f"recoverable tool error while calling {name}: {e}")
                if _allow_retry:
                    logger.info("Retrying tool %s once after recoverable error", name)
                    return await self.call_tool(name, arguments, _allow_retry=False)
            return [f"Tool {name} failed: {e}"]

    async def call_tool_raw(self, name: str, arguments: dict, _allow_retry: bool = True):
        if not self._session:
            await self.connect()
        if not self._session:
            return []
        available = self.get_tool_names()
        if available and name not in available:
            logger.warning("Tool '%s' not found. Available tools: %s", name, available)
            return []
        logger.info("Calling Zomato MCP Tool (raw): %s %s", name, arguments)
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments=arguments),
                timeout=MCP_TOOL_TIMEOUT,
            )
            content = list(result.content or [])
            preview = ""
            for chunk in content:
                text = getattr(chunk, "text", None)
                if text:
                    preview = text[:300]
                    break
            logger.info(
                "Tool %s returned %d raw content chunks, first text preview: %s",
                name,
                len(content),
                preview,
            )
            return content
        except asyncio.TimeoutError:
            logger.error("Tool %s TIMED OUT after %ds", name, MCP_TOOL_TIMEOUT)
            await self._reset_connection(f"tool timeout while calling {name}")
            if _allow_retry:
                logger.info("Retrying raw tool %s once after timeout", name)
                return await self.call_tool_raw(name, arguments, _allow_retry=False)
            return []
        except Exception as e:
            logger.error("Raw tool %s error: %s", name, e)
            if self._is_recoverable_tool_error(e):
                await self._reset_connection(f"recoverable raw tool error while calling {name}: {e}")
                if _allow_retry:
                    logger.info("Retrying raw tool %s once after recoverable error", name)
                    return await self.call_tool_raw(name, arguments, _allow_retry=False)
            return []

    async def close(self):
        try:
            await self._exit_stack.aclose()
        except Exception:
            pass
        self._session = None

global_zomato_mcp = ZomatoMCP()
