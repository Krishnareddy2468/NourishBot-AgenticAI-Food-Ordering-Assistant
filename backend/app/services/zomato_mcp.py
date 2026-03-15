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
        self._connecting = False

    async def connect(self):
        if self._session:
            return

        # Don't hammer a broken server — wait for cooldown before retrying
        if self._connect_failed_at and (time.time() - self._connect_failed_at) < MCP_RETRY_COOLDOWN:
            remaining = int(MCP_RETRY_COOLDOWN - (time.time() - self._connect_failed_at))
            logger.warning("Skipping MCP connect — in cooldown for %ds more", remaining)
            return

        if self._connecting:
            return
        self._connecting = True

        logger.info("Connecting to Zomato MCP Server (timeout=%ds)...", MCP_CONNECT_TIMEOUT)
        server_params = StdioServerParameters(
            command=app_config.mcp_command,
            args=app_config.mcp_args,
        )

        try:
            async def _do_connect():
                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self.read, self.write = stdio_transport
                self._session = await self._exit_stack.enter_async_context(
                    ClientSession(self.read, self.write)
                )
                await self._session.initialize()
                tools_response = await self._session.list_tools()
                self._tools = tools_response.tools

            await asyncio.wait_for(_do_connect(), timeout=MCP_CONNECT_TIMEOUT)
            tool_names = [t.name for t in self._tools]
            logger.info("Zomato MCP connected — %d tools available: %s", len(self._tools), tool_names)
            # Log tool schemas for debugging parameter issues
            for t in self._tools:
                logger.debug("Tool schema: %s — params: %s", t.name, t.inputSchema)
            self._connect_failed_at = 0
        except asyncio.TimeoutError:
            logger.error("Zomato MCP connect TIMED OUT after %ds (OAuth discovery hung)", MCP_CONNECT_TIMEOUT)
            self._session = None
            self._connect_failed_at = time.time()
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = AsyncExitStack()
        except Exception as e:
            logger.error("Zomato MCP connect FAILED: %s", e)
            self._session = None
            self._connect_failed_at = time.time()
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = AsyncExitStack()
        finally:
            self._connecting = False

    async def get_tools(self):
        if not self._session:
            await self.connect()
        return self._tools

    def get_tool_names(self) -> list:
        """Return a list of available tool names for validation."""
        return [t.name for t in self._tools]

    async def call_tool(self, name: str, arguments: dict):
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
            return [f"Tool {name} timed out — Zomato server is slow. Try a more specific search."]
        except Exception as e:
            logger.error("Tool %s error: %s", name, e)
            return [f"Tool {name} failed: {e}"]

    async def close(self):
        try:
            await self._exit_stack.aclose()
        except Exception:
            pass
        self._session = None

global_zomato_mcp = ZomatoMCP()
