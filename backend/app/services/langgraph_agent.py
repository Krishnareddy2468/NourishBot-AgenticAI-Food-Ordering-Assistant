import json
import logging
from typing import Any

from pydantic import BaseModel, create_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import app_config
from app.models.schemas import SearchFilters
from app.services.session_service import session_service
from app.services.zomato_mcp import global_zomato_mcp

logger = logging.getLogger(__name__)


class _EmptyArgs(BaseModel):
    pass


class LangGraphFoodAgent:
    def __init__(self):
        self._compiled_graph = None
        self._tool_cache: list[StructuredTool] = []

    def _build_system_prompt(self, user_location: str | None, filters: SearchFilters | None) -> str:
        location_text = user_location or "unknown"
        filter_text = "none"
        if filters:
            filter_text = (
                f"veg_only={filters.veg_only}, non_veg_only={filters.non_veg_only}, "
                f"min_rating={filters.min_rating}, max_distance_km={filters.max_distance_km}"
            )
        return (
            "You are a food ordering assistant.\n"
            "Rules:\n"
            "1) Use MCP tools for restaurants/menu/order data. Never invent data.\n"
            "2) Ask follow-up questions when required fields are missing.\n"
            "3) Keep responses short and actionable.\n"
            "4) Confirm before order placement.\n"
            f"Current user location context: {location_text}\n"
            f"Current active filters: {filter_text}\n"
        )

    def _extract_output_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            return "\n".join(c for c in chunks if c).strip()
        return str(content or "")

    def _extract_thinking_steps(self, messages: list[Any]) -> list[str]:
        steps: list[str] = []
        for msg in messages:
            tool_name = getattr(msg, "name", None)
            if tool_name:
                steps.append(f"Used tool: {tool_name}")
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                if isinstance(call, dict) and call.get("name"):
                    steps.append(f"Planned tool call: {call['name']}")
        deduped = []
        seen = set()
        for step in steps:
            if step in seen:
                continue
            seen.add(step)
            deduped.append(step)
        return deduped[:8]

    def _json_schema_to_args_model(self, name: str, schema: dict[str, Any] | None) -> type[BaseModel]:
        if not schema:
            return _EmptyArgs

        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        fields = {}

        type_map: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list[Any],
            "object": dict[str, Any],
        }

        for key, details in properties.items():
            json_type = "string"
            if isinstance(details, dict):
                json_type = details.get("type", "string")
            py_type = type_map.get(json_type, Any)
            default = ... if key in required else None
            fields[key] = (py_type, default)

        if not fields:
            return _EmptyArgs
        return create_model(f"{name.title().replace('_', '')}Args", **fields)

    async def _ensure_graph(self):
        if self._compiled_graph is not None:
            return
        if not app_config.llm_api_key:
            logger.warning("LLM API key is missing. Set LLM_API_KEY or GEMINI_API_KEY.")
            return

        mcp_tools = await global_zomato_mcp.get_tools()
        tool_defs: list[StructuredTool] = []
        for tool in mcp_tools:
            args_schema = self._json_schema_to_args_model(tool.name, tool.inputSchema or {})

            async def _runner(_tool_name: str = tool.name, **kwargs):
                result = await global_zomato_mcp.call_tool(_tool_name, kwargs)
                text_parts = [part for part in result if isinstance(part, str)]
                if text_parts:
                    return "\n".join(text_parts)
                return json.dumps(result, default=str)

            tool_defs.append(
                StructuredTool.from_function(
                    coroutine=_runner,
                    name=tool.name,
                    description=tool.description or tool.name,
                    args_schema=args_schema,
                )
            )

        llm = ChatOpenAI(
            model=app_config.llm_model,
            base_url=app_config.llm_base_url,
            api_key=app_config.llm_api_key,
            temperature=app_config.llm_temperature,
        )
        self._tool_cache = tool_defs
        self._compiled_graph = create_react_agent(llm, tool_defs)
        logger.info("LangGraph agent initialized with %d MCP tools", len(tool_defs))

    async def process_message(
        self,
        user_id: str,
        message: str,
        user_name: str | None = None,
        user_location: str | None = None,
        filters: SearchFilters | None = None,
        channel: str = "web",
        input_mode: str = "text",
    ) -> tuple[str, list[str]]:
        session = session_service.get_session(user_id, user_name)
        if user_location:
            session.current_location = user_location

        await self._ensure_graph()
        if self._compiled_graph is None:
            fallback = "LLM is not configured. Please set LLM_API_KEY/LLM_MODEL and retry."
            session_service.add_to_history(user_id, "user", message)
            session_service.add_to_history(user_id, "assistant", fallback)
            session_service.set_last_bot_message(user_id, fallback)
            return fallback, ["LLM unavailable"]

        history_messages = []
        for item in session.conversation_history[-10:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

        prompt = self._build_system_prompt(session.current_location, filters)
        runtime_context = (
            f"Channel={channel}; InputMode={input_mode}; "
            "For voice input, be concise and confirm key details before placing an order."
        )

        graph_input = {
            "messages": [
                SystemMessage(content=prompt),
                SystemMessage(content=runtime_context),
                *history_messages,
                HumanMessage(content=message),
            ]
        }

        result = await self._compiled_graph.ainvoke(graph_input)
        response_messages = result.get("messages", [])
        final_text = ""
        if response_messages:
            final_text = self._extract_output_text(response_messages[-1].content)
        if not final_text:
            final_text = "I could not generate a response. Please try again."

        session_service.add_to_history(user_id, "user", message)
        session_service.add_to_history(user_id, "assistant", final_text)
        session_service.set_last_bot_message(user_id, final_text)
        return final_text, self._extract_thinking_steps(response_messages)


langgraph_food_agent = LangGraphFoodAgent()
