"""MCP implementation of the provider adapter contract."""

import json
from typing import Any

from app.core.config import settings
from app.providers.base import AdapterResult, ProviderAdapter, ProviderPolicy
from app.providers.registry import (
    is_cancel_tool,
    is_status_tool,
    is_transaction_tool,
    tool_allowed,
    tool_schema,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    content = getattr(value, "content", value)
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        return {"content": content}
    text = str(content or "")
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except (TypeError, ValueError):
        return {"text": text}


def _field(payload: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = payload.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    for value in payload.values():
        if isinstance(value, dict):
            found = _field(value, *names)
            if found:
                return found
    return None


class MCPProviderAdapter(ProviderAdapter):
    def __init__(self, provider: str, tools: list[Any], policy: ProviderPolicy):
        self.provider = provider.upper()
        self.tools = tools
        self.policy = policy

    def _find(self, predicate, *, transaction_turn: bool = False):
        for tool in self.tools:
            name = getattr(tool, "name", "") or ""
            if predicate(name) and tool_allowed(self.policy, tool, transaction_turn=transaction_turn):
                return tool
        return None

    @staticmethod
    def _reference_args(tool: Any, reference: dict[str, Any]) -> dict[str, Any]:
        properties = (tool_schema(tool).get("properties") or {})
        for field in ("provider_order_id", "order_id", "request_id", "idempotency_key"):
            if field in properties and reference.get(field):
                return {field: reference[field]}
        return {
            key: value
            for key, value in reference.items()
            if value is not None and key in properties
        }

    @staticmethod
    async def _invoke(tool: Any, args: dict[str, Any]) -> dict[str, Any]:
        if hasattr(tool, "ainvoke"):
            return _as_dict(await tool.ainvoke(args))
        if hasattr(tool, "invoke"):
            return _as_dict(tool.invoke(args))
        raise RuntimeError("Provider tool does not expose invoke/ainvoke")

    async def discover(self, query: dict[str, Any]) -> AdapterResult:
        tool = self._find(
            lambda name: any(value in name.lower() for value in ("search", "discover", "restaurant"))
        )
        if tool is None:
            raise RuntimeError("Provider discovery capability is unavailable")
        properties = (tool_schema(tool).get("properties") or {})
        args = {key: value for key, value in query.items() if not properties or key in properties}
        if properties:
            if "query" in properties and "query" not in args:
                args["query"] = query.get("restaurant_name") or query.get("place_id") or "restaurant"
            for target in ("scheduled_time", "schedule_time", "delivery_time", "fulfillment_time", "preorder_time"):
                if target in properties and target not in args and query.get("scheduled_for"):
                    args[target] = query["scheduled_for"]
            if "fulfilment_type" in properties and query.get("fulfillment_type"):
                args["fulfilment_type"] = query["fulfillment_type"]
        response = await self._invoke(tool, args)
        return AdapterResult("SUCCEEDED", response)

    async def create_handoff(self, context: dict[str, Any]) -> AdapterResult:
        url = settings.ZOMATO_URL if self.provider == "ZOMATO" else settings.SWIGGY_URL
        if not url:
            raise RuntimeError("Provider handoff URL is not configured")
        return AdapterResult("HANDOFF_READY", {"url": url, "context": context})

    async def submit_scheduled_order(self, request: dict[str, Any]) -> AdapterResult:
        if not settings.provider_transactions_enabled(self.provider):
            raise RuntimeError("Public transactional release gate is closed")
        tool = self._find(is_transaction_tool, transaction_turn=True)
        if tool is None:
            raise RuntimeError("Verified scheduled transaction capability is unavailable")
        response = await self._invoke(tool, request)
        provider_order_id = _field(response, "provider_order_id", "order_id", "orderId")
        outcome = "CONFIRMED" if provider_order_id else "UNKNOWN"
        return AdapterResult(outcome, response, provider_order_id=provider_order_id)

    async def get_order_status(self, reference: dict[str, Any]) -> AdapterResult:
        tool = self._find(is_status_tool)
        if tool is None:
            raise RuntimeError("Verified provider status lookup is unavailable")
        response = await self._invoke(tool, self._reference_args(tool, reference))
        provider_order_id = _field(response, "provider_order_id", "order_id", "orderId")
        status = (_field(response, "status", "order_status", "state") or "UNKNOWN").upper()
        if status in {"CONFIRMED", "ACCEPTED", "PLACED", "PREPARING", "READY", "DELIVERED", "COMPLETED"}:
            outcome = "CONFIRMED"
        elif status in {"FAILED", "REJECTED", "CANCELLED", "NOT_FOUND", "EXPIRED"}:
            outcome = "FAILED"
        else:
            outcome = "UNKNOWN"
        return AdapterResult(outcome, response, provider_order_id=provider_order_id)

    async def cancel_order(self, reference: dict[str, Any]) -> AdapterResult:
        tool = self._find(is_cancel_tool)
        if tool is None:
            raise RuntimeError("Verified provider cancellation is unavailable")
        response = await self._invoke(tool, self._reference_args(tool, reference))
        status = (_field(response, "status", "order_status", "state") or "UNKNOWN").upper()
        outcome = "CANCELLED" if status in {"CANCELLED", "SUCCESS", "SUCCEEDED"} else "UNKNOWN"
        return AdapterResult(
            outcome,
            response,
            provider_order_id=_field(response, "provider_order_id", "order_id", "orderId"),
        )
