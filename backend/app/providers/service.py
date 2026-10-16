"""Provider mode summaries, receipts, reconciliation, tracking, and cancellation."""

import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.db.models import ProviderAttempt, ProviderReceipt
from app.db.session import AsyncSessionLocal
from app.providers.base import AdapterResult, ProviderMode
from app.providers.mcp_adapter import MCPProviderAdapter
from app.providers.redaction import content_hash, redact_text, redact_value
from app.providers.registry import current_policy, public_policy


def provider_handoff_url(provider: str) -> str:
    return settings.ZOMATO_URL if provider.upper() == "ZOMATO" else settings.SWIGGY_URL


async def provider_experience(provider: str) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        policy = await current_policy(db, provider)
    result = public_policy(policy)
    result["mode_label"] = {
        ProviderMode.DISCOVERY: "Mode A — discovery",
        ProviderMode.HANDOFF: "Mode B — user-controlled handoff",
        ProviderMode.TRANSACTIONAL: "Mode C — verified transactional",
    }[policy.mode]
    result["handoff_url"] = provider_handoff_url(provider) if policy.mode == ProviderMode.HANDOFF else None
    return result


def _reference(attempt: ProviderAttempt) -> dict[str, Any]:
    return {
        "provider_order_id": attempt.provider_order_id,
        "request_id": attempt.provider_request_id,
        "idempotency_key": attempt.idempotency_key,
    }


def add_receipt(
    db,
    *,
    provider: str,
    mode: str,
    operation: str,
    request: dict[str, Any],
    response: Any,
    outcome: str,
    restaurant_id: int,
    attempt_id: str | None = None,
    capability_version_id: int | None = None,
    provider_request_id: str | None = None,
    provider_order_id: str | None = None,
    tool_name: str | None = None,
    tool_schema_hash: str | None = None,
) -> ProviderReceipt:
    redacted_request = redact_value(request)
    redacted_response = redact_value(response)
    receipt = ProviderReceipt(
        restaurant_id=restaurant_id,
        provider_attempt_id=attempt_id,
        capability_version_id=capability_version_id,
        provider=provider.upper(),
        mode=mode,
        operation=operation,
        tool_name=tool_name,
        request_fingerprint=content_hash(redacted_request),
        request_redacted_json=redacted_request if isinstance(redacted_request, dict) else {"value": redacted_request},
        response_redacted_json=redacted_response if isinstance(redacted_response, dict) else {"value": redacted_response},
        response_hash=content_hash(redacted_response),
        outcome=outcome,
        provider_request_id=provider_request_id,
        provider_order_id=provider_order_id,
        tool_schema_hash=tool_schema_hash,
    )
    db.add(receipt)
    return receipt


async def persist_tool_receipts(
    *,
    provider: str,
    mode: str,
    channel_type: str,
    external_chat_id: str | int,
    records: list[dict[str, Any]],
    attempt_id: str | None,
    capability_version_id: int | None,
    tool_schema_hash: str | None,
) -> None:
    if not records:
        return
    async with AsyncSessionLocal() as db:
        for record in records:
            request = record.get("request") or {}
            response = record.get("response") or {}
            add_receipt(
                db,
                provider=provider,
                mode=mode,
                operation=record.get("operation") or "TOOL_CALL",
                request={
                    "channel_type": channel_type,
                    "external_chat_fingerprint": hashlib.sha256(str(external_chat_id).encode()).hexdigest(),
                    "arguments": request,
                },
                response=response,
                outcome=record.get("outcome") or "RETURNED",
                restaurant_id=settings.DEFAULT_RESTAURANT_ID,
                attempt_id=attempt_id,
                capability_version_id=capability_version_id,
                provider_request_id=record.get("provider_request_id"),
                provider_order_id=record.get("provider_order_id"),
                tool_name=record.get("tool_name"),
                tool_schema_hash=tool_schema_hash,
            )
        await db.commit()


async def _load_adapter(provider: str):
    from app.agent.mcp_clients import get_mcp_tools_async

    tools = await get_mcp_tools_async(provider.title())
    async with AsyncSessionLocal() as db:
        policy = await current_policy(db, provider)
    return MCPProviderAdapter(provider, tools, policy), policy


async def reconcile_attempt(attempt_id: str) -> AdapterResult:
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id)
        )).scalar_one_or_none()
        if not attempt:
            raise ValueError("Provider attempt not found")
        if attempt.status != "UNKNOWN":
            return AdapterResult(attempt.status, attempt.receipt_json or {}, attempt.provider_order_id)
        provider = attempt.provider
        reference = _reference(attempt)

    adapter, policy = await _load_adapter(provider)
    if not policy.allows_transactional_mode or not policy.capabilities.order_status_lookup:
        result = AdapterResult("UNKNOWN", {"reason": "STATUS_LOOKUP_NOT_VERIFIED"})
    else:
        try:
            result = await adapter.get_order_status(reference)
        except Exception as exc:
            result = AdapterResult("UNKNOWN", {"error": redact_text(str(exc))})

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id).with_for_update()
        )).scalar_one()
        attempt.reconciliation_count += 1
        attempt.last_reconciliation_at = now
        attempt.reconciled_at = now if result.outcome in {"CONFIRMED", "FAILED"} else None
        attempt.reconciliation_required = result.outcome == "UNKNOWN"
        attempt.next_action = "RECONCILE_BEFORE_RETRY" if result.outcome == "UNKNOWN" else None
        attempt.status = result.outcome
        attempt.provider_order_id = result.provider_order_id or attempt.provider_order_id
        attempt.receipt_json = redact_value(result.response)
        attempt.response_hash = content_hash(result.response)
        attempt.last_error = None if result.outcome != "UNKNOWN" else "Provider status remains uncertain"
        add_receipt(
            db,
            provider=attempt.provider,
            mode=attempt.mode,
            operation="RECONCILE_STATUS",
            request=reference,
            response=result.response,
            outcome=result.outcome,
            restaurant_id=attempt.restaurant_id,
            attempt_id=attempt.id,
            capability_version_id=attempt.capability_version_id,
            provider_order_id=attempt.provider_order_id,
            tool_schema_hash=attempt.tool_schema_hash,
        )
        await db.commit()
    return result


async def track_attempt(attempt_id: str) -> AdapterResult:
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id)
        )).scalar_one_or_none()
        if not attempt:
            raise ValueError("Provider attempt not found")
        provider, reference = attempt.provider, _reference(attempt)
    adapter, policy = await _load_adapter(provider)
    if not policy.allows_transactional_mode or not policy.capabilities.order_status_lookup:
        raise ValueError("Provider tracking capability is not verified")
    result = await adapter.get_order_status(reference)
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id).with_for_update()
        )).scalar_one()
        add_receipt(
            db,
            provider=attempt.provider,
            mode=attempt.mode,
            operation="TRACK_ORDER",
            request=reference,
            response=result.response,
            outcome=result.outcome,
            restaurant_id=attempt.restaurant_id,
            attempt_id=attempt.id,
            capability_version_id=attempt.capability_version_id,
            provider_order_id=result.provider_order_id or attempt.provider_order_id,
            tool_schema_hash=attempt.tool_schema_hash,
        )
        attempt.provider_order_id = result.provider_order_id or attempt.provider_order_id
        attempt.receipt_json = redact_value(result.response)
        attempt.response_hash = content_hash(result.response)
        await db.commit()
    return result


async def cancel_attempt(attempt_id: str) -> AdapterResult:
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id)
        )).scalar_one_or_none()
        if not attempt:
            raise ValueError("Provider attempt not found")
        provider, reference = attempt.provider, _reference(attempt)
    adapter, policy = await _load_adapter(provider)
    if not policy.allows_transactional_mode or not policy.capabilities.cancellation:
        raise ValueError("Provider cancellation capability is not verified")
    result = await adapter.cancel_order(reference)
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt).where(ProviderAttempt.id == attempt_id).with_for_update()
        )).scalar_one()
        add_receipt(
            db,
            provider=attempt.provider,
            mode=attempt.mode,
            operation="CANCEL_ORDER",
            request=reference,
            response=result.response,
            outcome=result.outcome,
            restaurant_id=attempt.restaurant_id,
            attempt_id=attempt.id,
            capability_version_id=attempt.capability_version_id,
            provider_order_id=result.provider_order_id or attempt.provider_order_id,
            tool_schema_hash=attempt.tool_schema_hash,
        )
        if result.outcome == "CANCELLED":
            attempt.status = "CANCELLED"
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.next_action = None
        else:
            attempt.status = "UNKNOWN"
            attempt.reconciliation_required = True
            attempt.next_action = "RECONCILE_CANCELLATION"
        await db.commit()
    return result
