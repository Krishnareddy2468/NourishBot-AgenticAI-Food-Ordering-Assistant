"""Versioned provider capability policy and MCP schema inventory."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.db.models import ProviderCapabilityVersion
from app.providers.base import CapabilityState, ProviderCapabilities, ProviderMode, ProviderPolicy
from app.providers.redaction import content_hash


READ_ONLY_MARKERS = (
    "search", "discover", "restaurant", "menu", "catalog", "price", "availability", "address",
)
TRANSACTION_MARKERS = (
    "checkout", "place_order", "create_order", "submit_order", "confirm_order", "purchase", "pay_order",
)
CANCEL_MARKERS = ("cancel_order", "cancel_purchase", "cancel_booking")
STATUS_MARKERS = ("order_status", "track_order", "get_order", "lookup_order")
OTHER_MUTATION_MARKERS = (
    "add_", "update_", "delete_", "remove_", "set_", "apply_", "clear_", "create_", "submit_", "confirm_",
)
SCHEDULE_FIELDS = (
    "scheduled_for", "scheduled_time", "schedule_time", "delivery_time", "fulfillment_time", "preorder_time",
)
IDEMPOTENCY_FIELDS = ("idempotency_key", "request_id", "client_request_id")


def tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    try:
        if schema is not None and hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        if schema is not None and hasattr(schema, "schema"):
            return schema.schema()
    except Exception:
        return {}
    return {}


def tool_inventory(tools: list[Any]) -> dict[str, Any]:
    records = []
    for tool in tools:
        records.append({
            "name": getattr(tool, "name", str(tool)),
            "description": (getattr(tool, "description", "") or "")[:1000],
            "schema": tool_schema(tool),
        })
    return {"tools": sorted(records, key=lambda item: item["name"])}


def infer_capabilities(tools: list[Any]) -> ProviderCapabilities:
    names = " ".join((getattr(tool, "name", "") or "").lower() for tool in tools)
    schemas = " ".join(str(tool_schema(tool)).lower() for tool in tools)
    return ProviderCapabilities(
        discovery=any(marker in names for marker in ("search", "discover", "restaurant")),
        menu_lookup="menu" in names or "catalog" in names,
        live_price_verification="price" in names or "cart" in names or "menu" in names,
        scheduled_ordering=(
            any(marker in names for marker in TRANSACTION_MARKERS)
            and any(field in schemas for field in SCHEDULE_FIELDS)
        ),
        delivery="delivery" in names or "delivery" in schemas,
        pickup="pickup" in names or "pickup" in schemas,
        curbside_drive_through=any(value in names or value in schemas for value in ("curbside", "drive_through", "drive-through")),
        cancellation=any(marker in names for marker in CANCEL_MARKERS),
        order_status_lookup=any(marker in names for marker in STATUS_MARKERS),
        user_owned_authentication=any(value in names for value in ("saved_address", "profile", "account")),
        stable_idempotency=any(field in schemas for field in IDEMPOTENCY_FIELDS),
    )


def policy_from_record(record: ProviderCapabilityVersion | None, provider: str) -> ProviderPolicy:
    if record is None:
        return ProviderPolicy(
            provider=provider.upper(),
            version_id=None,
            version=0,
            state=CapabilityState.UNVERIFIED,
            mode=ProviderMode.DISCOVERY,
            capabilities=ProviderCapabilities(),
        )
    return ProviderPolicy(
        provider=record.provider,
        version_id=record.id,
        version=record.version,
        state=CapabilityState(record.state),
        mode=ProviderMode(record.mode),
        capabilities=ProviderCapabilities.from_dict(record.capabilities_json),
        tool_schema_hash=record.tool_schema_hash,
        adapter_version=record.adapter_version,
    )


async def current_policy(db, provider: str) -> ProviderPolicy:
    record = (await db.execute(
        select(ProviderCapabilityVersion)
        .where(ProviderCapabilityVersion.provider == provider.upper())
        .order_by(ProviderCapabilityVersion.version.desc())
        .limit(1)
    )).scalar_one_or_none()
    return policy_from_record(record, provider)


async def sync_tool_inventory(db, provider: str, tools: list[Any]) -> ProviderPolicy:
    provider = provider.upper()
    # JSON schemas describe fields; they do not contain runtime request data.
    # Keep the complete schema so any change, including a nested address or
    # payment-field change, invalidates the previously verified hash.
    inventory = tool_inventory(tools)
    schema_hash = content_hash(inventory)
    current = await current_policy(db, provider)
    if current.tool_schema_hash == schema_hash:
        return current

    next_version = (await db.execute(
        select(func.coalesce(func.max(ProviderCapabilityVersion.version), 0) + 1)
        .where(ProviderCapabilityVersion.provider == provider)
    )).scalar_one()
    db.add(ProviderCapabilityVersion(
        provider=provider,
        version=next_version,
        state=CapabilityState.UNVERIFIED.value,
        mode=ProviderMode.DISCOVERY.value,
        capabilities_json=infer_capabilities(tools).as_dict(),
        tool_schema_hash=schema_hash,
        tool_schema_json=inventory,
        adapter_version="mcp-v1",
        verification_notes="Automatically observed tool inventory; requires explicit verification.",
    ))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
    return await current_policy(db, provider)


def validate_policy(state: str, mode: str, capabilities: ProviderCapabilities, schema_hash: str | None) -> None:
    state_value = CapabilityState(state)
    mode_value = ProviderMode(mode)
    if mode_value == ProviderMode.TRANSACTIONAL:
        if state_value != CapabilityState.VERIFIED:
            raise ValueError("Transactional mode requires VERIFIED capability state")
        if not schema_hash:
            raise ValueError("Transactional mode requires a recorded tool schema hash")
        if not (
            capabilities.scheduled_ordering
            and capabilities.order_status_lookup
            and capabilities.stable_idempotency
        ):
            raise ValueError(
                "Transactional mode requires scheduled ordering, order status lookup, and stable idempotency"
            )


def _schema_text(tool: Any) -> str:
    return str(tool_schema(tool)).lower()


def is_transaction_tool(name: str) -> bool:
    lowered = (name or "").lower()
    return any(marker in lowered for marker in TRANSACTION_MARKERS)


def is_cancel_tool(name: str) -> bool:
    lowered = (name or "").lower()
    return any(marker in lowered for marker in CANCEL_MARKERS)


def is_status_tool(name: str) -> bool:
    lowered = (name or "").lower()
    return any(marker in lowered for marker in STATUS_MARKERS)


def tool_allowed(policy: ProviderPolicy, tool: Any, *, transaction_turn: bool) -> bool:
    name = (getattr(tool, "name", "") or "").lower()
    if policy.state == CapabilityState.DISABLED:
        return False
    if is_transaction_tool(name):
        schema = _schema_text(tool)
        return (
            transaction_turn
            and policy.allows_transactional_mode
            and any(field in schema for field in SCHEDULE_FIELDS)
            and any(field in schema for field in IDEMPOTENCY_FIELDS)
        )
    if is_cancel_tool(name):
        return policy.allows_transactional_mode and policy.capabilities.cancellation
    if is_status_tool(name):
        return policy.allows_transactional_mode and policy.capabilities.order_status_lookup
    if any(marker in name for marker in OTHER_MUTATION_MARKERS):
        return False
    if any(marker in name for marker in READ_ONLY_MARKERS):
        return policy.mode in {ProviderMode.DISCOVERY, ProviderMode.HANDOFF, ProviderMode.TRANSACTIONAL}
    return False


def public_policy(policy: ProviderPolicy) -> dict[str, Any]:
    ordering_available = bool(
        settings.provider_transactions_enabled(policy.provider) and policy.allows_transactional_mode
    )
    return {
        "provider": policy.provider,
        "version": policy.version,
        "state": policy.state.value,
        "mode": policy.mode.value,
        "capabilities": policy.capabilities.as_dict(),
        "tool_schema_hash": policy.tool_schema_hash,
        "adapter_version": policy.adapter_version,
        "ordering_available": ordering_available,
        "release_gate": "OPEN" if ordering_available else "SAFE_MODE",
    }
