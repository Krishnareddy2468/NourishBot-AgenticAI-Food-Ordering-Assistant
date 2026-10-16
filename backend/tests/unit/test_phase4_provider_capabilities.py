from pathlib import Path

import pytest

from app.api.routes.providers import router as providers_router
from app.db.models import ProviderCapabilityVersion, ProviderReceipt
from app.providers.base import CapabilityState, ProviderCapabilities, ProviderMode, ProviderPolicy
from app.providers.mcp_adapter import MCPProviderAdapter
from app.providers.redaction import content_hash, redact_text, redact_value
from app.providers.registry import (
    policy_from_record,
    tool_allowed,
    tool_inventory,
    validate_policy,
)
from app.providers.service import add_receipt


class Schema:
    def __init__(self, properties):
        self.properties = properties

    def model_json_schema(self):
        return {"type": "object", "properties": self.properties}


class Tool:
    def __init__(self, name, properties=None, response=None):
        self.name = name
        self.description = f"Tool {name}"
        self.args_schema = Schema(properties or {})
        self.response = response or {}
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.response


def policy(*, mode="DISCOVERY", state="UNVERIFIED", capabilities=None, schema_hash="hash"):
    return ProviderPolicy(
        provider="ZOMATO",
        version_id=7,
        version=3,
        state=CapabilityState(state),
        mode=ProviderMode(mode),
        capabilities=capabilities or ProviderCapabilities(discovery=True),
        tool_schema_hash=schema_hash,
    )


def test_phase4_models_are_registered():
    assert ProviderCapabilityVersion.__tablename__ == "provider_capability_versions"
    assert ProviderReceipt.__tablename__ == "provider_receipts"


def test_missing_registry_record_defaults_to_unverified_discovery():
    value = policy_from_record(None, "Swiggy")
    assert value.state == CapabilityState.UNVERIFIED
    assert value.mode == ProviderMode.DISCOVERY
    assert value.allows_transactional_mode is False


def test_transactional_mode_requires_verified_scheduling_status_and_idempotency():
    incomplete = ProviderCapabilities(scheduled_ordering=True)
    with pytest.raises(ValueError, match="status lookup"):
        validate_policy("VERIFIED", "TRANSACTIONAL", incomplete, "schema-hash")
    with pytest.raises(ValueError, match="VERIFIED"):
        validate_policy("UNVERIFIED", "TRANSACTIONAL", incomplete, "schema-hash")

    complete = ProviderCapabilities(
        scheduled_ordering=True,
        order_status_lookup=True,
        stable_idempotency=True,
    )
    validate_policy("VERIFIED", "TRANSACTIONAL", complete, "schema-hash")


def test_discovery_mode_never_exposes_transaction_or_other_mutation_tools():
    discovery = policy()
    assert tool_allowed(discovery, Tool("search_restaurants", {"query": {}}), transaction_turn=False)
    assert not tool_allowed(
        discovery,
        Tool("place_order", {"scheduled_for": {}, "idempotency_key": {}}),
        transaction_turn=True,
    )
    assert not tool_allowed(discovery, Tool("delete_saved_address", {"address_id": {}}), transaction_turn=False)


def test_mode_c_requires_schedule_and_idempotency_fields_on_the_actual_tool():
    capabilities = ProviderCapabilities(
        scheduled_ordering=True,
        order_status_lookup=True,
        stable_idempotency=True,
    )
    transactional = policy(mode="TRANSACTIONAL", state="VERIFIED", capabilities=capabilities)
    assert tool_allowed(
        transactional,
        Tool("place_order", {"scheduled_for": {}, "idempotency_key": {}}),
        transaction_turn=True,
    )
    assert not tool_allowed(
        transactional,
        Tool("place_order", {"scheduled_for": {}}),
        transaction_turn=True,
    )
    assert not tool_allowed(
        transactional,
        Tool("place_order", {"idempotency_key": {}}),
        transaction_turn=True,
    )


def test_schema_inventory_hash_is_stable_and_versionable():
    first = tool_inventory([Tool("search_restaurants", {"query": {}})])
    second = tool_inventory([Tool("search_restaurants", {"query": {}})])
    changed = tool_inventory([Tool("search_restaurants", {"query": {}, "area": {}})])
    assert content_hash(first) == content_hash(second)
    assert content_hash(first) != content_hash(changed)


def test_provider_redaction_removes_nested_credentials_contact_and_payment_data():
    value = redact_value({
        "token": "secret-token",
        "customer": {"phone_number": "+91 98765 43210", "address": "1 Main Street"},
        "payment": {"card": "4111111111111111"},
        "safe": "visible",
    })
    assert value["token"] == "[REDACTED]"
    assert value["customer"]["phone_number"] == "[REDACTED]"
    assert value["customer"]["address"] == "[REDACTED]"
    assert value["payment"] == "[REDACTED]"
    assert value["safe"] == "visible"
    assert "98765" not in redact_text("phone=+91 98765 43210")
    assert "Main Street" not in redact_text('{"address": "1 Main Street"}')


@pytest.mark.asyncio
async def test_mode_c_without_provider_order_id_returns_unknown_not_success():
    capabilities = ProviderCapabilities(
        scheduled_ordering=True,
        order_status_lookup=True,
        stable_idempotency=True,
    )
    transaction_tool = Tool(
        "place_order",
        {"scheduled_for": {}, "idempotency_key": {}},
        {"status": "accepted_without_receipt"},
    )
    adapter = MCPProviderAdapter(
        "Zomato",
        [transaction_tool],
        policy(mode="TRANSACTIONAL", state="VERIFIED", capabilities=capabilities),
    )
    result = await adapter.submit_scheduled_order({
        "scheduled_for": "2026-10-20T13:00:00+05:30",
        "idempotency_key": "stable-key",
    })
    assert result.outcome == "UNKNOWN"
    assert result.provider_order_id is None


@pytest.mark.asyncio
async def test_tracking_adapter_uses_verified_status_tool():
    capabilities = ProviderCapabilities(
        scheduled_ordering=True,
        order_status_lookup=True,
        stable_idempotency=True,
    )
    status_tool = Tool("get_order_status", {"order_id": {}}, {"order_id": "P-7", "status": "PREPARING"})
    adapter = MCPProviderAdapter(
        "Zomato",
        [status_tool],
        policy(mode="TRANSACTIONAL", state="VERIFIED", capabilities=capabilities),
    )
    result = await adapter.get_order_status({"provider_order_id": "P-7", "order_id": "P-7"})
    assert result.outcome == "CONFIRMED"
    assert status_tool.calls == [{"order_id": "P-7"}]


def test_receipts_persist_only_redacted_payloads():
    class Session:
        def __init__(self):
            self.added = []

        def add(self, item):
            self.added.append(item)

    session = Session()
    receipt = add_receipt(
        session,
        provider="Zomato",
        mode="DISCOVERY",
        operation="SEARCH",
        request={"address": "1 Main Street", "query": "dinner"},
        response={"phone": "+91 98765 43210", "restaurant": "Example"},
        outcome="RETURNED",
        restaurant_id=1,
    )
    assert receipt.request_redacted_json["address"] == "[REDACTED]"
    assert receipt.response_redacted_json["phone"] == "[REDACTED]"
    assert receipt.response_redacted_json["restaurant"] == "Example"
    assert session.added == [receipt]


def test_provider_routes_expose_registry_receipts_reconciliation_tracking_and_cancel():
    paths = {route.path for route in providers_router.routes}
    assert {
        "/api/v1/providers/capabilities",
        "/api/v1/providers/{provider}/capabilities",
        "/api/v1/providers/attempts",
        "/api/v1/providers/attempts/{attempt_id}/receipts",
        "/api/v1/providers/attempts/{attempt_id}/reconcile",
        "/api/v1/providers/attempts/{attempt_id}/tracking",
        "/api/v1/providers/attempts/{attempt_id}/cancel",
    }.issubset(paths)


def test_phase4_migration_is_chained_seeded_and_append_only():
    migration = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0013_phase4_provider_capabilities.py"
    ).read_text()
    assert 'down_revision: Union[str, Sequence[str], None] = "0012"' in migration
    assert "UNVERIFIED" in migration and "VERIFIED" in migration
    assert "DISCOVERY" in migration and "HANDOFF" in migration and "TRANSACTIONAL" in migration
    assert "trg_provider_capability_versions_append_only" in migration
    assert "trg_provider_receipts_append_only" in migration
    assert "RECONCILE_BEFORE_RETRY" in migration
