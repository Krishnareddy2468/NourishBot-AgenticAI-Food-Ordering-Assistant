import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes import deliveries
from app.api.routes.operations import router as operations_router
from app.api.main import rate_limit
from app.core import redis_client
from app.core.config import settings
from app.core.logging_config import JsonFormatter
from app.db.models import Order
from app.domain.scheduling import normalize_scheduled_time
from app.operations.release_gate import REQUIRED_AUTOMATED_CHECKS, evaluate_release_evidence
from app.providers.base import CapabilityState, ProviderCapabilities, ProviderMode, ProviderPolicy
from app.providers.registry import public_policy


NOW = datetime(2026, 10, 17, 8, 0, tzinfo=timezone.utc)


def complete_evidence():
    recent = (NOW - timedelta(days=1)).isoformat()
    return {
        "environment": "production",
        "candidate_git_sha": "a" * 40,
        "open_p0_findings": 0,
        "open_p1_findings": 0,
        "automated_validation": {name: True for name in REQUIRED_AUTOMATED_CHECKS},
        "migration_rehearsal": {"upgrade_passed": True, "downgrade_passed": True, "verified_at": recent},
        "backup_restore": {"restore_passed": True, "row_count_verified": True, "checksum_verified": True, "verified_at": recent},
        "monitoring": {"active": True, "test_alert_received": True, "verified_at": recent},
        "deployment_smoke": {"passed": True, "rollback_rehearsed": True, "verified_at": recent},
        "provider_contracts": {
            "SWIGGY": {"mode": "DISCOVERY", "mutations_disabled": True},
            "ZOMATO": {"mode": "HANDOFF", "mutations_disabled": True},
        },
        "public_tracking_security": {"passed": True},
    }


def test_release_gate_passes_only_complete_fresh_production_evidence():
    result = evaluate_release_evidence(complete_evidence(), now=NOW)
    assert result["release_gate_passed"] is True
    assert result["decision"] == "ALLOW_SAFE_MODE_RELEASE"
    assert result["transactional_providers"] == []

    stale = complete_evidence()
    stale["backup_restore"]["verified_at"] = (NOW - timedelta(days=31)).isoformat()
    result = evaluate_release_evidence(stale, now=NOW)
    assert result["release_gate_passed"] is False
    assert "backup_restore" in result["failed_checks"]


def test_release_gate_requires_live_proof_for_transactional_providers():
    evidence = complete_evidence()
    evidence["provider_contracts"]["SWIGGY"] = {
        "mode": "TRANSACTIONAL",
        "mutations_disabled": False,
        "live_sandbox_passed": False,
        "tool_schema_hash": "schema",
        "verified_at": (NOW - timedelta(days=1)).isoformat(),
    }
    assert "provider_contracts" in evaluate_release_evidence(evidence, now=NOW)["failed_checks"]


def test_release_gate_rejects_placeholder_candidate_revision():
    evidence = complete_evidence()
    evidence["candidate_git_sha"] = "replace-with-full-sha"
    assert "candidate_git_sha" in evaluate_release_evidence(evidence, now=NOW)["failed_checks"]


def test_release_gate_accepts_only_owned_unexpired_risk_acceptance():
    evidence = complete_evidence()
    evidence["open_p1_findings"] = 1
    evidence["risk_acceptances"] = [{
        "finding_id": "P1-17", "owner": "CTO", "rationale": "Bounded exposure",
        "expires_at": (NOW + timedelta(days=7)).isoformat(),
    }]
    assert evaluate_release_evidence(evidence, now=NOW)["release_gate_passed"] is True
    evidence["risk_acceptances"][0]["expires_at"] = (NOW - timedelta(seconds=1)).isoformat()
    assert "p0_p1_findings" in evaluate_release_evidence(evidence, now=NOW)["failed_checks"]


def test_external_transactions_need_both_operator_switches(monkeypatch, tmp_path):
    evidence = complete_evidence()
    evidence["provider_contracts"]["ZOMATO"] = {
        "mode": "TRANSACTIONAL",
        "mutations_disabled": False,
        "live_sandbox_passed": True,
        "tool_schema_hash": "zomato-schema-v1",
        "verified_at": (NOW - timedelta(days=1)).isoformat(),
    }
    evidence_path = tmp_path / "release-evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "MONITORING_ALERTS_ENABLED", True)
    monkeypatch.setattr(settings, "MCP_TRANSACTIONAL_ENABLED", True)
    monkeypatch.setattr(settings, "RELEASE_EVIDENCE_FILE", str(evidence_path))
    monkeypatch.setattr(settings, "PUBLIC_TRANSACTIONAL_RELEASE", False)
    assert settings.external_transactions_enabled is False
    monkeypatch.setattr(settings, "PUBLIC_TRANSACTIONAL_RELEASE", True)
    assert settings.external_transactions_enabled is True
    assert settings.provider_transactions_enabled("ZOMATO") is True
    assert settings.provider_transactions_enabled("SWIGGY") is False


def test_public_provider_policy_reports_effective_safe_mode(monkeypatch):
    monkeypatch.setattr(settings, "MCP_TRANSACTIONAL_ENABLED", True)
    monkeypatch.setattr(settings, "PUBLIC_TRANSACTIONAL_RELEASE", False)
    policy = ProviderPolicy(
        provider="ZOMATO", version_id=1, version=1,
        state=CapabilityState.VERIFIED, mode=ProviderMode.TRANSACTIONAL,
        capabilities=ProviderCapabilities(
            scheduled_ordering=True, order_status_lookup=True, stable_idempotency=True,
        ),
        tool_schema_hash="schema",
    )
    payload = public_policy(policy)
    assert payload["ordering_available"] is False
    assert payload["release_gate"] == "SAFE_MODE"


def test_timezone_normalization_rejects_naive_missing_invalid_and_dst_edges():
    naive = datetime(2026, 10, 17, 13, 0)
    with pytest.raises(ValueError, match="explicit IANA timezone"):
        normalize_scheduled_time(naive)
    with pytest.raises(ValueError, match="Unknown"):
        normalize_scheduled_time(naive, "Mars/Olympus")
    with pytest.raises(ValueError, match="ambiguous or does not exist"):
        normalize_scheduled_time(datetime(2026, 3, 8, 2, 30), "America/New_York")
    with pytest.raises(ValueError, match="ambiguous or does not exist"):
        normalize_scheduled_time(datetime(2026, 11, 1, 1, 30), "America/New_York")
    assert normalize_scheduled_time(naive, "Asia/Kolkata").isoformat() == "2026-10-17T07:30:00+00:00"


@pytest.mark.asyncio
async def test_production_duplicate_claim_fails_closed_without_redis(monkeypatch):
    async def unavailable():
        return None

    monkeypatch.setattr(redis_client, "get_redis", unavailable)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    assert await redis_client.claim_once("payment:duplicate") is False
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    assert await redis_client.claim_once("payment:duplicate") is True


@pytest.mark.asyncio
async def test_development_rate_limit_does_not_retry_a_failed_request(monkeypatch):
    async def unavailable():
        return None

    calls = 0

    async def failing_app(_request):
        nonlocal calls
        calls += 1
        raise RuntimeError("application failure")

    from starlette.requests import Request

    request = Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/menu",
        "raw_path": b"/api/v1/menu",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    })
    monkeypatch.setattr(redis_client, "get_redis", unavailable)
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    with pytest.raises(RuntimeError, match="application failure"):
        await rate_limit(request, failing_app)
    assert calls == 1


def test_structured_logs_include_operational_fields_without_arguments():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "done", (), None)
    record.request_id = "request-7"
    record.status_code = 200
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "request-7"
    assert payload["status_code"] == 200
    assert "args" not in payload


def test_tracking_tokens_are_high_entropy_and_not_order_references():
    generator = Order.__table__.c.tracking_token.default.arg
    first, second = generator(None), generator(None)
    assert first != second
    assert len(first) >= 40 and len(second) >= 40


@pytest.mark.asyncio
async def test_public_tracking_rejects_short_tokens_before_database_access(monkeypatch):
    def fail_if_opened():
        raise AssertionError("short tracking tokens must not query the database")

    monkeypatch.setattr(deliveries, "AsyncSessionLocal", fail_if_opened)
    with pytest.raises(HTTPException) as exc:
        await deliveries.track_delivery_public("x" * 31)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_public_tracking_returns_same_not_found_for_unknown_long_tokens(monkeypatch):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement):
            return SimpleNamespace(scalar_one_or_none=lambda: None)

    monkeypatch.setattr(deliveries, "AsyncSessionLocal", Session)
    with pytest.raises(HTTPException) as exc:
        await deliveries.track_delivery_public("x" * 43)
    assert exc.value.status_code == 404
    assert exc.value.detail == "Order not found"


@pytest.mark.asyncio
async def test_public_tracking_response_excludes_customer_and_delivery_details(monkeypatch):
    order = SimpleNamespace(
        id=7,
        order_ref="NB-ABCD1234",
        status="CONFIRMED",
        provider="NOURISHBOT",
        source_channel="WEB",
        order_type="DELIVERY",
        scheduled_for=None,
        schedule_state="LIVE",
        customer_phone="+15555550123",
        delivery_address="private address",
    )
    delivery = SimpleNamespace(
        status="EN_ROUTE",
        driver_id=42,
        picked_up_at=None,
        delivered_at=None,
        customer_phone="+15555550123",
        address_json={"address": "private address"},
    )

    class Session:
        def __init__(self):
            self.results = iter((order, delivery))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def execute(self, _statement):
            value = next(self.results)
            return SimpleNamespace(scalar_one_or_none=lambda: value)

    monkeypatch.setattr(deliveries, "AsyncSessionLocal", Session)
    response = await deliveries.track_delivery_public("x" * 43)
    assert response == {
        "order_ref": "NB-ABCD1234",
        "order_status": "CONFIRMED",
        "provider": "NOURISHBOT",
        "source_channel": "WEB",
        "order_type": "DELIVERY",
        "scheduled_for": None,
        "delivery": {
            "status": "EN_ROUTE",
            "assigned": True,
            "picked_up_at": None,
            "delivered_at": None,
        },
    }
    assert "customer_phone" not in response
    assert "address_json" not in response["delivery"]


def test_phase8_operations_and_release_tooling_are_present():
    assert "/api/v1/operations/readiness" in {route.path for route in operations_router.routes}
    root = Path(__file__).resolve().parents[3]
    assert (root / "scripts/check_release_gate.py").is_file()
    assert (root / "scripts/postgres_backup_restore.py").is_file()
    assert (root / "scripts/smoke_test.py").is_file()
    assert (root / "scripts/load_probe.py").is_file()
    assert (root / "infra/compose/phase8-validation.yml").is_file()


def test_webhooks_require_dedupe_keys_before_side_effects():
    root = Path(__file__).resolve().parents[2] / "app" / "api" / "routes"
    payments = (root / "payments.py").read_text()
    whatsapp = (root / "whatsapp.py").read_text()
    assert "if not await claim_once(delivery_key)" in payments
    assert "await release_claim(delivery_key)" in payments
    assert 'payment.status == "CAPTURED" and new_status != "CAPTURED"' in payments
    assert "message_id and await claim_once" in whatsapp
