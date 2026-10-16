"""Fail-closed evaluation of signed-off production release evidence."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REQUIRED_AUTOMATED_CHECKS = {
    "confirmation_and_staleness",
    "duplicate_suppression",
    "scheduler_recovery_and_concurrency",
    "timezone_and_dst",
    "provider_unknown_reconciliation",
    "cancellation_and_rescheduling",
    "menu_revalidation",
    "fulfillment_eligibility",
    "tenant_and_authorization",
    "privacy_lifecycle",
    "budgets_and_routines",
    "postgres_and_redis_integration",
    "frontend_accessibility_and_e2e",
}


def _fresh(value: Any, now: datetime, *, days: int = 90) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return False
        parsed = parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False
    return now - timedelta(days=days) <= parsed <= now + timedelta(minutes=5)


def _not_expired(value: Any, now: datetime) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.astimezone(timezone.utc) > now
    except (TypeError, ValueError):
        return False


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_release_evidence(
    evidence: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate evidence without trusting a self-declared overall status."""
    now = now or datetime.now(timezone.utc)
    checks: list[dict[str, Any]] = []

    checks.append(_check(
        "production_environment",
        evidence.get("environment") == "production",
        "Evidence must come from the production environment.",
    ))
    candidate_sha = str(evidence.get("candidate_git_sha") or "").lower()
    checks.append(_check(
        "candidate_git_sha",
        bool(re.fullmatch(r"[0-9a-f]{40}", candidate_sha)),
        "Evidence must identify the exact candidate with a full Git SHA.",
    ))
    unresolved_findings = int(evidence.get("open_p0_findings") or 0) + int(evidence.get("open_p1_findings") or 0)
    acceptances = evidence.get("risk_acceptances") or []
    valid_acceptances = [
        item for item in acceptances
        if isinstance(item, dict)
        and item.get("finding_id")
        and item.get("owner")
        and item.get("rationale")
        and _not_expired(item.get("expires_at"), now)
    ]
    checks.append(_check(
        "p0_p1_findings",
        unresolved_findings == 0 or len(valid_acceptances) >= unresolved_findings,
        "P0 and P1 findings must be zero or represented by explicit, unexpired risk acceptances.",
    ))

    automated = evidence.get("automated_validation") or {}
    missing_automated = sorted(
        name for name in REQUIRED_AUTOMATED_CHECKS if automated.get(name) is not True
    )
    checks.append(_check(
        "automated_validation",
        not missing_automated,
        "Complete" if not missing_automated else f"Missing: {', '.join(missing_automated)}",
    ))

    migration = evidence.get("migration_rehearsal") or {}
    checks.append(_check(
        "migration_forward_and_rollback",
        migration.get("upgrade_passed") is True
        and migration.get("downgrade_passed") is True
        and _fresh(migration.get("verified_at"), now),
        "Forward and rollback rehearsal must have passed within 90 days.",
    ))

    backup = evidence.get("backup_restore") or {}
    checks.append(_check(
        "backup_restore",
        backup.get("restore_passed") is True
        and backup.get("row_count_verified") is True
        and backup.get("checksum_verified") is True
        and _fresh(backup.get("verified_at"), now, days=30),
        "A checksum and row-count verified restore drill must have passed within 30 days.",
    ))

    monitoring = evidence.get("monitoring") or {}
    checks.append(_check(
        "monitoring_and_alerts",
        monitoring.get("active") is True
        and monitoring.get("test_alert_received") is True
        and _fresh(monitoring.get("verified_at"), now, days=30),
        "Production monitoring and a test alert must be verified within 30 days.",
    ))

    smoke = evidence.get("deployment_smoke") or {}
    checks.append(_check(
        "deployment_smoke_and_rollback",
        smoke.get("passed") is True
        and smoke.get("rollback_rehearsed") is True
        and _fresh(smoke.get("verified_at"), now, days=30),
        "Production smoke and rollback rehearsal must pass within 30 days.",
    ))

    provider_records = evidence.get("provider_contracts") or {}
    provider_failures = []
    for provider in ("SWIGGY", "ZOMATO"):
        record = provider_records.get(provider) or {}
        mode = str(record.get("mode") or "").upper()
        safe_mode = mode in {"DISCOVERY", "HANDOFF"} and record.get("mutations_disabled") is True
        verified_mode = (
            mode == "TRANSACTIONAL"
            and record.get("live_sandbox_passed") is True
            and bool(record.get("tool_schema_hash"))
            and _fresh(record.get("verified_at"), now, days=30)
        )
        if not (safe_mode or verified_mode):
            provider_failures.append(provider)
    checks.append(_check(
        "provider_contracts",
        not provider_failures,
        "Complete" if not provider_failures else f"Unsafe or unverified: {', '.join(provider_failures)}",
    ))

    public_tracking = evidence.get("public_tracking_security") or {}
    checks.append(_check(
        "public_tracking_security",
        public_tracking.get("passed") is True,
        "Public tracking authorization and token-entropy checks must pass.",
    ))

    passed = all(item["passed"] for item in checks)
    transactional_providers = sorted(
        provider
        for provider in ("SWIGGY", "ZOMATO")
        for record in [provider_records.get(provider) or {}]
        if str((record or {}).get("mode") or "").upper() == "TRANSACTIONAL"
    )
    if not passed:
        decision = "KEEP_SAFE_MODE"
    elif transactional_providers:
        decision = "ALLOW_VERIFIED_TRANSACTIONAL_RELEASE"
    else:
        decision = "ALLOW_SAFE_MODE_RELEASE"
    return {
        "release_gate_passed": passed,
        "decision": decision,
        "transactional_providers": transactional_providers,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "evaluated_at": now.isoformat(),
    }


def load_release_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError("Release evidence file does not exist")
    if source.stat().st_size > 1_000_000:
        raise ValueError("Release evidence file is too large")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Release evidence file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Release evidence must be a JSON object")
    return value
