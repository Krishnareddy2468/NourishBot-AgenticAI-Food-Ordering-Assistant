"""Provider capability registry and guarded operational actions."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from app.auth.deps import require_admin, require_manager, require_staff, tenant_id
from app.db.models import ProviderAttempt, ProviderCapabilityVersion, ProviderReceipt
from app.db.session import AsyncSessionLocal
from app.providers.base import ProviderCapabilities
from app.providers.redaction import redact_value
from app.providers.registry import current_policy, public_policy, validate_policy
from app.providers.service import cancel_attempt, reconcile_attempt, track_attempt

router = APIRouter(prefix="/api/v1/providers", tags=["providers"])
SUPPORTED_PROVIDERS = {"NOURISHBOT", "SWIGGY", "ZOMATO"}


class CapabilityUpdate(BaseModel):
    state: str
    mode: str
    capabilities: dict[str, Any]
    verification_notes: str | None = None


def _provider_name(value: str) -> str:
    provider = value.strip().upper()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(422, "Unsupported provider")
    return provider


def _attempt_payload(attempt: ProviderAttempt) -> dict[str, Any]:
    return {
        "id": attempt.id,
        "provider": attempt.provider,
        "mode": attempt.mode,
        "operation": attempt.operation,
        "status": attempt.status,
        "attempt_count": attempt.attempt_count,
        "provider_order_id": attempt.provider_order_id,
        "reconciliation_required": attempt.reconciliation_required,
        "reconciliation_count": attempt.reconciliation_count,
        "next_action": attempt.next_action,
        "last_error": attempt.last_error,
        "submitted_at": attempt.submitted_at.isoformat() if attempt.submitted_at else None,
        "completed_at": attempt.completed_at.isoformat() if attempt.completed_at else None,
    }


@router.get("/capabilities")
async def list_capabilities(user=Depends(require_staff)):
    async with AsyncSessionLocal() as db:
        providers = (await db.execute(
            select(ProviderCapabilityVersion.provider).distinct().order_by(ProviderCapabilityVersion.provider)
        )).scalars().all()
        return [public_policy(await current_policy(db, provider)) for provider in providers]


@router.post("/{provider}/capabilities", status_code=201)
async def create_capability_version(
    provider: str,
    body: CapabilityUpdate,
    user=Depends(require_admin),
):
    provider = _provider_name(provider)
    capabilities = ProviderCapabilities.from_dict(body.capabilities)
    async with AsyncSessionLocal() as db:
        current = await current_policy(db, provider)
        try:
            validate_policy(body.state.upper(), body.mode.upper(), capabilities, current.tool_schema_hash)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        version = (await db.execute(
            select(func.coalesce(func.max(ProviderCapabilityVersion.version), 0) + 1)
            .where(ProviderCapabilityVersion.provider == provider)
        )).scalar_one()
        record = ProviderCapabilityVersion(
            provider=provider,
            version=version,
            state=body.state.upper(),
            mode=body.mode.upper(),
            capabilities_json=capabilities.as_dict(),
            tool_schema_hash=current.tool_schema_hash,
            tool_schema_json=(
                (await db.execute(
                    select(ProviderCapabilityVersion.tool_schema_json)
                    .where(ProviderCapabilityVersion.id == current.version_id)
                )).scalar_one_or_none()
                if current.version_id
                else {}
            ) or {},
            adapter_version=current.adapter_version,
            verified_at=datetime.now(timezone.utc) if body.state.upper() == "VERIFIED" else None,
            verified_by_user_id=user.get("user_id"),
            verification_notes=(body.verification_notes or "")[:1000] or None,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return public_policy(await current_policy(db, provider))


@router.get("/attempts")
async def list_provider_attempts(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    user=Depends(require_manager),
):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as db:
        query = select(ProviderAttempt).where(ProviderAttempt.restaurant_id == rid)
        if status:
            query = query.where(ProviderAttempt.status == status.strip().upper())
        attempts = (await db.execute(
            query.order_by(ProviderAttempt.created_at.desc()).limit(limit)
        )).scalars().all()
        return [_attempt_payload(attempt) for attempt in attempts]


@router.get("/attempts/{attempt_id}/receipts")
async def provider_attempt_receipts(attempt_id: str, user=Depends(require_manager)):
    rid = tenant_id(user)
    async with AsyncSessionLocal() as db:
        attempt = (await db.execute(
            select(ProviderAttempt.id).where(
                ProviderAttempt.id == attempt_id,
                ProviderAttempt.restaurant_id == rid,
            )
        )).scalar_one_or_none()
        if attempt is None:
            raise HTTPException(404, "Provider attempt not found")
        receipts = (await db.execute(
            select(ProviderReceipt)
            .where(
                ProviderReceipt.provider_attempt_id == attempt_id,
                ProviderReceipt.restaurant_id == rid,
            )
            .order_by(ProviderReceipt.id)
        )).scalars().all()
        return [
            {
                "id": receipt.id,
                "provider": receipt.provider,
                "mode": receipt.mode,
                "operation": receipt.operation,
                "tool_name": receipt.tool_name,
                "response": receipt.response_redacted_json,
                "response_hash": receipt.response_hash,
                "outcome": receipt.outcome,
                "provider_order_id": receipt.provider_order_id,
                "occurred_at": receipt.occurred_at.isoformat(),
            }
            for receipt in receipts
        ]


async def _tenant_attempt(attempt_id: str, rid: int) -> None:
    async with AsyncSessionLocal() as db:
        exists = (await db.execute(
            select(ProviderAttempt.id).where(
                ProviderAttempt.id == attempt_id,
                ProviderAttempt.restaurant_id == rid,
            )
        )).scalar_one_or_none()
    if exists is None:
        raise HTTPException(404, "Provider attempt not found")


@router.post("/attempts/{attempt_id}/reconcile")
async def reconcile_provider_attempt(attempt_id: str, user=Depends(require_manager)):
    await _tenant_attempt(attempt_id, tenant_id(user))
    try:
        result = await reconcile_attempt(attempt_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, "Provider reconciliation is temporarily unavailable") from exc
    return {"outcome": result.outcome, "provider_order_id": result.provider_order_id, "receipt": redact_value(result.response)}


@router.get("/attempts/{attempt_id}/tracking")
async def track_provider_attempt(attempt_id: str, user=Depends(require_manager)):
    await _tenant_attempt(attempt_id, tenant_id(user))
    try:
        result = await track_attempt(attempt_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, "Provider tracking is temporarily unavailable") from exc
    return {"outcome": result.outcome, "provider_order_id": result.provider_order_id, "receipt": redact_value(result.response)}


@router.post("/attempts/{attempt_id}/cancel")
async def cancel_provider_attempt(attempt_id: str, user=Depends(require_manager)):
    await _tenant_attempt(attempt_id, tenant_id(user))
    try:
        result = await cancel_attempt(attempt_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, "Provider cancellation is temporarily unavailable") from exc
    return {"outcome": result.outcome, "provider_order_id": result.provider_order_id, "receipt": redact_value(result.response)}
