"""Deterministic, snapshot-bound confirmation helpers for in-house orders."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import ConfirmationChallenge


def is_explicit_confirmation(message: str) -> bool:
    """Accept only a short, affirmative reply; never match an embedded "yes"."""
    normalized = " ".join((message or "").strip().lower().rstrip(".!?,…").split())
    return normalized in {
        "yes", "y", "yep", "yes confirm", "confirm", "confirmed",
        "place it", "place order", "go ahead", "proceed", "okay place it",
    }


def normalize_schedule(value: str) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Scheduled time must include a timezone.")
    return parsed.astimezone(timezone.utc).isoformat()


def build_snapshot(*, ctx: Any, args: dict, items: list[dict]) -> dict:
    """Build a stable snapshot from only transaction-relevant fields."""
    order_type = str(args.get("order_type") or ctx.order_type or "PICKUP").upper()
    address = str(args.get("address") or ctx.delivery_address or "").strip()
    scheduled_for = normalize_schedule(args.get("scheduled_for") or ctx.scheduled_for or "")
    provider = str(args.get("provider") or ctx.ordering_platform or "NOURISHBOT").upper()
    normalized_items = sorted(
        [
            {
                "item_id": int(item["item_id"]),
                "item_name": str(item.get("item_name") or item.get("item_name_snapshot") or ""),
                "qty": int(item["qty"]),
                "unit_price_cents": int(item["unit_price_cents"]),
            }
            for item in items
        ],
        key=lambda item: (item["item_id"], item["item_name"]),
    )
    return {
        "restaurant_id": int(ctx.restaurant_id),
        "customer_id": int(ctx.customer_id) if ctx.customer_id else None,
        "customer_phone": str(ctx.customer_phone or ""),
        "order_type": order_type,
        "delivery_address": address if order_type == "DELIVERY" else "",
        "scheduled_for": scheduled_for,
        "provider": provider,
        "items": normalized_items,
        "total_cents": sum(item["qty"] * item["unit_price_cents"] for item in normalized_items),
    }


def snapshot_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def create_challenge(db: AsyncSession, *, ctx: Any, snapshot: dict) -> ConfirmationChallenge:
    if not ctx.session_id:
        raise ValueError("A persisted chat session is required before confirmation.")

    digest = snapshot_hash(snapshot)
    now = datetime.now(timezone.utc)
    existing = (await db.execute(
        select(ConfirmationChallenge).where(
            ConfirmationChallenge.session_id == ctx.session_id,
            ConfirmationChallenge.snapshot_hash == digest,
            ConfirmationChallenge.used_at.is_(None),
            ConfirmationChallenge.invalidated_at.is_(None),
            ConfirmationChallenge.expires_at > now,
        ).order_by(ConfirmationChallenge.created_at.desc())
    )).scalars().first()
    if existing:
        return existing

    await db.execute(
        update(ConfirmationChallenge)
        .where(
            ConfirmationChallenge.session_id == ctx.session_id,
            ConfirmationChallenge.used_at.is_(None),
            ConfirmationChallenge.invalidated_at.is_(None),
        )
        .values(invalidated_at=now)
    )
    challenge_id = secrets.token_urlsafe(24)
    challenge = ConfirmationChallenge(
        id=challenge_id,
        restaurant_id=ctx.restaurant_id,
        session_id=ctx.session_id,
        customer_id=ctx.customer_id,
        snapshot_hash=digest,
        snapshot_json=snapshot,
        idempotency_key=f"confirm:{challenge_id}",
        expires_at=now + timedelta(minutes=settings.CONFIRMATION_TTL_MINUTES),
    )
    db.add(challenge)
    await db.flush()
    return challenge


async def lock_valid_challenge(
    db: AsyncSession,
    *,
    challenge_id: str,
    restaurant_id: int,
    session_id: int,
) -> ConfirmationChallenge | None:
    now = datetime.now(timezone.utc)
    return (await db.execute(
        select(ConfirmationChallenge)
        .where(
            ConfirmationChallenge.id == challenge_id,
            ConfirmationChallenge.restaurant_id == restaurant_id,
            ConfirmationChallenge.session_id == session_id,
            ConfirmationChallenge.used_at.is_(None),
            ConfirmationChallenge.invalidated_at.is_(None),
            ConfirmationChallenge.expires_at > now,
        )
        .with_for_update()
    )).scalar_one_or_none()
