"""Deterministic customer identity, consent, and personalization controls."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import (
    Channel,
    CustomerConsent,
    IdentityLinkChallenge,
    MemoryLearningEvent,
    PrivacyRequest,
    RouteDiscovery,
    SavedLocation,
    SavedRoute,
    UserPreferences,
)
from app.db.session import get_session_factory
from app.core.redis_client import invalidate_session_cache


CONSENT_PURPOSES = ("PERSONALIZATION", "LOCATION", "REMINDERS", "ANALYTICS")
CHANNEL_TYPES = {"TELEGRAM", "WHATSAPP", "WEB"}


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _verification_method(channel_type: str) -> str:
    return {
        "WHATSAPP": "WHATSAPP_SIGNED_WEBHOOK",
        "TELEGRAM": "TELEGRAM_PLATFORM",
        "WEB": "WEB_SESSION",
    }.get(channel_type, "PLATFORM")


async def ensure_channel(
    db: AsyncSession,
    *,
    restaurant_id: int,
    channel_type: str,
    external_id: str | int,
) -> Channel:
    """Return the tenant-scoped platform identity and mark it as platform verified."""
    channel_type = channel_type.upper()
    if channel_type not in CHANNEL_TYPES:
        raise ValueError("Unsupported identity channel")
    channel = (await db.execute(
        select(Channel).where(
            Channel.restaurant_id == restaurant_id,
            Channel.type == channel_type,
            Channel.external_id == str(external_id),
        )
    )).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if not channel:
        channel = Channel(
            restaurant_id=restaurant_id,
            type=channel_type,
            external_id=str(external_id),
            verified_at=now,
            verification_method=_verification_method(channel_type),
            last_seen_at=now,
        )
        db.add(channel)
        await db.flush()
    else:
        channel.last_seen_at = now
        if not channel.verified_at:
            channel.verified_at = now
            channel.verification_method = _verification_method(channel_type)
    return channel


async def has_consent_in_session(
    db: AsyncSession,
    *,
    customer_id: int,
    purpose: str,
    restaurant_id: int | None = None,
) -> bool:
    purpose = purpose.upper()
    if purpose not in CONSENT_PURPOSES:
        return False
    conditions = [
        CustomerConsent.customer_id == customer_id,
        CustomerConsent.purpose == purpose,
    ]
    if restaurant_id is not None:
        conditions.append(CustomerConsent.restaurant_id == restaurant_id)
    decision = (await db.execute(
        select(CustomerConsent)
        .where(*conditions)
        .order_by(CustomerConsent.id.desc())
        .limit(1)
    )).scalar_one_or_none()
    return bool(decision and decision.granted)


async def _latest_consent_decision(
    db: AsyncSession,
    *,
    customer_id: int,
    purpose: str,
    restaurant_id: int,
) -> bool | None:
    decision = (await db.execute(
        select(CustomerConsent.granted)
        .where(
            CustomerConsent.customer_id == customer_id,
            CustomerConsent.restaurant_id == restaurant_id,
            CustomerConsent.purpose == purpose,
        )
        .order_by(CustomerConsent.id.desc())
        .limit(1)
    )).scalar_one_or_none()
    return None if decision is None else bool(decision)


async def has_consent(customer_id: int, purpose: str, restaurant_id: int | None = None) -> bool:
    async with get_session_factory()() as db:
        return await has_consent_in_session(
            db,
            customer_id=customer_id,
            purpose=purpose,
            restaurant_id=restaurant_id,
        )


async def channel_has_consent(
    *,
    chat_id: str | int,
    channel_type: str,
    purpose: str,
    restaurant_id: int | None = None,
) -> bool:
    restaurant_id = restaurant_id or settings.DEFAULT_RESTAURANT_ID
    async with get_session_factory()() as db:
        channel = (await db.execute(select(Channel).where(
            Channel.restaurant_id == restaurant_id,
            Channel.type == channel_type.upper(),
            Channel.external_id == str(chat_id),
        ))).scalar_one_or_none()
        if not channel or not channel.customer_id:
            return False
        return await has_consent_in_session(
            db,
            customer_id=channel.customer_id,
            purpose=purpose,
            restaurant_id=restaurant_id,
        )


async def _set_consent(
    db: AsyncSession,
    *,
    channel: Channel,
    purpose: str,
    granted: bool,
) -> None:
    if not channel.customer_id:
        raise ValueError("A linked customer identity is required")
    purpose = purpose.upper()
    if purpose not in CONSENT_PURPOSES:
        raise ValueError("Unsupported consent purpose")
    current = await _latest_consent_decision(
        db,
        customer_id=channel.customer_id,
        purpose=purpose,
        restaurant_id=channel.restaurant_id,
    )
    if current == granted:
        return
    db.add(CustomerConsent(
        restaurant_id=channel.restaurant_id,
        customer_id=channel.customer_id,
        purpose=purpose,
        granted=granted,
        policy_version=settings.CONSENT_POLICY_VERSION,
        source_channel_id=channel.id,
    ))


async def _current_consents(db: AsyncSession, channel: Channel) -> dict[str, bool]:
    if not channel.customer_id:
        return {purpose: False for purpose in CONSENT_PURPOSES}
    return {
        purpose: await has_consent_in_session(
            db,
            customer_id=channel.customer_id,
            purpose=purpose,
            restaurant_id=channel.restaurant_id,
        )
        for purpose in CONSENT_PURPOSES
    }


def _masked_external_id(value: str) -> str:
    value = str(value or "")
    return f"***{value[-4:]}" if len(value) > 4 else "***"


def _parse_retention_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def prune_session_history(
    history: object,
    cutoff: datetime,
    session_updated_at: datetime | None,
) -> object:
    """Remove expired chat turns while retaining non-message session state."""
    session_is_recent = bool(session_updated_at and session_updated_at >= cutoff)
    if isinstance(history, list):
        return history if session_is_recent else []
    if not isinstance(history, dict):
        return history
    retained = []
    for turn in list(history.get("turns") or []):
        recorded_at = _parse_retention_timestamp(
            turn.get("at") if isinstance(turn, dict) else None
        )
        if (recorded_at and recorded_at >= cutoff) or (
            recorded_at is None and session_is_recent
        ):
            retained.append(turn)
    return {**history, "turns": retained}


def prune_session_location_data(
    history: object,
    cutoff: datetime,
    session_updated_at: datetime | None,
) -> object:
    """Remove expired shared coordinates and transient delivery addresses."""
    session_is_recent = bool(session_updated_at and session_updated_at >= cutoff)
    if isinstance(history, list):
        turns = history
        metadata = None
    elif isinstance(history, dict):
        turns = list(history.get("turns") or [])
        metadata = dict(history.get("meta") or {})
    else:
        return history

    retained = []
    for turn in turns:
        content = str(turn.get("content") or "") if isinstance(turn, dict) else ""
        contains_shared_location = content.startswith("My current location is ")
        recorded_at = _parse_retention_timestamp(
            turn.get("at") if isinstance(turn, dict) else None
        )
        location_is_expired = contains_shared_location and (
            (recorded_at is not None and recorded_at < cutoff)
            or (recorded_at is None and not session_is_recent)
        )
        if not location_is_expired:
            retained.append(turn)

    if metadata is None:
        return retained
    if not session_is_recent:
        metadata.pop("delivery_address", None)
    return {**history, "meta": metadata, "turns": retained}


async def _export_personalization(db: AsyncSession, channel: Channel) -> dict[str, Any]:
    prefs = (await db.execute(
        select(UserPreferences).where(UserPreferences.customer_id == channel.customer_id)
    )).scalar_one_or_none()
    channels = (await db.execute(
        select(Channel).where(
            Channel.restaurant_id == channel.restaurant_id,
            Channel.customer_id == channel.customer_id,
        ).order_by(Channel.id)
    )).scalars().all()
    consents = await _current_consents(db, channel)
    export: dict[str, Any] = {
        "consents": {key.lower(): value for key, value in consents.items()},
        "linked_channels": [
            {
                "type": item.type,
                "external_id": _masked_external_id(item.external_id),
                "verified": bool(item.verified_at),
            }
            for item in channels
        ],
        "preferences": None,
    }
    if prefs:
        export["preferences"] = {
            "spice_level": prefs.spice_level,
            "cuisine_weights": prefs.cuisine_weights or {},
            "price_band": prefs.price_band,
            "order_timing": prefs.order_timing or {},
            "platform_preference": prefs.platform_preference,
            "dietary_flags": prefs.dietary_flags or [],
            "health_goals": prefs.health_goals or [],
            "allergies": prefs.allergies or [],
            "craving_cycles": prefs.craving_cycles or {},
            "total_orders": prefs.total_orders or 0,
            "total_spent_cents": prefs.total_spent_cents or 0,
            "last_order_at": prefs.last_order_at.isoformat() if prefs.last_order_at else None,
        }
    return export


async def _delete_personalization(db: AsyncSession, channel: Channel) -> None:
    await db.execute(delete(MemoryLearningEvent).where(
        MemoryLearningEvent.customer_id == channel.customer_id,
        MemoryLearningEvent.restaurant_id == channel.restaurant_id,
    ))
    await db.execute(delete(UserPreferences).where(UserPreferences.customer_id == channel.customer_id))
    await _set_consent(db, channel=channel, purpose="PERSONALIZATION", granted=False)
    db.add(PrivacyRequest(
        restaurant_id=channel.restaurant_id,
        customer_id=channel.customer_id,
        request_type="DELETE",
        source_channel_id=channel.id,
    ))


async def _delete_saved_location_data(db: AsyncSession, channel: Channel) -> None:
    """Physically remove route/location data when location consent is withdrawn."""
    scope = (
        RouteDiscovery.customer_id == channel.customer_id,
        RouteDiscovery.restaurant_id == channel.restaurant_id,
    )
    await db.execute(delete(RouteDiscovery).where(*scope))
    await db.execute(delete(SavedRoute).where(
        SavedRoute.customer_id == channel.customer_id,
        SavedRoute.restaurant_id == channel.restaurant_id,
    ))
    await db.execute(delete(SavedLocation).where(
        SavedLocation.customer_id == channel.customer_id,
        SavedLocation.restaurant_id == channel.restaurant_id,
    ))


async def _create_link_code(db: AsyncSession, channel: Channel, target_type: str) -> str:
    if not channel.customer_id or not channel.verified_at:
        raise ValueError("Complete your customer profile on this verified channel first")
    target_type = target_type.upper()
    if target_type not in CHANNEL_TYPES or target_type == channel.type:
        raise ValueError("Choose a different channel: Telegram, WhatsApp, or web")
    now = datetime.now(timezone.utc)
    await db.execute(
        update(IdentityLinkChallenge)
        .where(
            IdentityLinkChallenge.customer_id == channel.customer_id,
            IdentityLinkChallenge.used_at.is_(None),
            IdentityLinkChallenge.expires_at > now,
        )
        .values(used_at=now)
    )
    token = secrets.token_urlsafe(18)
    db.add(IdentityLinkChallenge(
        id=secrets.token_urlsafe(24),
        restaurant_id=channel.restaurant_id,
        customer_id=channel.customer_id,
        source_channel_id=channel.id,
        target_channel_type=target_type,
        token_hash=_token_hash(token),
        expires_at=now + timedelta(minutes=settings.IDENTITY_LINK_TTL_MINUTES),
    ))
    return token


async def _redeem_link_code(db: AsyncSession, channel: Channel, token: str) -> str:
    now = datetime.now(timezone.utc)
    challenge = (await db.execute(
        select(IdentityLinkChallenge)
        .where(
            IdentityLinkChallenge.restaurant_id == channel.restaurant_id,
            IdentityLinkChallenge.token_hash == _token_hash(token),
            IdentityLinkChallenge.target_channel_type == channel.type,
            IdentityLinkChallenge.used_at.is_(None),
            IdentityLinkChallenge.expires_at > now,
        )
        .with_for_update()
    )).scalar_one_or_none()
    if not challenge:
        return "That link code is invalid, expired, already used, or intended for another channel."
    if channel.customer_id and channel.customer_id != challenge.customer_id:
        return "This channel is already linked to another customer. Contact support instead of merging accounts automatically."
    channel.customer_id = challenge.customer_id
    channel.linked_at = now
    channel.verified_at = channel.verified_at or now
    channel.verification_method = "ONE_TIME_LINK_CODE"
    challenge.used_at = now
    challenge.target_channel_id = channel.id
    return "Your channel is now securely linked to the same NourishBot customer profile."


def _normalized_command(message: str) -> str:
    return " ".join((message or "").strip().split())


def is_account_command(message: str) -> bool:
    lowered = _normalized_command(message).lower().lstrip("/")
    exact = {
        "privacy", "privacy status", "memory", "memory status", "view memory",
        "enable memory", "allow personalization", "disable memory", "stop personalization",
        "export memory", "export my memory", "export my data", "delete memory",
        "delete my memory permanently", "enable location", "allow location",
        "disable location", "deny location", "enable reminders", "disable reminders",
        "enable analytics", "disable analytics",
    }
    return lowered in exact or lowered.startswith("link ")


async def handle_customer_account_command(
    message: str,
    *,
    chat_id: str | int,
    channel_type: str,
    restaurant_id: int | None = None,
) -> str | None:
    """Handle exact privacy/linking commands without involving an LLM."""
    if not is_account_command(message):
        return None
    restaurant_id = restaurant_id or settings.DEFAULT_RESTAURANT_ID
    raw = _normalized_command(message)
    command = raw.lower().lstrip("/")

    async with get_session_factory()() as db:
        channel = await ensure_channel(
            db,
            restaurant_id=restaurant_id,
            channel_type=channel_type,
            external_id=chat_id,
        )

        if command.startswith("link "):
            argument = raw.split(" ", 1)[1].strip()
            target = argument.upper()
            if target in CHANNEL_TYPES:
                try:
                    token = await _create_link_code(db, channel, target)
                except ValueError as exc:
                    return str(exc)
                await db.commit()
                return (
                    f"Open NourishBot on {target.title()} and send: link {token}\n"
                    f"This one-time code expires in {settings.IDENTITY_LINK_TTL_MINUTES} minutes."
                )
            reply = await _redeem_link_code(db, channel, argument)
            await db.commit()
            if reply.startswith("Your channel is now securely linked"):
                await invalidate_session_cache(chat_id, channel_type.upper())
            return reply

        if not channel.customer_id:
            await db.commit()
            return "Complete your customer name and phone in NourishBot before managing privacy or linked channels."

        if command in {"privacy", "privacy status", "memory", "memory status", "view memory"}:
            decisions = await _current_consents(db, channel)
            db.add(PrivacyRequest(
                restaurant_id=channel.restaurant_id,
                customer_id=channel.customer_id,
                request_type="VIEW",
                source_channel_id=channel.id,
            ))
            await db.commit()
            labels = ", ".join(
                f"{purpose.title()}: {'enabled' if granted else 'disabled'}"
                for purpose, granted in decisions.items()
            )
            return f"Your privacy settings — {labels}. You can say ‘export my memory’, ‘disable memory’, or ‘delete memory’."

        purpose_commands = {
            "enable memory": ("PERSONALIZATION", True),
            "allow personalization": ("PERSONALIZATION", True),
            "disable memory": ("PERSONALIZATION", False),
            "stop personalization": ("PERSONALIZATION", False),
            "enable location": ("LOCATION", True),
            "allow location": ("LOCATION", True),
            "disable location": ("LOCATION", False),
            "deny location": ("LOCATION", False),
            "enable reminders": ("REMINDERS", True),
            "disable reminders": ("REMINDERS", False),
            "enable analytics": ("ANALYTICS", True),
            "disable analytics": ("ANALYTICS", False),
        }
        if command in purpose_commands:
            purpose, granted = purpose_commands[command]
            await _set_consent(db, channel=channel, purpose=purpose, granted=granted)
            if purpose == "LOCATION" and not granted:
                await _delete_saved_location_data(db, channel)
            if purpose == "PERSONALIZATION" and not granted:
                db.add(PrivacyRequest(
                    restaurant_id=channel.restaurant_id,
                    customer_id=channel.customer_id,
                    request_type="DISABLE",
                    source_channel_id=channel.id,
                ))
            await db.commit()
            state = "enabled" if granted else "disabled"
            suffix = " Existing memory is retained until you delete it." if purpose == "PERSONALIZATION" and not granted else ""
            return f"{purpose.title()} consent is now {state}.{suffix}"

        if command in {"export memory", "export my memory", "export my data"}:
            payload = await _export_personalization(db, channel)
            db.add(PrivacyRequest(
                restaurant_id=channel.restaurant_id,
                customer_id=channel.customer_id,
                request_type="EXPORT",
                source_channel_id=channel.id,
            ))
            await db.commit()
            return "Your NourishBot personalization export:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)

        if command == "delete memory":
            await db.commit()
            return "This permanently deletes your stored personalization. Send exactly: delete my memory permanently"

        if command == "delete my memory permanently":
            await _delete_personalization(db, channel)
            await db.commit()
            return "Your stored personalization has been permanently deleted and memory is disabled."

    return None
