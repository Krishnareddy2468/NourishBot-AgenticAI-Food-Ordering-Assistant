"""Customer web identity linking and privacy self-service endpoints."""

from __future__ import annotations

import hashlib
import secrets

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agent.privacy_service import handle_customer_account_command
from app.core.config import settings
from app.db.models import Channel
from app.db.session import AsyncSessionLocal

router = APIRouter(prefix="/api/v1/customer", tags=["customer-privacy"])


class LinkCodeRequest(BaseModel):
    code: str = Field(min_length=16, max_length=128)


class ConsentRequest(BaseModel):
    purpose: str = Field(min_length=1, max_length=32)
    granted: bool


class DeleteMemoryRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=64)


def _browser_identity(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _set_browser_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.CUSTOMER_IDENTITY_COOKIE_NAME,
        value=token,
        max_age=365 * 24 * 60 * 60,
        httponly=True,
        secure=settings.is_production,
        samesite=settings.AUTH_COOKIE_SAMESITE,
        path="/",
    )


def _prevent_sensitive_caching(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


async def _linked_browser_id(request: Request) -> str:
    raw = request.cookies.get(settings.CUSTOMER_IDENTITY_COOKIE_NAME)
    if not raw:
        raise HTTPException(401, "No linked customer web identity")
    external_id = _browser_identity(raw)
    async with AsyncSessionLocal() as db:
        channel = (await db.execute(select(Channel).where(
            Channel.restaurant_id == settings.DEFAULT_RESTAURANT_ID,
            Channel.type == "WEB",
            Channel.external_id == external_id,
            Channel.customer_id.is_not(None),
        ))).scalar_one_or_none()
    if not channel:
        raise HTTPException(401, "Customer web identity is not linked")
    return external_id


async def linked_customer_channel(request: Request) -> Channel:
    """Resolve the tenant/customer owned by the opaque linked browser cookie."""
    raw = request.cookies.get(settings.CUSTOMER_IDENTITY_COOKIE_NAME)
    if not raw:
        raise HTTPException(401, "No linked customer web identity")
    external_id = _browser_identity(raw)
    async with AsyncSessionLocal() as db:
        channel = (await db.execute(select(Channel).where(
            Channel.restaurant_id == settings.DEFAULT_RESTAURANT_ID,
            Channel.type == "WEB",
            Channel.external_id == external_id,
            Channel.customer_id.is_not(None),
        ))).scalar_one_or_none()
    if not channel:
        raise HTTPException(401, "Customer web identity is not linked")
    return channel


@router.post("/identity/link")
async def link_web_identity(body: LinkCodeRequest, request: Request, response: Response):
    _prevent_sensitive_caching(response)
    token = request.cookies.get(settings.CUSTOMER_IDENTITY_COOKIE_NAME) or secrets.token_urlsafe(32)
    external_id = _browser_identity(token)
    message = await handle_customer_account_command(
        f"link {body.code.strip()}",
        chat_id=external_id,
        channel_type="WEB",
    )
    if not message or not message.startswith("Your channel is now securely linked"):
        raise HTTPException(400, message or "Unable to link customer identity")
    _set_browser_cookie(response, token)
    return {"linked": True, "message": message}


@router.get("/privacy")
async def privacy_status(request: Request, response: Response):
    _prevent_sensitive_caching(response)
    external_id = await _linked_browser_id(request)
    message = await handle_customer_account_command(
        "privacy status", chat_id=external_id, channel_type="WEB"
    )
    return {"message": message}


@router.post("/privacy/consent")
async def update_consent(body: ConsentRequest, request: Request, response: Response):
    _prevent_sensitive_caching(response)
    external_id = await _linked_browser_id(request)
    purpose = body.purpose.strip().upper()
    commands = {
        ("PERSONALIZATION", True): "enable memory",
        ("PERSONALIZATION", False): "disable memory",
        ("LOCATION", True): "enable location",
        ("LOCATION", False): "disable location",
        ("REMINDERS", True): "enable reminders",
        ("REMINDERS", False): "disable reminders",
        ("ANALYTICS", True): "enable analytics",
        ("ANALYTICS", False): "disable analytics",
    }
    command = commands.get((purpose, body.granted))
    if not command:
        raise HTTPException(400, "Unsupported consent purpose")
    message = await handle_customer_account_command(command, chat_id=external_id, channel_type="WEB")
    return {"purpose": purpose, "granted": body.granted, "message": message}


@router.get("/privacy/export")
async def export_memory(request: Request, response: Response):
    _prevent_sensitive_caching(response)
    external_id = await _linked_browser_id(request)
    message = await handle_customer_account_command(
        "export my memory", chat_id=external_id, channel_type="WEB"
    )
    return {"message": message}


@router.delete("/privacy/memory")
async def delete_memory(body: DeleteMemoryRequest, request: Request, response: Response):
    _prevent_sensitive_caching(response)
    if body.confirmation.strip().lower() != "delete my memory permanently":
        raise HTTPException(400, "Exact deletion confirmation is required")
    external_id = await _linked_browser_id(request)
    message = await handle_customer_account_command(
        body.confirmation, chat_id=external_id, channel_type="WEB"
    )
    return {"deleted": True, "message": message}
