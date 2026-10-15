"""Verified WhatsApp Cloud API webhook for the NourishBot conversation pipeline."""

from __future__ import annotations

import hashlib
import hmac
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from app.agent.pipeline import process_message
from app.core.config import settings
from app.core.redis_client import claim_once
from app.db.crud import get_session, save_session
from app.agent.privacy_service import channel_has_consent, handle_customer_account_command

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/channels/whatsapp", tags=["whatsapp"])


def _valid_signature(body: bytes, signature: str) -> bool:
    if not settings.WHATSAPP_APP_SECRET or not signature.startswith("sha256="):
        return False
    expected = hmac.new(
        settings.WHATSAPP_APP_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature.removeprefix("sha256="), expected)


def _message_text(message: dict) -> str:
    message_type = message.get("type")
    if message_type == "text":
        return str((message.get("text") or {}).get("body") or "").strip()
    if message_type == "interactive":
        interactive = message.get("interactive") or {}
        reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
        return str(reply.get("title") or reply.get("id") or "").strip()
    if message_type == "location":
        location = message.get("location") or {}
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if latitude is not None and longitude is not None:
            return (
                f"My current location is {latitude},{longitude}. Use it as my route origin. "
                "Ask for my destination to find drive-through restaurants on my route."
            )
    return ""


async def _send_text(to: str, text: str) -> None:
    if not all([
        settings.WHATSAPP_GRAPH_API_VERSION,
        settings.WHATSAPP_PHONE_NUMBER_ID,
        settings.WHATSAPP_ACCESS_TOKEN,
    ]):
        logger.error("WhatsApp reply skipped because Cloud API credentials are incomplete")
        return

    url = (
        f"https://graph.facebook.com/{settings.WHATSAPP_GRAPH_API_VERSION}/"
        f"{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text[:4096]},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()


async def _process_message(wa_id: str, user_name: str, text: str) -> None:
    try:
        account_reply = await handle_customer_account_command(
            text,
            chat_id=wa_id,
            channel_type="WHATSAPP",
        )
        if account_reply is not None:
            await _send_text(wa_id, account_reply)
            return
        if text.startswith("My current location is ") and not await channel_has_consent(
            chat_id=wa_id,
            channel_type="WHATSAPP",
            purpose="LOCATION",
        ):
            await _send_text(
                wa_id,
                "Location use is disabled. Send 'enable location' first, then share the location again.",
            )
            return
        providers = ["NourishBot", "Swiggy", "Zomato"]
        normalized = text.strip().casefold()
        selected = next(
            (
                provider
                for provider in providers
                if normalized in {provider.casefold(), f"use {provider.casefold()}"}
            ),
            None,
        )
        if selected:
            await save_session(
                wa_id,
                {"ordering_platform": selected},
                channel_type="WHATSAPP",
            )
            reply = (
                f"{selected} selected. Tell me the meal and the future date, time, "
                "and timezone you want. Instant ordering is disabled."
            )
        else:
            session = await get_session(wa_id, "WHATSAPP")
            platform = session.get("ordering_platform") or "NourishBot"
            if platform == "NourishBot":
                reply = await process_message(
                    message=text,
                    chat_id=wa_id,
                    user_name=user_name,
                    channel_type="WHATSAPP",
                )
            else:
                from app.agent.mcp_agent import get_mcp_response
                reply = await get_mcp_response(
                    text,
                    wa_id,
                    user_name,
                    platform,
                    channel_type="WHATSAPP",
                )
                if reply is None:
                    reply = (
                        f"{platform} scheduled ordering is unavailable. Send 'NourishBot' "
                        "or choose Swiggy or Zomato."
                    )
        await _send_text(wa_id, reply)
    except Exception:
        logger.exception("WhatsApp message processing failed for wa_id=%s", wa_id)
        raise


@router.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(
    mode: str = Query(default="", alias="hub.mode"),
    token: str = Query(default="", alias="hub.verify_token"),
    challenge: str = Query(default="", alias="hub.challenge"),
):
    if not settings.WHATSAPP_ENABLED:
        raise HTTPException(status_code=404, detail="WhatsApp channel is disabled")
    if mode == "subscribe" and hmac.compare_digest(token, settings.WHATSAPP_VERIFY_TOKEN):
        return challenge
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    if not settings.WHATSAPP_ENABLED:
        raise HTTPException(status_code=404, detail="WhatsApp channel is disabled")

    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")
    if not _valid_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    queued = 0
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            contacts = value.get("contacts") or []
            names = {
                str(contact.get("wa_id") or ""): str(
                    ((contact.get("profile") or {}).get("name") or "")
                )
                for contact in contacts
            }
            for message in value.get("messages") or []:
                wa_id = str(message.get("from") or "").strip()
                message_id = str(message.get("id") or "").strip()
                text = _message_text(message)
                if wa_id and text and (
                    not message_id or await claim_once(f"whatsapp:{message_id}")
                ):
                    try:
                        from app.workers.notification_worker import process_whatsapp_message
                        process_whatsapp_message.delay(wa_id, names.get(wa_id, ""), text)
                    except Exception as exc:
                        logger.warning("Celery enqueue failed; using in-process fallback: %s", exc)
                        background_tasks.add_task(
                            _process_message,
                            wa_id,
                            names.get(wa_id, ""),
                            text,
                        )
                    queued += 1
    return {"ok": True, "queued": queued}
