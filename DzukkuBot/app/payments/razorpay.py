"""
Razorpay payment integration for Dzukku vNext.

Handles:
  - Creating payment orders (intents)
  - Verifying webhook signatures
  - Capturing payment status updates
"""

import hmac
import hashlib
import logging
import os
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"


def _get_credentials() -> tuple[str, str]:
    """Get Razorpay credentials from settings."""
    return settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET


async def create_payment_order(
    amount_cents: int,
    currency: str = "INR",
    order_ref: str = "",
    receipt: str = "",
) -> dict:
    """
    Create a Razorpay order (payment intent).

    Args:
        amount_cents: Amount in smallest currency unit (paise for INR).
        currency: Currency code (default INR).
        order_ref: Internal order reference for notes.
        receipt: Receipt identifier (max 40 chars).

    Returns:
        Razorpay order object dict.
    """
    key_id, key_secret = _get_credentials()
    if not key_id or not key_secret:
        raise RuntimeError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set")

    payload = {
        "amount": amount_cents,
        "currency": currency,
        "receipt": receipt or order_ref[:40],
        "notes": {"order_ref": order_ref},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RAZORPAY_BASE_URL}/orders",
            json=payload,
            auth=(key_id, key_secret),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


def verify_webhook_signature(
    payload_body: bytes,
    razorpay_signature: str,
    webhook_secret: Optional[str] = None,
) -> bool:
    """
    Verify the authenticity of a Razorpay webhook.

    Args:
        payload_body: Raw request body bytes.
        razorpay_signature: X-Razorpay-Signature header value.
        webhook_secret: Webhook secret from Razorpay dashboard.

    Returns:
        True if signature is valid.
    """
    secret = webhook_secret or settings.RAZORPAY_WEBHOOK_SECRET
    if not secret:
        logger.error("RAZORPAY_WEBHOOK_SECRET not configured")
        return False

    expected = hmac.new(
        secret.encode(),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, razorpay_signature)


async def fetch_payment(payment_id: str) -> dict:
    """Fetch payment details from Razorpay."""
    key_id, key_secret = _get_credentials()
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{RAZORPAY_BASE_URL}/payments/{payment_id}",
            auth=(key_id, key_secret),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
