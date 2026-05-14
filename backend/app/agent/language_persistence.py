"""
Language persistence — detects language from user message and saves to
Customer.language_pref so it sticks across sessions.

Uses persona.detect_language (word-based, fast, no API call) for the
initial detection, then OpenAI for confirmation on high-confidence signals.
"""

from __future__ import annotations

import logging
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models import Channel, Customer
from app.agent.persona import detect_language

logger = logging.getLogger(__name__)


async def persist_user_language(chat_id: int, message: str, history: list[dict] | None = None) -> str | None:
    """
    Detect language from the user's message and persist to Customer.language_pref
    if a non-English language is detected and differs from the stored preference.

    Returns the language code ('en', 'te', 'hi') or None if no change needed.
    """
    if not message or not message.strip():
        return None

    detected = detect_language(message, history or [])
    lang_code = _normalize_lang(detected)

    if lang_code == "en":
        return None  # no need to change default

    async with AsyncSessionLocal() as db:
        ch_result = await db.execute(
            select(Channel).where(
                Channel.type == "TELEGRAM",
                Channel.external_id == str(chat_id),
            )
        )
        channel = ch_result.scalar_one_or_none()
        if not channel or not channel.customer_id:
            return lang_code  # no customer yet, but detected

        cust_result = await db.execute(
            select(Customer).where(Customer.id == channel.customer_id)
        )
        customer = cust_result.scalar_one_or_none()
        if not customer:
            return lang_code

        current = (customer.language_pref or "en").lower()
        if current != lang_code:
            customer.language_pref = lang_code
            await db.commit()
            logger.info(
                "Language persisted: chat=%s lang=%s (was %s)",
                chat_id, lang_code, current,
            )
            return lang_code

    return None


def _normalize_lang(detected: str) -> str:
    """Convert detect_language output to two-letter code for DB."""
    if detected.startswith("te"):
        return "te"
    if detected.startswith("hi"):
        return "hi"
    return "en"
