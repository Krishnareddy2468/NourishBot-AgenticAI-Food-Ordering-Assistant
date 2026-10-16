"""Shared asynchronous Google Gemini text-generation client with model fallbacks."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)
_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError(
                "Google Gemini SDK is not installed; run pip install -r requirements.txt"
            ) from exc
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


async def generate_text(
    prompt: str,
    *,
    temperature: float,
    json_mode: bool = False,
    max_output_tokens: int | None = None,
) -> tuple[str, str, str]:
    """Generate text, trying the configured Gemini models in order."""
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "Google Gemini SDK is not installed; run pip install -r requirements.txt"
        ) from exc

    client = _get_client()
    models = dict.fromkeys((
        settings.GEMINI_PRIMARY,
        settings.GEMINI_FALLBACK,
        settings.GEMINI_FALLBACK_2,
    ))
    last_error: Exception | None = None
    for model in models:
        if not model:
            continue
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens or settings.GEMINI_MAX_TOKENS,
                response_mime_type="application/json" if json_mode else "text/plain",
            )
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            candidate = (response.candidates or [None])[0]
            finish_reason = str(getattr(candidate, "finish_reason", "") or "")
            if not text:
                raise RuntimeError(f"Gemini model {model} returned an empty response")
            return text, finish_reason, model
        except Exception as exc:
            last_error = exc
            logger.warning("Gemini model %s failed: %s", model, exc.__class__.__name__)
    raise RuntimeError("All configured Gemini models failed") from last_error
