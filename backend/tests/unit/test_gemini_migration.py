"""Regression tests for the legacy-provider-to-Gemini runtime migration."""

from types import SimpleNamespace

import pytest

from app.core import gemini_client
from app.core.config import settings


def test_configuration_exposes_only_gemini_llm_settings():
    assert settings.GEMINI_PRIMARY.startswith("gemini-")
    assert settings.GEMINI_FALLBACK.startswith("gemini-")
    assert settings.GEMINI_FALLBACK_2.startswith("gemini-")
    assert settings.GEMINI_MAX_TOKENS > 0
    legacy_key_name = "OPEN" + "AI_API_KEY"
    assert not hasattr(settings, legacy_key_name)


@pytest.mark.asyncio
async def test_gemini_client_uses_json_mode_and_falls_back(monkeypatch):
    calls = []

    class Models:
        async def generate_content(self, *, model, contents, config):
            calls.append((model, contents, config))
            if model == "gemini-primary-test":
                raise RuntimeError("temporary provider failure")
            return SimpleNamespace(
                text='{"goal":"SUPPORT"}',
                candidates=[SimpleNamespace(finish_reason="STOP")],
            )

    fake_client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    monkeypatch.setattr(gemini_client, "_client", fake_client)
    monkeypatch.setattr(settings, "GEMINI_PRIMARY", "gemini-primary-test")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK", "gemini-fallback-test")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_2", "gemini-last-test")

    text, finish_reason, model = await gemini_client.generate_text(
        "Return JSON",
        temperature=0.1,
        json_mode=True,
        max_output_tokens=321,
    )

    assert text == '{"goal":"SUPPORT"}'
    assert finish_reason == "STOP"
    assert model == "gemini-fallback-test"
    assert [call[0] for call in calls] == ["gemini-primary-test", "gemini-fallback-test"]
    assert calls[-1][2].response_mime_type == "application/json"
    assert calls[-1][2].max_output_tokens == 321
