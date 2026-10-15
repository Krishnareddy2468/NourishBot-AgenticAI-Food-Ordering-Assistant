import hashlib
import hmac

from app.api.routes.whatsapp import _message_text, _valid_signature
from app.core.config import settings


def test_whatsapp_signature_validation(monkeypatch):
    monkeypatch.setattr(settings, "WHATSAPP_APP_SECRET", "test-secret")
    body = b'{"object":"whatsapp_business_account"}'
    digest = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()

    assert _valid_signature(body, f"sha256={digest}") is True
    assert _valid_signature(body, "sha256=wrong") is False


def test_extracts_text_and_interactive_replies():
    assert _message_text({"type": "text", "text": {"body": "Schedule lunch"}}) == "Schedule lunch"
    assert _message_text({
        "type": "interactive",
        "interactive": {"button_reply": {"id": "provider", "title": "NourishBot"}},
    }) == "NourishBot"


def test_extracts_shared_location_for_route_search():
    message = {
        "type": "location",
        "location": {"latitude": 17.4483, "longitude": 78.3915},
    }

    text = _message_text(message)

    assert "17.4483,78.3915" in text
    assert "route origin" in text
