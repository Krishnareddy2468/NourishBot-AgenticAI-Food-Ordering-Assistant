"""Central redaction and hashing for provider requests, receipts, and logs."""

import hashlib
import json
import re
from typing import Any


_SENSITIVE_PARTS = {
    "token", "secret", "authorization", "cookie", "password", "api_key", "apikey",
    "address", "phone", "mobile", "email", "payment", "card", "cvv", "upi",
    "latitude", "longitude", "lat", "lng", "location",
}
_TOKEN_RE = re.compile(r"(?i)(bearer\s+)[a-z0-9._~+\-/=]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){12,19}(?!\d)")
_KV_RE = re.compile(
    r"(?i)\b(token|secret|password|authorization|address|phone|payment|card|cvv|upi)\s*[:=]\s*([^,;\n]+)"
)
_QUOTED_KV_RE = re.compile(
    r'''(?i)(["'](?:token|secret|password|authorization|address|phone|payment|card|cvv|upi)["']\s*:\s*)(["'])(.*?)\2'''
)


def _sensitive_key(key: object) -> bool:
    normalized = str(key or "").lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_PARTS)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _sensitive_key(key) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


def redact_text(value: str) -> str:
    text = str(value or "")
    text = _TOKEN_RE.sub(r"\1[REDACTED]", text)
    text = _QUOTED_KV_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        text,
    )
    text = _KV_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _CARD_RE.sub("[REDACTED_NUMBER]", text)
    return _PHONE_RE.sub("[REDACTED_NUMBER]", text)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
