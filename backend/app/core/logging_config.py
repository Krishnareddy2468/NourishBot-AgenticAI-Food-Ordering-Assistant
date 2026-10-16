"""
Central logging configuration.
Logs go to:
  - stdout (so Railway/Procfile capture them)
  - logs/nourishbot.log (rotating file, 5 MB x 5 backups)

Call setup_logging() once at process startup (main.py / app.api.main).
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from app.core.config import settings

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


class JsonFormatter(logging.Formatter):
    """Compact structured logs for alert routing without serializing request data."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("event", "request_id", "method", "path", "status_code", "duration_ms"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = (
        JsonFormatter()
        if settings.LOG_FORMAT == "json"
        else logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    # Replace any pre-existing handlers (e.g. from logging.basicConfig)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Provider payloads stay out of verbose logs in production.
    provider_level = logging.INFO if settings.is_production else logging.DEBUG
    logging.getLogger("app.agent.mcp_clients").setLevel(provider_level)
    logging.getLogger("app.agent.mcp_agent").setLevel(provider_level)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging initialised — file=%s level=%s", settings.LOG_FILE, settings.LOG_LEVEL
    )
