"""
Central logging configuration.
Logs go to:
  - stdout (so Railway/Procfile capture them)
  - logs/dzukku.log (rotating file, 5 MB x 5 backups)

Call setup_logging() once at process startup (main.py / app.api.main).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import settings

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

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
    # Ensure Swiggy/Zomato MCP modules always log at DEBUG regardless of root level
    logging.getLogger("app.agent.mcp_clients").setLevel(logging.DEBUG)
    logging.getLogger("app.agent.mcp_agent").setLevel(logging.DEBUG)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging initialised — file=%s level=%s", settings.LOG_FILE, settings.LOG_LEVEL
    )
