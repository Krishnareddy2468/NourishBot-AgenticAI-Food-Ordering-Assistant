"""
Dzukku Bot — Top-level entrypoint.
Delegates to app.api.main, which boots FastAPI + Telegram bot together.
"""

import os
import warnings

# Silence noisy third-party deprecation warnings (urllib3/LibreSSL, google.generativeai)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

import uvicorn

from app.core.logging_config import setup_logging

setup_logging()

from app.api.main import api  # noqa: E402,F401  (imported for uvicorn target)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.api.main:api",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
