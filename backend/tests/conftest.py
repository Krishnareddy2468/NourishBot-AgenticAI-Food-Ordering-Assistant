"""Test path setup for running pytest from the repository root."""

import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Modules construct the SQLAlchemy engine at import time. Unit tests never
# connect, but they still need a syntactically valid driver URL.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://nourishbot:nourishbot@127.0.0.1:5432/nourishbot_test",
)
os.environ.setdefault(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg2://nourishbot:nourishbot@127.0.0.1:5432/nourishbot_test",
)
os.environ.setdefault("TELEGRAM_BOT_ENABLED", "false")
