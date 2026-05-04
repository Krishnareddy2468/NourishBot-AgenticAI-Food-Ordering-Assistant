"""
Async PostgreSQL session factory for Dzukku vNext.

Provides:
  - async_engine / AsyncSessionLocal: for the main FastAPI event loop
  - get_session_factory(): returns a sessionmaker bound to the *current*
    event loop's engine — required when called from the Telegram thread
    (which runs its own loop, distinct from the main uvicorn loop).
"""

import asyncio
import threading

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from typing import AsyncGenerator

from app.core.config import settings

async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── Per-loop engine cache ────────────────────────────────────────────────────
# The Telegram bot runs in its own thread with its own asyncio loop.
# asyncpg connections cannot be shared across loops, so we create a
# separate engine (and session factory) per loop and cache them.

_loop_engines: dict[int, tuple] = {}  # {loop_id: (engine, sessionmaker)}
_engines_lock = threading.Lock()


def _get_or_create_for_loop(loop: asyncio.AbstractEventLoop):
    """Return (engine, sessionmaker) for the given loop, creating if needed."""
    loop_id = id(loop)
    with _engines_lock:
        if loop_id not in _loop_engines:
            engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DB_ECHO,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
            factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            _loop_engines[loop_id] = (engine, factory)
        return _loop_engines[loop_id]


def get_session_factory() -> async_sessionmaker:
    """
    Return an async_sessionmaker safe for the *current* event loop.

    Use this in any code that may run from the Telegram thread or other
    non-main loops (e.g. crud.py helpers).  The main FastAPI routes can
    continue using the module-level ``AsyncSessionLocal`` directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    _, factory = _get_or_create_for_loop(loop)
    return factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
