"""
Async Redis client — shared connection pool for NourishBot.

Provides a single async Redis instance for:
  - Session state caching (get_session / save_session in crud.py)
  - Rate limiting
  - Short-term memory store
  - Celery broker / backend
"""

from __future__ import annotations

import logging
import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis | None:
    """Return a shared async Redis client, or None if unavailable."""
    global _pool
    if _pool is not None:
        return _pool
    try:
        _pool = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=3,
            socket_keepalive=True,
            retry_on_timeout=True,
        )
        await _pool.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL)
    except Exception as e:
        logger.warning("Redis unavailable (%s) — session cache disabled", e)
        _pool = None
    return _pool


async def close_redis() -> None:
    """Close the Redis connection pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("Redis connection closed.")


def _session_key(chat_id: str | int, channel_type: str = "TELEGRAM") -> str:
    return f"session:{channel_type.lower()}:{chat_id}"


async def cache_session(chat_id: str | int, data: dict, channel_type: str = "TELEGRAM") -> None:
    """Write session dict to Redis with 24-hour TTL."""
    r = await get_redis()
    if r is None:
        return
    try:
        import json
        key = _session_key(chat_id, channel_type)
        await r.set(key, json.dumps(data), ex=86400)
    except Exception as e:
        logger.debug("Redis cache write failed (chat=%s): %s", chat_id, e)


async def get_cached_session(chat_id: str | int, channel_type: str = "TELEGRAM") -> dict | None:
    """Read session dict from Redis. Returns None on miss or failure."""
    r = await get_redis()
    if r is None:
        return None
    try:
        import json
        key = _session_key(chat_id, channel_type)
        raw = await r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("Redis cache read failed (chat=%s): %s", chat_id, e)
        return None


async def invalidate_session_cache(chat_id: str | int, channel_type: str = "TELEGRAM") -> None:
    """Delete session from Redis cache."""
    r = await get_redis()
    if r is None:
        return
    try:
        key = _session_key(chat_id, channel_type)
        await r.delete(key)
    except Exception as e:
        logger.debug("Redis cache delete failed (chat=%s): %s", chat_id, e)


async def claim_once(key: str, ttl_seconds: int = 86400) -> bool:
    """Atomically claim an event key. Returns False for duplicates."""
    r = await get_redis()
    if r is None:
        # Keep the webhook available during a cache outage. Provider-side
        # idempotency still applies, but duplicate suppression is degraded.
        return True
    try:
        return bool(await r.set(f"dedupe:{key}", "1", ex=ttl_seconds, nx=True))
    except Exception as e:
        logger.warning("Redis idempotency claim failed for %s: %s", key, e)
        return True
