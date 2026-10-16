"""Phase 8 integration checks against disposable PostgreSQL and Redis services."""

from __future__ import annotations

import asyncio
import os

import asyncpg
import httpx
import pytest
import redis.asyncio as redis_async


pytestmark = pytest.mark.skipif(
    os.getenv("PHASE8_INTEGRATION") != "1",
    reason="Set PHASE8_INTEGRATION=1 with disposable PostgreSQL and Redis URLs",
)


def postgres_dsn() -> str:
    return os.environ["PHASE8_POSTGRES_DSN"]


def redis_url() -> str:
    return os.environ["PHASE8_REDIS_URL"]


@pytest.mark.asyncio
async def test_postgres_is_at_expected_head_with_immutable_histories():
    connection = await asyncpg.connect(postgres_dsn())
    try:
        assert await connection.fetchval("SELECT version_num FROM alembic_version") == "0016"
        tables = await connection.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        table_names = {row["tablename"] for row in tables}
        assert {"orders", "scheduled_jobs", "provider_attempts", "spending_ledger_entries"}.issubset(table_names)
        triggers = {
            row["tgname"] for row in await connection.fetch(
                "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal"
            )
        }
        assert {
            "trg_order_events_append_only",
            "trg_provider_receipts_append_only",
            "trg_scheduled_job_attempts_append_only",
            "trg_spending_ledger_append_only",
        }.issubset(triggers)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_redis_duplicate_claim_is_atomic_under_concurrency():
    client = redis_async.from_url(redis_url(), decode_responses=True)
    key = "phase8:dedupe:concurrency"
    try:
        await client.delete(key)

        async def claim():
            return await client.set(key, "1", ex=60, nx=True)

        results = await asyncio.gather(*(claim() for _ in range(50)))
        assert sum(bool(value) for value in results) == 1
        assert await client.ttl(key) > 0
    finally:
        await client.delete(key)
        await client.aclose()


@pytest.mark.asyncio
async def test_application_readiness_checks_real_postgres_and_redis():
    from app.api.main import api

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api),
        base_url="http://phase8.test",
    ) as client:
        response = await client.get("/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "database": "ready", "redis": "ready"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"]
