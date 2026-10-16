"""
NourishBot Bot — Main Entrypoint
============================
Runs:
  1. Telegram bot (python-telegram-bot polling, same event loop as FastAPI)
  2. FastAPI REST API (uvicorn, main thread) — serves POS frontend
  3. WebSocket endpoint for real-time updates (KDS, waiter, admin)
  4. Outbox worker as background task

Database: PostgreSQL (via SQLAlchemy async + Alembic migrations).
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.db.session import async_engine
from app.bot.telegram import build_app as build_telegram_app
from app.realtime.ws_manager import ws_manager
from app.workers.outbox_worker import outbox_worker_loop
from app.core.config import settings
from app.auth.jwt import decode_access_token

load_dotenv()

from app.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── DB init on startup ────────────────────────────────────────────────────────

async def init_db():
    """Verify PostgreSQL connectivity on startup."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        logger.info("PostgreSQL connection verified.")
    except Exception as e:
        logger.error("PostgreSQL connection failed: %s", e)
        raise


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.is_production:
        settings.validate()
    await init_db()

    # Start outbox worker as background task
    outbox_task = asyncio.create_task(outbox_worker_loop())
    logger.info("Outbox worker started.")

    tg_app = None
    telegram_enabled = env_flag("TELEGRAM_BOT_ENABLED", default=True)
    if telegram_enabled:
        try:
            # Start Telegram bot in the SAME event loop as FastAPI so all DB
            # sessions (asyncpg) are bound to one loop — avoids "attached to a
            # different loop" RuntimeErrors.
            tg_app = build_telegram_app()
            await tg_app.initialize()
            await tg_app.start()
            await tg_app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram bot started (polling, same event loop).")
        except Exception as e:
            logger.exception(
                "Telegram bot startup failed. API will continue without Telegram polling. "
                "Set TELEGRAM_BOT_ENABLED=false to disable bot startup explicitly. Error: %s",
                e,
            )
            tg_app = None
    else:
        logger.info("Telegram bot startup disabled via TELEGRAM_BOT_ENABLED=false.")

    yield

    # Graceful shutdown: stop Telegram first, then other tasks
    if tg_app is not None:
        try:
            await tg_app.updater.stop()
            await tg_app.stop()
            await tg_app.shutdown()
        except Exception as e:
            logger.warning("Telegram shutdown warning: %s", e)

    outbox_task.cancel()
    await async_engine.dispose()
    logger.info("Shutting down…")


api = FastAPI(
    title="NourishBot Backend API",
    version="3.0.0",
    lifespan=lifespan,
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(self)"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@api.middleware("http")
async def csrf_origin_guard(request: Request, call_next):
    """Cookie-authenticated state changes must originate from an approved frontend."""
    customer_browser_mutation = request.url.path.startswith("/api/v1/customer/")
    public_auth_mutation = request.url.path in {
        "/api/v1/auth/login",
        "/api/v1/auth/signup",
        "/api/v1/auth/logout",
    }
    has_cookie_identity = bool(
        request.cookies.get(settings.AUTH_COOKIE_NAME)
        or request.cookies.get(settings.CUSTOMER_IDENTITY_COOKIE_NAME)
    )
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and (has_cookie_identity or customer_browser_mutation or public_auth_mutation)
    ):
        origin = request.headers.get("origin", "")
        if origin not in settings.ALLOWED_ORIGINS:
            return JSONResponse(status_code=403, content={"detail": "Untrusted request origin"})
    return await call_next(request)


@api.middleware("http")
async def rate_limit(request: Request, call_next):
    """Redis-backed per-client protection for public HTTP endpoints."""
    if request.method == "OPTIONS" or request.url.path in ("/api/health", "/api/ready"):
        return await call_next(request)

    path = request.url.path
    if path in {"/api/v1/auth/login", "/api/v1/auth/signup"}:
        bucket, limit = "login", settings.LOGIN_RATE_LIMIT_PER_MINUTE
    elif "/webhook" in path:
        bucket, limit = "webhook", settings.WEBHOOK_RATE_LIMIT_PER_MINUTE
    else:
        bucket, limit = "api", settings.API_RATE_LIMIT_PER_MINUTE

    try:
        from app.core.redis_client import get_redis
        redis = await get_redis()
        if redis is not None:
            client = request.client.host if request.client else "unknown"
            minute = int(time.time() // 60)
            key = f"rate:{bucket}:{client}:{minute}"
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, 65)
            if count > limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests"},
                    headers={"Retry-After": "60"},
                )
    except Exception as exc:
        logger.warning("Rate limiter unavailable: %s", exc)
    return await call_next(request)


# ── Register v1 routes ───────────────────────────────────────────────────────

from app.api.routes.auth import router as auth_router
from app.api.routes.menu import router as menu_router
from app.api.routes.orders import router as orders_router
from app.api.routes.tables import router as tables_router
from app.api.routes.kitchen import router as kitchen_router
from app.api.routes.payments import router as payments_router
from app.api.routes.deliveries import router as deliveries_router
from app.api.routes.reservations import router as reservations_router
from app.api.routes.staff import router as staff_router
from app.api.routes.invoices import router as invoices_router
from app.api.routes.whatsapp import router as whatsapp_router
from app.api.routes.customer_privacy import router as customer_privacy_router
from app.api.routes.scheduler import router as scheduler_router
from app.api.routes.providers import router as providers_router
from app.api.routes.meal_plans import router as meal_plans_router
from app.api.routes.fulfillment import router as fulfillment_router

api.include_router(auth_router)
api.include_router(menu_router)
api.include_router(orders_router)
api.include_router(tables_router)
api.include_router(kitchen_router)
api.include_router(payments_router)
api.include_router(deliveries_router)
api.include_router(reservations_router)
api.include_router(staff_router)
api.include_router(invoices_router)
api.include_router(whatsapp_router)
api.include_router(customer_privacy_router)
api.include_router(scheduler_router)
api.include_router(providers_router)
api.include_router(meal_plans_router)
api.include_router(fulfillment_router)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@api.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket for KDS, waiter, admin clients."""
    if websocket.headers.get("origin", "") not in settings.ALLOWED_ORIGINS:
        await websocket.close(code=4403, reason="Untrusted origin")
        return
    protocols = [part.strip() for part in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    if "nourishbot" not in protocols:
        await websocket.close(code=4401, reason="Authentication required")
        return
    try:
        raw_token = websocket.cookies.get(settings.AUTH_COOKIE_NAME)
        if not raw_token:
            raise ValueError("missing session")
        payload = decode_access_token(raw_token)
        restaurant_id = int(payload["restaurant_id"])
        if restaurant_id <= 0:
            raise ValueError("invalid tenant")
    except Exception:
        await websocket.close(code=4401, reason="Invalid or expired credentials")
        return
    client_id = str(id(websocket))
    await ws_manager.connect(websocket, restaurant_id, client_id, subprotocol="nourishbot")
    try:
        while True:
            # Keep connection alive; client messages can be used for subscriptions
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(restaurant_id, client_id)


# ── Health ────────────────────────────────────────────────────────────────────

@api.get("/api/health")
async def health():
    return {"ok": True, "service": "nourishbot-backend", "version": "3.0.0"}


@api.get("/api/ready")
async def readiness():
    """Deployment readiness probe for PostgreSQL and Redis."""
    try:
        async with async_engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        from app.core.redis_client import get_redis
        redis = await get_redis()
        if redis is None or not await redis.ping():
            raise RuntimeError("Redis is unavailable")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Service dependencies are unavailable") from exc
    return {"ok": True, "database": "ready", "redis": "ready"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting NourishBot Backend (FastAPI + Telegram)…")
    uvicorn.run(
        "app.api.main:api",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
