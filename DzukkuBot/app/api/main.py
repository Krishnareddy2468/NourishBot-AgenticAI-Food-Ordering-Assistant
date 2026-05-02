"""
Dzukku Bot — Main Entrypoint
============================
Runs:
  1. Telegram bot (python-telegram-bot polling, in its own thread)
  2. FastAPI REST API (uvicorn, main thread) — serves POS frontend
  3. WebSocket endpoint for real-time updates (KDS, waiter, admin)
  4. Outbox worker as background task

Database: PostgreSQL (via SQLAlchemy async + Alembic migrations).
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.db.session import async_engine
from app.bot.telegram import build_app as build_telegram_app
from app.realtime.ws_manager import ws_manager
from app.workers.outbox_worker import outbox_worker_loop

load_dotenv()

from app.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


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


# ── Telegram thread ───────────────────────────────────────────────────────────

def run_telegram():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tg_app = build_telegram_app()
    logger.info("Telegram bot starting (polling)…")
    tg_app.run_polling(drop_pending_updates=True, stop_signals=None)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Start outbox worker as background task
    outbox_task = asyncio.create_task(outbox_worker_loop())
    logger.info("Outbox worker started.")

    # Start Telegram bot in its own thread
    tg_thread = threading.Thread(target=run_telegram, daemon=True)
    tg_thread.start()
    logger.info("Telegram thread started.")

    yield

    outbox_task.cancel()
    await async_engine.dispose()
    logger.info("Shutting down…")


api = FastAPI(
    title="Dzukku Backend API",
    version="3.0.0",
    lifespan=lifespan,
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register v1 routes ───────────────────────────────────────────────────────

from app.api.routes.auth import router as auth_router
from app.api.routes.menu import router as menu_router
from app.api.routes.orders import router as orders_router
from app.api.routes.tables import router as tables_router
from app.api.routes.kitchen import router as kitchen_router
from app.api.routes.payments import router as payments_router
from app.api.routes.deliveries import router as deliveries_router

api.include_router(auth_router)
api.include_router(menu_router)
api.include_router(orders_router)
api.include_router(tables_router)
api.include_router(kitchen_router)
api.include_router(payments_router)
api.include_router(deliveries_router)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@api.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket, restaurant_id: int = 1):
    """Real-time WebSocket for KDS, waiter, admin clients."""
    client_id = str(id(websocket))
    await ws_manager.connect(websocket, restaurant_id, client_id)
    try:
        while True:
            # Keep connection alive; client messages can be used for subscriptions
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(restaurant_id, client_id)


# ── Health ────────────────────────────────────────────────────────────────────

@api.get("/api/health")
async def health():
    return {"ok": True, "service": "dzukku-backend", "version": "3.0.0"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting Dzukku Backend (FastAPI + Telegram)…")
    uvicorn.run(
        "app.api.main:api",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
