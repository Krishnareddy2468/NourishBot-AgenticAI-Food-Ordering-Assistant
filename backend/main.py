import os
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# need to load env vars before importing our modules since they read API keys on init
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import app_config
from app.services.telegram_bot import telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """handles startup and shutdown - starts telegram bot in bg"""
    logger.info("Starting AI Restaurant Order Bot...")
    asyncio.create_task(telegram_bot.start())
    logger.info("Backend is ready!")
    yield
    logger.info("Shutting down...")
    await telegram_bot.stop()


app = FastAPI(
    title="AI Restaurant Order Bot",
    description="AI-powered restaurant order automation - Telegram-first backend",
    version="1.0.0",
    lifespan=lifespan,
)

# allow requests from the next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if app_config.enable_web_chat:
    from app.routes.chat import router as chat_router
    from app.routes.restaurants import router as restaurant_router

    app.include_router(chat_router)
    app.include_router(restaurant_router)


@app.get("/")
async def root():
    return {
        "service": "AI Restaurant Order Bot",
        "status": "running",
        "version": "1.0.0",
        "telegram_bot": {
            "running": telegram_bot._running,
            "configured": bool(telegram_bot.token),
            "last_error": telegram_bot._last_error,
        },
        "web_chat_enabled": app_config.enable_web_chat,
        "endpoints": {
            "docs": "/docs",
            **(
                {
                    "chat": "/api/chat/message",
                    "restaurants": "/api/restaurants/",
                }
                if app_config.enable_web_chat
                else {}
            ),
        },
    }


@app.get("/health")
async def health():
    telegram_status = "connected" if telegram_bot._running else "degraded"
    return {
        "status": "healthy" if telegram_bot._running or not telegram_bot.token else "degraded",
        "telegram": {
            "status": telegram_status,
            "configured": bool(telegram_bot.token),
            "last_error": telegram_bot._last_error,
        },
        "web_chat_enabled": app_config.enable_web_chat,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")
