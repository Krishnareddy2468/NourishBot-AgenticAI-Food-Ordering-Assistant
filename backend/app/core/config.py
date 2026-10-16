"""
Central configuration — reads from environment variables.
All other modules import from here; never read os.getenv directly elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the project root (NourishBotBot/) regardless of where Python is invoked from
ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")


class Settings:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").strip().lower()
    # ── Paths ──────────────────────────────────────────────────────────────────
    ROOT_DIR:     Path = ROOT_DIR
    STORAGE_DIR:  Path = ROOT_DIR / "storage"
    DATA_DIR:     Path = ROOT_DIR / "data"
    LOGS_DIR:     Path = ROOT_DIR / "logs"
    LOG_FILE:     Path = ROOT_DIR / "logs" / "nourishbot.log"
    LOG_LEVEL:    str  = os.getenv("LOG_LEVEL", "INFO").upper()
    CREDS_PATH:   Path = ROOT_DIR / "config" / "credentials.json"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── PostgreSQL (vNext) ────────────────────────────────────────────────────
    DATABASE_URL:      str = os.getenv("DATABASE_URL", "")
    DATABASE_URL_SYNC: str = os.getenv("DATABASE_URL_SYNC", "")
    DB_ECHO:          bool = os.getenv("DB_ECHO", "false").lower() in ("1", "true", "yes")
    DEFAULT_RESTAURANT_ID: int = int(os.getenv("DEFAULT_RESTAURANT_ID", "1"))

    # ── Object storage (vNext) ────────────────────────────────────────────────
    STORAGE_PROVIDER: str = os.getenv("STORAGE_PROVIDER", "local")  # local | s3 | gcs | azure
    STORAGE_BUCKET:   str = os.getenv("STORAGE_BUCKET", "")
    STORAGE_BASE_URL: str = os.getenv("STORAGE_BASE_URL", "")

    # ── Telegram ───────────────────────────────────────────────────────────────
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")

    # ── WhatsApp Cloud API ────────────────────────────────────────────────────
    WHATSAPP_ENABLED: bool = os.getenv("WHATSAPP_ENABLED", "false").lower() in ("1", "true", "yes")
    WHATSAPP_ACCESS_TOKEN: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WHATSAPP_VERIFY_TOKEN: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    WHATSAPP_APP_SECRET: str = os.getenv("WHATSAPP_APP_SECRET", "")
    WHATSAPP_GRAPH_API_VERSION: str = os.getenv("WHATSAPP_GRAPH_API_VERSION", "")

    # ── OpenAI LLM ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY:      str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_PRIMARY:      str = os.getenv("OPENAI_PRIMARY_MODEL",    "gpt-4o")
    OPENAI_FALLBACK:     str = os.getenv("OPENAI_FALLBACK_MODEL",   "gpt-4o-mini")
    OPENAI_FALLBACK_2:   str = os.getenv("OPENAI_FALLBACK_2_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS:   int = 1024
    AGENT_MAX_ITERATIONS: int = 6

    # ── Razorpay (vNext) ────────────────────────────────────────────────────
    RAZORPAY_KEY_ID:       str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET:   str = os.getenv("RAZORPAY_KEY_SECRET", "")
    RAZORPAY_WEBHOOK_SECRET: str = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    # ── JWT Auth (vNext) ────────────────────────────────────────────────────
    JWT_SECRET:          str = os.getenv("JWT_SECRET", "nourishbot-dev-secret-change-in-production")
    JWT_EXPIRY_MINUTES:  int = int(os.getenv("JWT_EXPIRY_MINUTES", "480"))
    AUTH_COOKIE_NAME: str = os.getenv("AUTH_COOKIE_NAME", "nourishbot_session")
    CUSTOMER_IDENTITY_COOKIE_NAME: str = os.getenv(
        "CUSTOMER_IDENTITY_COOKIE_NAME",
        "nourishbot_customer_identity",
    )
    AUTH_COOKIE_SAMESITE: str = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
    CONFIRMATION_TTL_MINUTES: int = int(os.getenv("CONFIRMATION_TTL_MINUTES", "15"))
    IDENTITY_LINK_TTL_MINUTES: int = int(os.getenv("IDENTITY_LINK_TTL_MINUTES", "10"))
    CONSENT_POLICY_VERSION: str = os.getenv("CONSENT_POLICY_VERSION", "2026-10")
    MESSAGE_RETENTION_DAYS: int = int(os.getenv("MESSAGE_RETENTION_DAYS", "30"))
    LOCATION_RETENTION_DAYS: int = int(os.getenv("LOCATION_RETENTION_DAYS", "7"))
    PROVIDER_RESPONSE_RETENTION_DAYS: int = int(os.getenv("PROVIDER_RESPONSE_RETENTION_DAYS", "30"))

    # ── External Ordering Platforms (Zomato / Swiggy) ──────────────────────────
    ZOMATO_URL: str = os.getenv(
        "ZOMATO_URL",
        "https://www.zomato.com/hyderabad/nourishbot-restaurant",
    )
    SWIGGY_URL: str = os.getenv(
        "SWIGGY_URL",
        "https://www.swiggy.com/restaurants/nourishbot-restaurant",
    )

    # ── MCP (Model Context Protocol) — Zomato / Swiggy ordering ─────────────────
    # When True, Zomato/Swiggy buttons in Telegram route into the LangGraph
    # MCP agent instead of just opening external app links.
    MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "false").lower() in ("1", "true", "yes")
    MCP_TRANSACTIONAL_ENABLED: bool = os.getenv("MCP_TRANSACTIONAL_ENABLED", "false").lower() in ("1", "true", "yes")
    # Path to the npx executable used to spawn mcp-remote. Override if your
    # node install lives somewhere unusual. mcp-remote handles OAuth on its
    # own loopback callback and caches tokens under ~/.mcp-auth/ by default.
    NPX_BIN: str = os.getenv("NPX_BIN", "npx")
    # Cloud / Docker deployments don't share the owner's home directory.
    # Set MCP_AUTH_DIR to a path that holds the pre-generated mcp-remote
    # token cache (mounted from a secret volume). When set, it is exported
    # to mcp-remote subprocesses via MCP_REMOTE_CONFIG_DIR. Leave empty for
    # local dev — mcp-remote will use ~/.mcp-auth as usual.
    MCP_AUTH_DIR: str = os.getenv("MCP_AUTH_DIR", "")
    # Per-MCP toggles (set to "false" to skip mounting that server's tools)
    MCP_ZOMATO_ENABLED: bool = os.getenv("MCP_ZOMATO_ENABLED", "true").lower() in ("1", "true", "yes")
    MCP_SWIGGY_FOOD_ENABLED:     bool = os.getenv("MCP_SWIGGY_FOOD_ENABLED",     "true").lower()  in ("1", "true", "yes")
    # Timeouts / retry behaviour
    MCP_TOOL_TIMEOUT_S: int = int(os.getenv("MCP_TOOL_TIMEOUT_S", "45"))
    MCP_RECONNECT_COOLDOWN_S: int = int(os.getenv("MCP_RECONNECT_COOLDOWN_S", "30"))
    # Remote MCP endpoints (override if Zomato/Swiggy ever moves them)
    MCP_ZOMATO_URL:           str = os.getenv("MCP_ZOMATO_URL",           "https://mcp-server.zomato.com/mcp")
    MCP_SWIGGY_FOOD_URL:      str = os.getenv("MCP_SWIGGY_FOOD_URL",      "https://mcp.swiggy.com/food")

    # ── Google Maps restaurant and route discovery ─────────────────────────
    GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")
    MAPS_REQUEST_TIMEOUT_S: int = int(os.getenv("MAPS_REQUEST_TIMEOUT_S", "15"))

    # ── Scheduled meal policy ─────────────────────────────────────────────────
    MIN_SCHEDULE_LEAD_MINUTES: int = int(os.getenv("MIN_SCHEDULE_LEAD_MINUTES", "30"))
    MAX_SCHEDULE_DAYS: int = int(os.getenv("MAX_SCHEDULE_DAYS", "14"))
    SCHEDULE_ACTIVATION_LEAD_MINUTES: int = int(os.getenv("SCHEDULE_ACTIVATION_LEAD_MINUTES", "20"))
    SCHEDULE_CONFIRMATION_REMINDER_MINUTES: int = int(os.getenv("SCHEDULE_CONFIRMATION_REMINDER_MINUTES", "1440"))
    SCHEDULE_FULFILLMENT_REMINDER_MINUTES: int = int(os.getenv("SCHEDULE_FULFILLMENT_REMINDER_MINUTES", "60"))
    SCHEDULE_LATE_GRACE_MINUTES: int = int(os.getenv("SCHEDULE_LATE_GRACE_MINUTES", "10"))
    SCHEDULE_LATE_POLICY: str = os.getenv("SCHEDULE_LATE_POLICY", "MANUAL_REVIEW").strip().upper()
    SCHEDULE_JOB_MAX_ATTEMPTS: int = int(os.getenv("SCHEDULE_JOB_MAX_ATTEMPTS", "5"))
    SCHEDULE_JOB_BATCH_SIZE: int = int(os.getenv("SCHEDULE_JOB_BATCH_SIZE", "50"))
    SCHEDULE_LEASE_SECONDS: int = int(os.getenv("SCHEDULE_LEASE_SECONDS", "120"))
    MEAL_PLAN_MAX_HORIZON_DAYS: int = int(os.getenv("MEAL_PLAN_MAX_HORIZON_DAYS", "31"))
    ROUTINE_MATERIALIZATION_HORIZON_DAYS: int = int(
        os.getenv("ROUTINE_MATERIALIZATION_HORIZON_DAYS", "14")
    )
    PLAN_CONFIRMATION_REMINDER_HOURS: int = int(
        os.getenv("PLAN_CONFIRMATION_REMINDER_HOURS", "24")
    )
    PLANNED_MEAL_REMINDER_MINUTES: int = int(
        os.getenv("PLANNED_MEAL_REMINDER_MINUTES", "60")
    )
    SCHEDULE_RETRY_BASE_SECONDS: int = int(os.getenv("SCHEDULE_RETRY_BASE_SECONDS", "30"))
    SCHEDULE_RETRY_MAX_SECONDS: int = int(os.getenv("SCHEDULE_RETRY_MAX_SECONDS", "1800"))
    SCHEDULER_HEARTBEAT_STALE_SECONDS: int = int(os.getenv("SCHEDULER_HEARTBEAT_STALE_SECONDS", "180"))
    OUTBOX_LEASE_SECONDS: int = int(os.getenv("OUTBOX_LEASE_SECONDS", "600"))
    OUTBOX_BATCH_SIZE: int = int(os.getenv("OUTBOX_BATCH_SIZE", "50"))
    OUTBOX_CONCURRENCY: int = int(os.getenv("OUTBOX_CONCURRENCY", "10"))
    OUTBOX_MAX_ATTEMPTS: int = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "8"))
    OUTBOX_RETRY_BASE_SECONDS: int = int(os.getenv("OUTBOX_RETRY_BASE_SECONDS", "5"))
    OUTBOX_RETRY_MAX_SECONDS: int = int(os.getenv("OUTBOX_RETRY_MAX_SECONDS", "300"))

    # ── FastAPI ────────────────────────────────────────────────────────────────
    PORT:          int = int(os.getenv("PORT", "8000"))
    API_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "120"))
    LOGIN_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("LOGIN_RATE_LIMIT_PER_MINUTE", "10"))
    WEBHOOK_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("WEBHOOK_RATE_LIMIT_PER_MINUTE", "300"))
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    GOOGLE_REDIRECT_URI: str = os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://localhost:8000/api/v1/auth/oauth/google/callback",
    ).strip()
    OAUTH_STATE_COOKIE_NAME: str = os.getenv(
        "OAUTH_STATE_COOKIE_NAME",
        "nourishbot_oauth_state",
    )
    OAUTH_STATE_TTL_SECONDS: int = int(os.getenv("OAUTH_STATE_TTL_SECONDS", "600"))
    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def google_oauth_enabled(self) -> bool:
        return bool(self.GOOGLE_CLIENT_ID and self.GOOGLE_CLIENT_SECRET and self.GOOGLE_REDIRECT_URI)

    # ── Synthetic canteen information ──────────────────────────────────────────
    RESTAURANT_NAME:     str = "NourishBot Demo Canteen"
    RESTAURANT_TAGLINE:  str = "Schedule ahead for pickup, drive-through, or home delivery."
    RESTAURANT_TIMINGS:  str = "8:00 AM – 8:00 PM, weekdays"
    RESTAURANT_LOCATION: str = "Demo Campus Food Court"
    RESTAURANT_CUISINE:  str = "Indian canteen meals – Veg & Non-Veg"
    RESTAURANT_DELIVERY: str = "Scheduled pickup, drive-through, and home delivery"

    def validate(self) -> None:
        missing = []
        telegram_enabled = os.getenv("TELEGRAM_BOT_ENABLED", "true").lower() in ("1", "true", "yes")
        if telegram_enabled and not self.TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        if not self.DATABASE_URL_SYNC:
            missing.append("DATABASE_URL_SYNC")
        if (
            self.JWT_SECRET == "nourishbot-dev-secret-change-in-production"
            or len(self.JWT_SECRET) < 32
            or "replace" in self.JWT_SECRET.lower()
        ):
            missing.append("JWT_SECRET")
        if self.WHATSAPP_ENABLED:
            whatsapp_required = {
                "WHATSAPP_ACCESS_TOKEN": self.WHATSAPP_ACCESS_TOKEN,
                "WHATSAPP_PHONE_NUMBER_ID": self.WHATSAPP_PHONE_NUMBER_ID,
                "WHATSAPP_VERIFY_TOKEN": self.WHATSAPP_VERIFY_TOKEN,
                "WHATSAPP_APP_SECRET": self.WHATSAPP_APP_SECRET,
                "WHATSAPP_GRAPH_API_VERSION": self.WHATSAPP_GRAPH_API_VERSION,
            }
            missing.extend(name for name, value in whatsapp_required.items() if not value)
        if self.is_production and (
            not self.ALLOWED_ORIGINS or "*" in self.ALLOWED_ORIGINS
        ):
            missing.append("ALLOWED_ORIGINS(explicit production origins)")
        if self.AUTH_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
            missing.append("AUTH_COOKIE_SAMESITE(lax|strict|none)")
        if bool(self.GOOGLE_CLIENT_ID) != bool(self.GOOGLE_CLIENT_SECRET):
            missing.append("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (configure both or neither)")
        if self.OAUTH_STATE_TTL_SECONDS <= 0:
            missing.append("OAUTH_STATE_TTL_SECONDS(positive integer)")
        if not self.CUSTOMER_IDENTITY_COOKIE_NAME:
            missing.append("CUSTOMER_IDENTITY_COOKIE_NAME")
        if self.IDENTITY_LINK_TTL_MINUTES <= 0:
            missing.append("IDENTITY_LINK_TTL_MINUTES(positive integer)")
        if not self.CONSENT_POLICY_VERSION:
            missing.append("CONSENT_POLICY_VERSION")
        if min(
            self.MESSAGE_RETENTION_DAYS,
            self.LOCATION_RETENTION_DAYS,
            self.PROVIDER_RESPONSE_RETENTION_DAYS,
        ) <= 0:
            missing.append("privacy retention periods(positive day counts)")
        if self.SCHEDULE_LATE_POLICY not in {"MANUAL_REVIEW"}:
            missing.append("SCHEDULE_LATE_POLICY(MANUAL_REVIEW)")
        if min(
            self.SCHEDULE_JOB_MAX_ATTEMPTS,
            self.SCHEDULE_JOB_BATCH_SIZE,
            self.SCHEDULE_LEASE_SECONDS,
            self.SCHEDULE_RETRY_BASE_SECONDS,
            self.SCHEDULE_RETRY_MAX_SECONDS,
            self.OUTBOX_LEASE_SECONDS,
            self.OUTBOX_BATCH_SIZE,
            self.OUTBOX_CONCURRENCY,
            self.OUTBOX_MAX_ATTEMPTS,
            self.MEAL_PLAN_MAX_HORIZON_DAYS,
            self.ROUTINE_MATERIALIZATION_HORIZON_DAYS,
            self.PLAN_CONFIRMATION_REMINDER_HOURS,
            self.PLANNED_MEAL_REMINDER_MINUTES,
        ) <= 0:
            missing.append("scheduler/outbox limits(positive integers)")
        if self.ROUTINE_MATERIALIZATION_HORIZON_DAYS > self.MEAL_PLAN_MAX_HORIZON_DAYS:
            missing.append("ROUTINE_MATERIALIZATION_HORIZON_DAYS(<= MEAL_PLAN_MAX_HORIZON_DAYS)")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


settings = Settings()
