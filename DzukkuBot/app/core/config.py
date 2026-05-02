"""
Central configuration — reads from environment variables.
All other modules import from here; never read os.getenv directly elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the project root (DzukkuBot/) regardless of where Python is invoked from
ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")


class Settings:
    # ── Paths ──────────────────────────────────────────────────────────────────
    ROOT_DIR:     Path = ROOT_DIR
    STORAGE_DIR:  Path = ROOT_DIR / "storage"
    DATA_DIR:     Path = ROOT_DIR / "data"
    LOGS_DIR:     Path = ROOT_DIR / "logs"
    LOG_FILE:     Path = ROOT_DIR / "logs" / "dzukku.log"
    LOG_LEVEL:    str  = os.getenv("LOG_LEVEL", "INFO").upper()
    XLSX_PATH:    Path = ROOT_DIR / "data" / "Project_Dzukku.xlsx"
    MENU_SHEET:   str  = os.getenv("MENU_SHEET", "Master_Menu")
    CREDS_PATH:   Path = ROOT_DIR / "config" / "credentials.json"

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

    # ── Gemini AI ──────────────────────────────────────────────────────────────
    GEMINI_API_KEY:      str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_PRIMARY:      str = os.getenv("GEMINI_PRIMARY_MODEL",    "gemini-2.5-flash")
    GEMINI_FALLBACK:     str = os.getenv("GEMINI_FALLBACK_MODEL",   "gemini-2.0-flash")
    GEMINI_FALLBACK_2:   str = os.getenv("GEMINI_FALLBACK_2_MODEL", "gemini-1.5-flash")
    GEMINI_MAX_TOKENS:   int = 1024
    AGENT_MAX_ITERATIONS: int = 6

    # ── Google Sheets (deprecated — feature-flagged, will be removed) ──────
    SHEETS_ENABLED:    bool = os.getenv("SHEETS_ENABLED", "false").lower() in ("1", "true", "yes")
    GOOGLE_SHEET_ID:    str = os.getenv("GOOGLE_SHEET_ID", "")
    GOOGLE_CREDENTIALS: str = os.getenv("GOOGLE_CREDENTIALS", "")

    # ── Razorpay (vNext) ────────────────────────────────────────────────────
    RAZORPAY_KEY_ID:       str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET:   str = os.getenv("RAZORPAY_KEY_SECRET", "")
    RAZORPAY_WEBHOOK_SECRET: str = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    # ── JWT Auth (vNext) ────────────────────────────────────────────────────
    JWT_SECRET:          str = os.getenv("JWT_SECRET", "dzukku-dev-secret-change-in-production")
    JWT_EXPIRY_MINUTES:  int = int(os.getenv("JWT_EXPIRY_MINUTES", "480"))

    # ── External Ordering Platforms (Zomato / Swiggy) ──────────────────────────
    ZOMATO_URL: str = os.getenv(
        "ZOMATO_URL",
        "https://www.zomato.com/hyderabad/dzukku-restaurant",
    )
    SWIGGY_URL: str = os.getenv(
        "SWIGGY_URL",
        "https://www.swiggy.com/restaurants/dzukku-restaurant",
    )

    # ── MCP (Model Context Protocol) — Zomato / Swiggy ordering ─────────────────
    # When True, Zomato/Swiggy buttons in Telegram route into the LangGraph
    # MCP agent instead of just opening external app links.
    MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "false").lower() in ("1", "true", "yes")
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
    MCP_SWIGGY_INSTAMART_ENABLED: bool = os.getenv("MCP_SWIGGY_INSTAMART_ENABLED", "false").lower() in ("1", "true", "yes")
    MCP_SWIGGY_DINEOUT_ENABLED:   bool = os.getenv("MCP_SWIGGY_DINEOUT_ENABLED",   "false").lower() in ("1", "true", "yes")
    # Timeouts / retry behaviour
    MCP_TOOL_TIMEOUT_S: int = int(os.getenv("MCP_TOOL_TIMEOUT_S", "45"))
    MCP_RECONNECT_COOLDOWN_S: int = int(os.getenv("MCP_RECONNECT_COOLDOWN_S", "30"))
    # Remote MCP endpoints (override if Zomato/Swiggy ever moves them)
    MCP_ZOMATO_URL:           str = os.getenv("MCP_ZOMATO_URL",           "https://mcp-server.zomato.com/mcp")
    MCP_SWIGGY_FOOD_URL:      str = os.getenv("MCP_SWIGGY_FOOD_URL",      "https://mcp.swiggy.com/food")
    MCP_SWIGGY_INSTAMART_URL: str = os.getenv("MCP_SWIGGY_INSTAMART_URL", "https://mcp.swiggy.com/im")
    MCP_SWIGGY_DINEOUT_URL:   str = os.getenv("MCP_SWIGGY_DINEOUT_URL",   "https://mcp.swiggy.com/dineout")

    # ── FastAPI ────────────────────────────────────────────────────────────────
    PORT:          int = int(os.getenv("PORT", "8000"))
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # ── Restaurant static info ─────────────────────────────────────────────────
    RESTAURANT_NAME:     str = "Dzukku Restaurant"
    RESTAURANT_TAGLINE:  str = "Where every bite hits different ❤️"
    RESTAURANT_TIMINGS:  str = "11:00 AM – 11:00 PM, all days"
    RESTAURANT_LOCATION: str = "Hyderabad, Telangana"
    RESTAURANT_CUISINE:  str = "Indian – Veg & Non-Veg"
    RESTAURANT_DELIVERY: str = "Via Swiggy & Zomato"

    def validate(self) -> None:
        missing = []
        if not self.TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not self.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


settings = Settings()
