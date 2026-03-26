import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    enable_web_chat: bool
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_models_raw: str
    llm_app_name: str
    llm_site_url: str
    llm_temperature: float
    mcp_command: str
    mcp_args_raw: str
    mcp_connect_timeout: int
    mcp_tool_timeout: int
    mcp_retry_cooldown: int
    telegram_bot_token: str | None
    voice_model: str
    voice_base_url: str
    voice_api_key: str | None

    @property
    def mcp_args(self) -> list[str]:
        return [part.strip() for part in self.mcp_args_raw.split(",") if part.strip()]

    @property
    def llm_models(self) -> list[str]:
        models = [part.strip() for part in self.llm_models_raw.split(",") if part.strip()]
        if not models and self.llm_model:
            return [self.llm_model]
        return models or ["gemini-1.5-flash"]

    @property
    def llm_default_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if "openrouter.ai" in (self.llm_base_url or ""):
            headers["HTTP-Referer"] = self.llm_site_url
            headers["X-Title"] = self.llm_app_name
        return headers


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    voice_api_key = os.getenv("VOICE_API_KEY") or llm_api_key
    return AppConfig(
        enable_web_chat=_env_bool("ENABLE_WEB_CHAT", True),
        llm_api_key=llm_api_key,
        llm_base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        llm_model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
        llm_models_raw=os.getenv("LLM_MODELS", os.getenv("LLM_MODEL", "gemini-1.5-flash")),
        llm_app_name=os.getenv("LLM_APP_NAME", "Swiggybot"),
        llm_site_url=os.getenv("LLM_SITE_URL", "http://localhost:3000"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        mcp_command=os.getenv("MCP_COMMAND", "npx"),
        mcp_args_raw=os.getenv(
            "MCP_ARGS",
            "-y,mcp-remote@0.1.37,https://mcp-server.zomato.com/mcp",
        ),
        mcp_connect_timeout=_env_int("MCP_CONNECT_TIMEOUT", 30),
        mcp_tool_timeout=_env_int("MCP_TOOL_TIMEOUT", 25),
        mcp_retry_cooldown=_env_int("MCP_RETRY_COOLDOWN", 60),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        voice_model=os.getenv("VOICE_MODEL", "gemini-1.5-flash"),
        voice_base_url=os.getenv("VOICE_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")),
        voice_api_key=voice_api_key,
    )


app_config = load_config()
