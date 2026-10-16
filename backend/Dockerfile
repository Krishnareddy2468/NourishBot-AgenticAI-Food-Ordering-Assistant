# syntax=docker/dockerfile:1.6
#
# NourishBot Bot — production image
# =============================
# - Python 3.11 slim base
# - Node 20 added so `npx mcp-remote` can run when MCP_ENABLED=true
# - mcp-remote pre-installed globally to avoid first-run npx download in prod
# - Non-root user
# - Optional MCP_AUTH_DIR mount point at /run/mcp-auth (read-only secret volume)
#
# Build:    docker build -t nourishbot-bot .
# Run:      docker run --env-file .env -p 8000:8000 nourishbot-bot
# With MCP: see docker-compose.yml (mounts ~/.mcp-auth as a secret volume)

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NODE_MAJOR=20

# System deps: curl/ca-certs for Node, build-essential for some wheels (pandas etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g mcp-remote \
    && apt-get purge -y --auto-remove curl gnupg build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
ARG APP_UID=10001
RUN useradd --uid ${APP_UID} --create-home --shell /usr/sbin/nologin nourishbot

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app   ./app
COPY main.py .
COPY data  ./data
COPY alembic ./alembic
COPY alembic.ini .

# Runtime dirs
RUN mkdir -p /app/config /app/logs /app/storage /run/mcp-auth \
    && chown -R nourishbot:nourishbot /app /run/mcp-auth

USER nourishbot

# Sensible defaults — override at deploy time:
#   MCP_ENABLED=false  → in-house scheduled ordering only
#   MCP_ENABLED=true   → verify provider scheduling support, then set
#                        MCP_AUTH_DIR and mount the OAuth cache securely
ENV PORT=8000 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    MCP_ENABLED=false \
    MCP_TRANSACTIONAL_ENABLED=false \
    PUBLIC_TRANSACTIONAL_RELEASE=false \
    MCP_AUTH_DIR="" \
    NPX_BIN=npx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=4).read()"

# FastAPI is the entrypoint; it boots the Telegram bot in a background thread.
CMD ["python", "main.py"]
