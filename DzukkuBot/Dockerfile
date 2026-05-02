# syntax=docker/dockerfile:1.6
#
# Dzukku Bot — production image
# =============================
# - Python 3.11 slim base
# - Node 20 added so `npx mcp-remote` can run when MCP_ENABLED=true
# - mcp-remote pre-installed globally to avoid first-run npx download in prod
# - Non-root user
# - Optional MCP_AUTH_DIR mount point at /run/mcp-auth (read-only secret volume)
#
# Build:    docker build -t dzukku-bot .
# Run:      docker run --env-file .env -p 8000:8000 dzukku-bot
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
RUN useradd --uid ${APP_UID} --create-home --shell /usr/sbin/nologin dzukku

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app   ./app
COPY main.py .
COPY data  ./data
COPY config ./config

# Runtime dirs
RUN mkdir -p /app/logs /app/storage /run/mcp-auth \
    && chown -R dzukku:dzukku /app /run/mcp-auth

USER dzukku

# Sensible defaults — override at deploy time:
#   MCP_ENABLED=false  → cloud uses redirect-link UX (recommended)
#   MCP_ENABLED=true   → also set MCP_AUTH_DIR=/run/mcp-auth and mount a
#                        secret volume containing your local ~/.mcp-auth
ENV PORT=8000 \
    LOG_LEVEL=INFO \
    MCP_ENABLED=false \
    MCP_AUTH_DIR="" \
    NPX_BIN=npx

EXPOSE 8000

# FastAPI is the entrypoint; it boots the Telegram bot in a background thread.
CMD ["python", "main.py"]
