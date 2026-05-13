#!/usr/bin/env bash
# scripts/export_mcp_auth.sh
# =========================
# Packages the local mcp-remote OAuth token cache (~/.mcp-auth) into a
# single tarball that can be uploaded to a cloud secret store and mounted
# back into the bot container at /run/mcp-auth (matching MCP_AUTH_DIR).
#
# Usage:
#   ./scripts/export_mcp_auth.sh                # writes ./mcp-auth-export.tar.gz
#   ./scripts/export_mcp_auth.sh /tmp/out.tgz   # custom output path
#
# Then on your cloud:
#   1. Upload the tarball as a secret/file (Railway, Render, Fly.io, K8s
#      Secret, Docker secret, etc.)
#   2. Mount it (or the extracted contents) at the path you set in
#      MCP_AUTH_DIR (default in our docker-compose: /run/mcp-auth)
#   3. Set MCP_ENABLED=true in the deployed env
#
# SECURITY: this tarball contains live OAuth refresh tokens for your
# Zomato / Swiggy account. Treat it like a password. Do NOT commit it.

set -euo pipefail

SRC="${HOME}/.mcp-auth"
OUT="${1:-./mcp-auth-export.tar.gz}"

if [ ! -d "${SRC}" ]; then
    echo "ERROR: ${SRC} does not exist."
    echo "Run an OAuth login first, e.g.:"
    echo "    npx -y mcp-remote https://mcp-server.zomato.com/mcp"
    echo "    npx -y mcp-remote https://mcp.swiggy.com/food"
    exit 1
fi

if [ -z "$(ls -A "${SRC}" 2>/dev/null)" ]; then
    echo "ERROR: ${SRC} is empty — no tokens cached. Run mcp-remote OAuth first."
    exit 1
fi

tar -czf "${OUT}" -C "${SRC}" .
chmod 600 "${OUT}"

echo "Wrote ${OUT}"
echo "Size: $(du -h "${OUT}" | cut -f1)"
echo
echo "Next steps:"
echo "  1. Upload ${OUT} to your cloud secret store (treat as a password)."
echo "  2. Extract it on the deployment host into the dir you mount as"
echo "     MCP_AUTH_DIR (e.g. /run/mcp-auth in our docker-compose)."
echo "  3. In the deployed env: MCP_ENABLED=true, MCP_AUTH_DIR=/run/mcp-auth"
