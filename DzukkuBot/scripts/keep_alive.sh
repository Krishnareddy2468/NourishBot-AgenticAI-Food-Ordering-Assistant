#!/bin/bash
# keep_alive.sh — restarts DzukkuBot on crash, runs for 24 hours
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/logs/dzukku.log"
VENV="$SCRIPT_DIR/env/bin/python3"
DEADLINE=$(( $(date +%s) + 86400 ))  # now + 24 hours

echo "[keep_alive] Started at $(date). Will run until $(date -r $DEADLINE)." >> "$LOG"

while [ $(date +%s) -lt $DEADLINE ]; do
    echo "[keep_alive] Launching DzukkuBot at $(date)" >> "$LOG"
    cd "$SCRIPT_DIR"
    "$VENV" main.py >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "[keep_alive] Bot exited with code $EXIT_CODE at $(date)" >> "$LOG"
    if [ $(date +%s) -lt $DEADLINE ]; then
        echo "[keep_alive] Restarting in 5 seconds..." >> "$LOG"
        sleep 5
    fi
done

echo "[keep_alive] 24-hour window ended at $(date). Exiting." >> "$LOG"
