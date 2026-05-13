"""
WebSocket / SSE connection manager for Dzukku vNext.

Manages live connections from KDS, waiter, admin clients.
Subscriptions are per-restaurant and optionally filtered by event type.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events."""

    def __init__(self):
        # {restaurant_id: {client_id: WebSocket}}
        self._connections: dict[int, dict[str, WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, restaurant_id: int, client_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            if restaurant_id not in self._connections:
                self._connections[restaurant_id] = {}
            self._connections[restaurant_id][client_id] = websocket
        logger.info("WS connected: client=%s restaurant=%s", client_id, restaurant_id)

    async def disconnect(self, restaurant_id: int, client_id: str) -> None:
        async with self._lock:
            if restaurant_id in self._connections:
                self._connections[restaurant_id].pop(client_id, None)
                if not self._connections[restaurant_id]:
                    del self._connections[restaurant_id]
        logger.info("WS disconnected: client=%s restaurant=%s", client_id, restaurant_id)

    async def broadcast(self, restaurant_id: int, event: dict) -> None:
        """Send an event dict to all connected clients for a restaurant."""
        async with self._lock:
            connections = list(self._connections.get(restaurant_id, {}).values())

        message = json.dumps(event)
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                logger.warning("Failed to send to WS, will be cleaned up on next disconnect")

    async def send_to_client(self, restaurant_id: int, client_id: str, event: dict) -> None:
        """Send an event to a specific client."""
        async with self._lock:
            ws = self._connections.get(restaurant_id, {}).get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                logger.warning("Failed to send to client %s", client_id)

    @property
    def active_connections(self) -> int:
        return sum(len(clients) for clients in self._connections.values())


# Singleton instance
ws_manager = ConnectionManager()
