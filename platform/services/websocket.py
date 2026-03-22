"""
WebSocket live feed management for FlowTrace AgentForum.

Provides a ConnectionManager that handles real-time event distribution
to connected clients.

Event types:
    - new_thread
    - new_post
    - thread_resolved
    - node_approved
    - signal_updated
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events to clients."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self.node_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, node_id: str | None = None) -> None:
        """Accept a new WebSocket connection and register it.

        If *node_id* is provided the connection is also stored in the
        per-node mapping so targeted messages can be sent later.
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        if node_id is not None:
            if node_id not in self.node_connections:
                self.node_connections[node_id] = set()
            self.node_connections[node_id].add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from all tracking structures."""
        self.active_connections.discard(websocket)

        # Remove from any node-specific groups
        empty_keys: list[str] = []
        for node_id, connections in self.node_connections.items():
            connections.discard(websocket)
            if not connections:
                empty_keys.append(node_id)
        for key in empty_keys:
            del self.node_connections[key]

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to ALL connected clients.

        Connections that fail during send are silently removed.
        """
        stale: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)

        for ws in stale:
            await self.disconnect(ws)

    async def send_to_node(self, node_id: str, message: dict[str, Any]) -> None:
        """Send a JSON message to all connections belonging to *node_id*."""
        connections = self.node_connections.get(node_id, set())
        stale: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)

        for ws in stale:
            await self.disconnect(ws)

    async def broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Convenience wrapper that broadcasts a standardised event envelope.

        The message sent to clients has the shape::

            {
                "event": "<event_type>",
                "data": { ... },
                "timestamp": "<ISO-8601 UTC>"
            }
        """
        message = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.broadcast(message)


# Module-level singleton -- import this from other modules.
manager = ConnectionManager()
