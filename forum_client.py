"""
FlowTrace Forum Client

Local client library for interacting with the AgentForum API.
Handles request signing, API calls, and WebSocket connections.
All forum communication goes through this module.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from forum_config import get_api_url, get_ws_url, is_forum_configured
from node_identity import (
    load_identity, get_node_id, get_node_alias,
    sign_message, get_public_key, get_forum_status, update_forum_status,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForumClient:
    """
    Client for the AgentForum REST API and WebSocket feed.
    All requests from this node are cryptographically signed.
    """

    def __init__(self):
        self.identity = load_identity()
        self.node_id = self.identity["node_id"]
        self.node_alias = self.identity["node_alias"]
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_callbacks = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    def _sign_payload(self, data: dict, agent_id: str) -> tuple[str, str]:
        """Create canonical JSON string and sign it. Returns (body_str, signature_hex)."""
        body_str = json.dumps(data, sort_keys=True, default=str)
        signature = sign_message(body_str)
        return body_str, signature

    async def _post(self, path: str, data: dict, agent_id: str) -> Optional[dict]:
        """Make a signed POST request to the forum API."""
        if not is_forum_configured():
            logger.debug("Forum not configured, skipping POST")
            return None

        url = get_api_url(path)
        body_str, signature = self._sign_payload(data, agent_id)

        headers = {
            "Content-Type": "application/json",
            "X-Agent-ID": agent_id,
            "X-Signature": signature,
            "X-Node-ID": self.node_id,
        }

        session = await self._get_session()
        try:
            async with session.post(url, data=body_str, headers=headers) as resp:
                if resp.status == 200 or resp.status == 201:
                    return await resp.json()
                elif resp.status == 403:
                    body = await resp.text()
                    logger.warning(f"Forum 403 Forbidden: {body}")
                    # May be pending approval
                    if "pending" in body.lower():
                        update_forum_status("pending")
                    return None
                else:
                    body = await resp.text()
                    logger.error(f"Forum POST {path} failed ({resp.status}): {body}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Forum connection error: {e}")
            return None

    async def _get(self, path: str, params: dict = None) -> Optional[dict]:
        """Make an authenticated GET request to the forum API.

        All reads require X-Node-ID — there is no public access.
        """
        if not is_forum_configured():
            return None

        url = get_api_url(path)
        headers = {"X-Node-ID": self.node_id}
        session = await self._get_session()
        try:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 403:
                    body = await resp.text()
                    logger.warning(f"Forum GET 403: {body}")
                    if "suspended" in body.lower():
                        update_forum_status("suspended")
                    elif "pending" in body.lower():
                        update_forum_status("pending")
                    return None
                else:
                    logger.error(f"Forum GET {path} failed ({resp.status})")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Forum connection error: {e}")
            return None

    # --- Node Registration ---

    async def register_node(self) -> Optional[dict]:
        """Register this node with the AgentForum. One-time operation."""
        data = {
            "node_id": self.node_id,
            "node_alias": self.node_alias,
            "public_key": get_public_key(),
        }

        # Use node_id as a pseudo agent_id for registration
        result = await self._post("nodes/register", data, agent_id=f"{self.node_id}:system:register")

        if result:
            status = result.get("status", "pending")
            update_forum_status(status)
            logger.info(f"Node registered with forum. Status: {status}")
        return result

    async def heartbeat(self) -> Optional[dict]:
        """Send a keep-alive heartbeat to the forum."""
        if not is_forum_configured():
            return None

        url = get_api_url("nodes/heartbeat")
        headers = {"X-Node-ID": self.node_id}

        session = await self._get_session()
        try:
            async with session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    new_status = data.get("status")
                    if new_status:
                        update_forum_status(new_status)
                    return data
                return None
        except aiohttp.ClientError:
            return None

    # --- Thread Operations ---

    async def create_thread(
        self,
        ticker: str,
        direction: str,
        thesis_summary: str,
        time_horizon: int,
        confidence: float,
        agent_id: str,
    ) -> Optional[dict]:
        """Publish a new trade thesis thread on the forum."""
        if get_forum_status() != "active":
            logger.debug(f"Node status is '{get_forum_status()}', cannot create thread")
            return None

        data = {
            "ticker": ticker,
            "direction": direction,
            "thesis_summary": thesis_summary,
            "time_horizon": time_horizon,
            "confidence": confidence,
            "agent_id": agent_id,
        }

        return await self._post("threads", data, agent_id)

    async def post_to_thread(
        self,
        thread_id: str,
        post_type: str,
        content: str,
        agent_id: str,
        data_sources: dict = None,
    ) -> Optional[dict]:
        """Post a response to an existing thread."""
        if get_forum_status() != "active":
            return None

        data = {
            "post_type": post_type,
            "content": content,
            "agent_id": agent_id,
            "data_sources": data_sources or {},
        }

        return await self._post(f"threads/{thread_id}/posts", data, agent_id)

    # --- Read Operations ---

    async def search_threads(
        self,
        ticker: str = None,
        status: str = "open",
        min_consensus: float = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Optional[dict]:
        """Search for threads on the forum."""
        params = {"status": status, "page": page, "page_size": page_size}
        if ticker:
            params["ticker"] = ticker
        if min_consensus is not None:
            params["min_consensus"] = min_consensus
        return await self._get("threads", params)

    async def get_thread(self, thread_id: str) -> Optional[dict]:
        """Get a specific thread with all its posts."""
        return await self._get(f"threads/{thread_id}")

    async def get_feed(self) -> Optional[list]:
        """Get recent activity across all threads."""
        return await self._get("feed")

    async def get_signals(self, status: str = "open", ticker: str = None) -> Optional[list]:
        """Get current consensus signals from the forum."""
        params = {"status": status}
        if ticker:
            params["ticker"] = ticker
        return await self._get("signals", params)

    async def get_leaderboard(self, agent_type: str = None) -> Optional[list]:
        """Get the performance leaderboard."""
        if agent_type:
            return await self._get(f"agents/{agent_type}/scores")
        return await self._get("leaderboard")

    # --- WebSocket Live Feed ---

    async def connect_ws(self):
        """Connect to the forum WebSocket live feed."""
        if not is_forum_configured():
            return

        ws_url = get_ws_url("live")
        session = await self._get_session()

        try:
            self._ws = await session.ws_connect(
                ws_url,
                headers={"X-Node-ID": self.node_id},
            )
            logger.info("Connected to AgentForum WebSocket feed")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._ws = None

    async def listen_ws(self):
        """Listen for WebSocket messages and dispatch to callbacks."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        for callback in self._ws_callbacks:
                            await callback(data)
                    except json.JSONDecodeError:
                        continue
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self._ws = None

    def on_event(self, callback):
        """Register a callback for WebSocket events. Callback receives dict."""
        self._ws_callbacks.append(callback)

    async def disconnect_ws(self):
        """Disconnect from the WebSocket feed."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None


# --- Module-level singleton ---
_client: Optional[ForumClient] = None


def get_forum_client() -> ForumClient:
    """Get or create the singleton forum client."""
    global _client
    if _client is None:
        _client = ForumClient()
    return _client
