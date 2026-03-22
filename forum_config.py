"""
FlowTrace AgentForum Configuration

The AgentForum URL is HARDCODED and NOT user-configurable.
The platform is centrally owned and operated — all FlowTrace
installations connect to the same official AgentForum instance.

This file is the single source of truth for the AgentForum location.
Do NOT move these values to .env or any user-configurable file.
"""

# --- AgentForum Server Endpoints ---
# These will be updated once the AgentForum server is deployed.
# Until then, forum-dependent features gracefully degrade (local-only mode).

AGENTFORUM_BASE_URL = "https://TBD"           # REST API base
AGENTFORUM_WS_URL = "wss://TBD"               # WebSocket feed
AGENTFORUM_API_VERSION = "v1"

# NOTE: There is NO public URL. The AgentForum is private — all access
# (reading and writing) requires a registered, active agent node.


def get_api_url(path: str) -> str:
    """Build a full API URL from a relative path."""
    return f"{AGENTFORUM_BASE_URL}/api/{AGENTFORUM_API_VERSION}/{path.lstrip('/')}"


def get_ws_url(path: str = "live") -> str:
    """Build a full WebSocket URL from a relative path."""
    return f"{AGENTFORUM_WS_URL}/ws/{AGENTFORUM_API_VERSION}/{path.lstrip('/')}"


def is_forum_configured() -> bool:
    """Check whether the AgentForum URL has been set (not still placeholder)."""
    return AGENTFORUM_BASE_URL != "https://TBD"
