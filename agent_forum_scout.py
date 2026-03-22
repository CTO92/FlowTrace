"""
FlowTrace ForumScoutAgent

Monitors the AgentForum for relevant discussions and brings
external agent perspectives back to the local system. Filters
for tickers on the local watchlist or in the knowledge graph.
"""

import os
import sqlite3
import logging
from typing import Optional

from forum_client import get_forum_client
from forum_config import is_forum_configured
from node_identity import generate_agent_id, get_forum_status
from learning_config_manager import get_intensity_thresholds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")

# Session identity
_agent_id = None


def _get_agent_id() -> str:
    global _agent_id
    if _agent_id is None:
        _agent_id = generate_agent_id("ForumScoutAgent")
    return _agent_id


def _get_watchlist_tickers() -> list:
    """Get tickers from both the watchlist and knowledge graph hub companies."""
    tickers = set()

    # From portfolio watchlist
    try:
        from portfolio_manager import get_watchlist
        for item in get_watchlist():
            tickers.add(item["ticker"])
    except Exception:
        pass

    # From knowledge graph hub companies
    try:
        if os.path.exists(KG_DB_PATH):
            conn = sqlite3.connect(KG_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM companies LIMIT 50")
            for row in cursor.fetchall():
                tickers.add(row[0])
            conn.close()
    except Exception:
        pass

    return list(tickers)


async def scan_relevant_threads() -> list:
    """
    Scan the forum for threads relevant to our watchlist tickers.
    Returns a list of thread summaries with debate info.
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return []

    client = get_forum_client()
    tickers = _get_watchlist_tickers()

    if not tickers:
        logger.debug("[ForumScout] No tickers to monitor")
        return []

    relevant_threads = []

    for ticker in tickers[:20]:  # Limit to top 20 to manage API calls
        result = await client.search_threads(ticker=ticker, status="open")
        if result and isinstance(result, dict):
            threads = result.get("items", [])
            for thread in threads:
                relevant_threads.append({
                    "thread_id": thread.get("thread_id"),
                    "ticker": thread.get("ticker"),
                    "direction": thread.get("direction"),
                    "thesis_summary": thread.get("thesis_summary"),
                    "confidence": thread.get("confidence"),
                    "time_horizon": thread.get("time_horizon"),
                    "created_by": thread.get("created_by"),
                    "post_count": thread.get("post_count", 0),
                })

    logger.info(f"[ForumScout] Found {len(relevant_threads)} relevant threads")
    return relevant_threads


async def get_thread_debate(thread_id: str) -> Optional[dict]:
    """
    Fetch a full thread with all debate posts.
    Returns the thread data with posts for analysis by other agents.
    """
    if not is_forum_configured():
        return None

    client = get_forum_client()
    return await client.get_thread(thread_id)


async def get_high_reputation_conflicts(local_direction: str, ticker: str) -> list:
    """
    Find threads where high-reputation agents disagree with our local analysis.
    These are the most valuable learning opportunities.
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return []

    client = get_forum_client()
    result = await client.search_threads(ticker=ticker, status="open")

    if not result or not isinstance(result, dict):
        return []

    conflicts = []
    for thread in result.get("items", []):
        thread_direction = thread.get("direction")
        if thread_direction and thread_direction != local_direction:
            conflicts.append(thread)

    return conflicts


async def summarize_network_sentiment(ticker: str) -> dict:
    """
    Aggregate forum sentiment for a specific ticker across all active threads.
    Returns: {bullish_count, bearish_count, neutral_count, avg_confidence, top_thesis}
    """
    if not is_forum_configured():
        return {"bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "avg_confidence": 0, "top_thesis": None}

    client = get_forum_client()
    result = await client.search_threads(ticker=ticker, status="open", page_size=50)

    if not result or not isinstance(result, dict):
        return {"bullish_count": 0, "bearish_count": 0, "neutral_count": 0, "avg_confidence": 0, "top_thesis": None}

    threads = result.get("items", [])
    bullish = sum(1 for t in threads if t.get("direction") == "BULLISH")
    bearish = sum(1 for t in threads if t.get("direction") == "BEARISH")
    neutral = sum(1 for t in threads if t.get("direction") == "NEUTRAL")

    confidences = [t.get("confidence", 0) for t in threads if t.get("confidence")]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    # Top thesis = highest confidence thread
    top = max(threads, key=lambda t: t.get("confidence", 0)) if threads else None

    return {
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "avg_confidence": round(avg_conf, 4),
        "top_thesis": top,
    }


async def monitor_live_feed(callback):
    """
    Connect to the forum WebSocket and forward relevant events to a callback.
    The callback receives event dicts for threads matching our watchlist.
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return

    client = get_forum_client()
    tickers = set(_get_watchlist_tickers())

    async def _filter_event(event: dict):
        """Only forward events for tickers we care about."""
        event_ticker = event.get("data", {}).get("ticker")
        if event_ticker and event_ticker in tickers:
            await callback(event)

    client.on_event(_filter_event)
    await client.connect_ws()
    await client.listen_ws()
