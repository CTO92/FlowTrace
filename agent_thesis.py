"""
FlowTrace ThesisAgent

Synthesizes local research into trade theses and publishes them
to the AgentForum for cross-network debate. Monitors responses
and feeds challenges back to local agents for counter-analysis.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta

from forum_client import get_forum_client
from forum_config import is_forum_configured
from node_identity import generate_agent_id, get_forum_status
from learning_config_manager import load_config, get_intensity_thresholds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")

# Session identity
_agent_id = None
_theses_published_today = 0
_last_reset_date = None


def _get_agent_id() -> str:
    global _agent_id
    if _agent_id is None:
        _agent_id = generate_agent_id("ThesisAgent")
    return _agent_id


def _check_daily_limit() -> bool:
    """Check if we've exceeded the daily thesis publication limit."""
    global _theses_published_today, _last_reset_date

    today = datetime.now(timezone.utc).date()
    if _last_reset_date != today:
        _theses_published_today = 0
        _last_reset_date = today

    thresholds = get_intensity_thresholds()
    max_theses = thresholds.get("max_theses_per_day", 10)

    return _theses_published_today < max_theses


def _get_recent_local_signals(min_confidence: float = 70, hours: int = 4) -> list:
    """Fetch recent high-confidence signals from the local knowledge graph."""
    if not os.path.exists(KG_DB_PATH):
        return []

    try:
        conn = sqlite3.connect(KG_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute("""
            SELECT * FROM signals
            WHERE confidence >= ? AND timestamp >= ?
            ORDER BY confidence DESC
            LIMIT 20
        """, (min_confidence, cutoff))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error fetching local signals: {e}")
        return []


async def _is_forum_sparse() -> bool:
    """
    Check if the forum has very few threads — indicating it's new or underutilized.
    When sparse, agents should be MORE eager to post, not less.
    """
    try:
        client = get_forum_client()
        result = await client.search_threads(status="open", page_size=1)
        if not result or not isinstance(result, dict):
            return True  # can't tell, assume sparse
        # If fewer than 10 open threads, forum is sparse
        total = result.get("total", 0)
        return total < 10
    except Exception:
        return True


async def publish_high_confidence_signals() -> list:
    """
    Review recent local signals and publish qualifying ones as forum theses.
    Returns list of published thread IDs.

    CONTRIBUTION MOTIVATION: Agents are coded to WANT to contribute.
    On a sparse forum (few threads), confidence thresholds are lowered
    and the agent publishes more aggressively to seed discussions.
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return []

    if not _check_daily_limit():
        logger.debug("[ThesisAgent] Daily thesis limit reached")
        return []

    global _theses_published_today

    config = load_config()
    thresholds = get_intensity_thresholds()
    min_confidence = thresholds.get("min_publish_confidence", 0.75) * 100  # convert to 0-100

    # EAGER CONTRIBUTION: On a sparse/new forum, lower the confidence
    # threshold significantly to seed discussions and build momentum.
    forum_sparse = await _is_forum_sparse()
    if forum_sparse:
        min_confidence = min(min_confidence, 55)  # publish anything above 55% confidence
        logger.info("[ThesisAgent] Forum is sparse — lowering publish threshold to seed discussions")

    signals = _get_recent_local_signals(min_confidence=min_confidence, hours=24 if forum_sparse else 4)
    if not signals:
        return []

    client = get_forum_client()
    published = []

    for signal in signals:
        ticker = signal.get("target_ticker")
        if not ticker:
            continue

        # Determine direction from expected move
        expected_move = signal.get("expected_move_pct", 0)
        if expected_move > 0:
            direction = "BULLISH"
        elif expected_move < 0:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = signal.get("confidence", 0) / 100.0  # normalize to 0-1
        reasoning = signal.get("reasoning", "")
        summary = signal.get("summary", "")

        thesis = f"{summary} {reasoning}".strip()
        if not thesis:
            thesis = f"Signal detected for {ticker}: expected {expected_move:+.1f}% move"

        # Enrich thesis with structured evidence if available
        try:
            from evidence_schema import evidence_to_debate_text
            evidence_json = signal.get("evidence_summary")
            if evidence_json and isinstance(evidence_json, str):
                import json
                evidence_list = json.loads(evidence_json)
                evidence_text = " | ".join(
                    evidence_to_debate_text(ev) for ev in evidence_list[:3]
                )
                thesis = f"{thesis}\n\nEvidence: {evidence_text}"
        except Exception:
            pass

        # Add valuation context if available
        val_json = signal.get("valuation_summary")
        if val_json:
            try:
                import json
                val = json.loads(val_json) if isinstance(val_json, str) else val_json
                verdict = val.get("valuation_verdict", "")
                upside = val.get("upside_to_mid", 0)
                if verdict and upside:
                    thesis += f"\n\nValuation: {verdict} ({upside:+.1f}% to fair value mid)"
            except Exception:
                pass

        result = await client.create_thread(
            ticker=ticker,
            direction=direction,
            thesis_summary=thesis[:2000],  # cap at 2000 chars
            time_horizon=5,
            confidence=confidence,
            agent_id=_get_agent_id(),
        )

        if result:
            thread_id = result.get("thread_id")
            published.append(thread_id)
            _theses_published_today += 1
            logger.info(f"[ThesisAgent] Published thesis: {ticker} {direction} (conf={confidence:.0%})")

            if not _check_daily_limit():
                break

    return published


async def monitor_thesis_responses(thread_ids: list) -> list:
    """
    Check for new responses to our published theses.
    Returns challenges that need counter-analysis.
    """
    if not is_forum_configured() or not thread_ids:
        return []

    client = get_forum_client()
    challenges = []

    for thread_id in thread_ids:
        thread = await client.get_thread(thread_id)
        if not thread:
            continue

        posts = thread.get("posts", [])
        for post in posts:
            # Skip our own posts
            if post.get("agent_id", "").startswith(get_forum_client().node_id):
                continue

            if post.get("post_type") == "CHALLENGE":
                challenges.append({
                    "thread_id": thread_id,
                    "post_id": post.get("post_id"),
                    "challenger_agent": post.get("agent_id"),
                    "content": post.get("content"),
                    "ticker": thread.get("ticker"),
                    "our_direction": thread.get("direction"),
                })

    return challenges


async def update_thesis_confidence(thread_id: str, new_confidence: float, reason: str) -> bool:
    """
    Post an UPDATE to a thread revising our confidence based on debate outcomes.
    """
    if not is_forum_configured() or get_forum_status() != "active":
        return False

    client = get_forum_client()
    content = f"Confidence revised to {new_confidence:.0%}. Reason: {reason}"

    result = await client.post_to_thread(
        thread_id=thread_id,
        post_type="UPDATE",
        content=content,
        agent_id=_get_agent_id(),
        data_sources={"confidence_update": new_confidence, "reason": reason},
    )

    return result is not None


async def synthesize_local_signals(ticker: str) -> dict:
    """
    Aggregate all local agent findings for a specific ticker
    into a summary suitable for thesis formulation.
    """
    signals = _get_recent_local_signals(min_confidence=50, hours=24)
    ticker_signals = [s for s in signals if s.get("target_ticker") == ticker]

    if not ticker_signals:
        return {"ticker": ticker, "signal_count": 0, "summary": "No recent signals"}

    confidences = [s.get("confidence", 0) for s in ticker_signals]
    moves = [s.get("expected_move_pct", 0) for s in ticker_signals]

    avg_conf = sum(confidences) / len(confidences)
    avg_move = sum(moves) / len(moves) if moves else 0

    reasons = [s.get("reasoning", "") for s in ticker_signals if s.get("reasoning")]
    event_types = [s.get("event_type", "") for s in ticker_signals if s.get("event_type")]

    return {
        "ticker": ticker,
        "signal_count": len(ticker_signals),
        "avg_confidence": avg_conf,
        "avg_expected_move": avg_move,
        "direction": "BULLISH" if avg_move > 0 else "BEARISH" if avg_move < 0 else "NEUTRAL",
        "event_types": list(set(event_types)),
        "reasoning_summary": " | ".join(reasons[:3]),
    }
