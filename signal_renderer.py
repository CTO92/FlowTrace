"""
FlowTrace Signal Renderer

Builds multi-section recommendation cards from consensus signals.
Separates rendering logic from app.py so the dashboard can call
a single function and receive a structured dict ready for display.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")


# ---------------------------------------------------------------------------
# Freshness thresholds (minutes) per trading style
# ---------------------------------------------------------------------------

_FRESHNESS_THRESHOLDS = {
    "day_trader": {"fresh": 30, "aging": 120},
    "swing_single_week": {"fresh": 240, "aging": 720},
    "swing_multi_week": {"fresh": 720, "aging": 2880},
    "value_investor": {"fresh": 2880, "aging": 10080},
}

_DEFAULT_STYLE = "swing_multi_week"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_recommendation(consensus_signal: dict, portfolio: dict = None) -> dict:
    """
    Build a multi-section recommendation card from a consensus signal.

    Args:
        consensus_signal: A consensus signal dict as produced by
            ``agent_consensus.calculate_consensus`` / fetched from the DB.
        portfolio: Optional pre-fetched portfolio context dict.  When *None*
            the function will attempt to look it up via ``get_portfolio_context``.

    Returns:
        Dict with keys: header, thesis, evidence, risks, debate,
        portfolio, confidence_breakdown, valuation, catalysts,
        trade_plan, freshness.
    """
    ticker = consensus_signal.get("ticker", "")

    # --- header -----------------------------------------------------------
    regime = consensus_signal.get("market_regime")
    if not regime:
        regime = _safe_json_field(consensus_signal, "market_snapshot", {}).get("regime")

    signal_age = _compute_age_minutes(consensus_signal.get("timestamp"))

    header = {
        "ticker": ticker,
        "direction": consensus_signal.get("direction", ""),
        "consensus_score": consensus_signal.get("consensus_score", 0),
        "expected_move": consensus_signal.get("expected_move_pct", 0),
        "time_horizon": consensus_signal.get("time_horizon_days", 5),
        "signal_age": signal_age,
        "regime": regime,
    }

    # --- thesis -----------------------------------------------------------
    thesis = consensus_signal.get("reasoning", "")

    # --- evidence ---------------------------------------------------------
    evidence = _build_evidence(consensus_signal)

    # --- risks ------------------------------------------------------------
    raw_risks = _safe_json_field(consensus_signal, "risk_factors", [])
    if isinstance(raw_risks, str):
        try:
            raw_risks = json.loads(raw_risks)
        except (json.JSONDecodeError, TypeError):
            raw_risks = [raw_risks] if raw_risks else []
    risks = raw_risks if isinstance(raw_risks, list) else []

    # --- debate -----------------------------------------------------------
    swarm_brief = None
    try:
        from swarm_synthesizer import synthesize_cycle
        swarm_brief, _ = synthesize_cycle()
    except Exception:
        pass

    forum_threads = _load_forum_threads(ticker)
    debate = build_debate_section(ticker, swarm_brief=swarm_brief, forum_threads=forum_threads)

    # --- portfolio --------------------------------------------------------
    if portfolio is None:
        portfolio = get_portfolio_context(ticker)

    # --- confidence breakdown ---------------------------------------------
    weights = consensus_signal.get("weights_applied")
    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except (json.JSONDecodeError, TypeError):
            weights = {}
    confidence_breakdown = weights if isinstance(weights, dict) else {}

    # --- valuation --------------------------------------------------------
    valuation = _safe_json_field(consensus_signal, "valuation_summary", None)

    # --- catalysts --------------------------------------------------------
    catalysts = _extract_catalysts(consensus_signal, ticker)

    # --- trade plan -------------------------------------------------------
    trade_plan = consensus_signal.get("trade_plan")
    if isinstance(trade_plan, str):
        try:
            trade_plan = json.loads(trade_plan)
        except (json.JSONDecodeError, TypeError):
            trade_plan = None

    # --- freshness --------------------------------------------------------
    freshness = compute_signal_freshness(consensus_signal.get("timestamp", ""))

    return {
        "header": header,
        "thesis": thesis,
        "evidence": evidence,
        "risks": risks,
        "debate": debate,
        "portfolio": portfolio,
        "confidence_breakdown": confidence_breakdown,
        "valuation": valuation,
        "catalysts": catalysts,
        "trade_plan": trade_plan,
        "freshness": freshness,
    }


def compute_signal_freshness(signal_timestamp: str, trader_profile: dict = None) -> str:
    """
    Classify signal age relative to the trader's profile.

    Args:
        signal_timestamp: ISO-format timestamp of the signal.
        trader_profile: Optional trader profile dict (as returned by
            ``trader_profile.load_profile``).  When *None* the function
            tries to import and load the profile; falls back to
            ``swing_multi_week`` thresholds.

    Returns:
        One of ``"fresh"``, ``"aging"``, or ``"stale"``.
    """
    age_minutes = _compute_age_minutes(signal_timestamp)
    if age_minutes is None:
        return "stale"

    # Determine trading style
    style = _DEFAULT_STYLE
    if trader_profile is None:
        try:
            from trader_profile import load_profile
            trader_profile = load_profile()
        except Exception:
            trader_profile = {}

    if trader_profile:
        ts = trader_profile.get("trading_style", "")
        horizon = trader_profile.get("swing_horizon", "")
        if ts == "day_trader":
            style = "day_trader"
        elif ts == "value_investor":
            style = "value_investor"
        elif ts == "swing_trader" and horizon == "single_week":
            style = "swing_single_week"
        elif ts == "swing_trader":
            style = "swing_multi_week"

    thresholds = _FRESHNESS_THRESHOLDS.get(style, _FRESHNESS_THRESHOLDS[_DEFAULT_STYLE])

    if age_minutes < thresholds["fresh"]:
        return "fresh"
    elif age_minutes < thresholds["aging"]:
        return "aging"
    return "stale"


def get_portfolio_context(ticker: str) -> dict:
    """
    Gather portfolio context for *ticker*.

    Tries to import ``portfolio_manager`` and queries:
    - Current position (shares, avg_price, unrealized PnL approximation)
    - Sector exposure (% of portfolio in this ticker's sector)
    - Conflicting signals (opposite-direction consensus signals on same ticker)

    Returns:
        Dict with keys: has_position, shares, avg_price, pct_of_portfolio,
        sector_exposure, conflicts (list), concentration_warning (bool).
    """
    result = {
        "has_position": False,
        "shares": 0,
        "avg_price": 0.0,
        "pct_of_portfolio": 0.0,
        "sector_exposure": 0.0,
        "conflicts": [],
        "concentration_warning": False,
    }

    # -- position data -----------------------------------------------------
    try:
        import portfolio_manager

        summary = portfolio_manager.get_portfolio_summary()
        positions = summary.get("positions", [])
        cash = summary.get("cash", 0.0)

        total_invested = sum(p["quantity"] * p["avg_price"] for p in positions)
        total_value = cash + total_invested

        target = next((p for p in positions if p["ticker"] == ticker.upper()), None)

        if target:
            result["has_position"] = True
            result["shares"] = target.get("quantity", 0)
            result["avg_price"] = target.get("avg_price", 0.0)

            if total_value > 0:
                position_value = target["quantity"] * target["avg_price"]
                result["pct_of_portfolio"] = round(position_value / total_value * 100, 2)

        # Sector exposure approximation: count positions sharing the same
        # sector as *ticker*.  We look up the sector stored in the most recent
        # consensus signal for each held ticker.
        ticker_sector = _lookup_sector(ticker)
        if ticker_sector and total_value > 0:
            sector_value = 0.0
            for p in positions:
                pos_sector = _lookup_sector(p["ticker"])
                if pos_sector and pos_sector == ticker_sector:
                    sector_value += p["quantity"] * p["avg_price"]
            result["sector_exposure"] = round(sector_value / total_value * 100, 2)

        # Concentration warning: any single position > 20 % of portfolio
        if result["pct_of_portfolio"] > 20:
            result["concentration_warning"] = True

    except ImportError:
        logger.debug("portfolio_manager not available; skipping position data.")
    except Exception as exc:
        logger.warning("Error fetching portfolio context: %s", exc)

    # -- conflicting signals -----------------------------------------------
    try:
        conflicts = _find_conflicting_signals(ticker)
        result["conflicts"] = conflicts
    except Exception as exc:
        logger.warning("Error checking conflicting signals: %s", exc)

    return result


def build_debate_section(
    ticker: str,
    swarm_brief: dict = None,
    forum_threads: list = None,
) -> dict:
    """
    Aggregate debate data from the swarm synthesizer and forum threads.

    Args:
        ticker: The ticker symbol to focus on.
        swarm_brief: Optional SwarmBrief dict from ``swarm_synthesizer.synthesize_cycle``.
        forum_threads: Optional list of forum thread dicts for the ticker.

    Returns:
        Dict with: bull_case, bear_case, archetype_breakdown, status, intensity.
    """
    bull_case = ""
    bear_case = ""
    archetype_breakdown = {}
    status = "resolved"
    intensity = "low"

    # -- extract from swarm brief ------------------------------------------
    if swarm_brief and isinstance(swarm_brief, dict):
        top_theses = swarm_brief.get("top_theses", [])
        for thesis in top_theses:
            if thesis.get("ticker") != ticker:
                continue

            direction = thesis.get("direction", "")
            top_arg = thesis.get("top_argument", "")
            challenge = thesis.get("strongest_challenge", "")

            if direction == "BULLISH":
                bull_case = top_arg
                bear_case = challenge
            elif direction == "BEARISH":
                bear_case = top_arg
                bull_case = challenge
            else:
                bull_case = top_arg
                bear_case = challenge

            archetype_breakdown = thesis.get("archetype_breakdown", {})

            # Determine status from consensus strength
            strength = thesis.get("consensus_strength", 0)
            if strength > 0.8:
                status = "resolved"
            elif strength > 0.5:
                status = "active"
            else:
                status = "contentious"

            break  # first matching ticker

        # Check active debates for additional context
        active_debates = swarm_brief.get("active_debates", [])
        for debate in active_debates:
            if debate.get("ticker") != ticker:
                continue
            # An active debate overrides resolved status
            if status == "resolved":
                status = "active"
            # Use debate content if we don't already have cases
            if not bull_case:
                bull_case = debate.get("bull_summary", "")
            if not bear_case:
                bear_case = debate.get("bear_summary", "")
            break

    # -- augment from forum threads ----------------------------------------
    if forum_threads and isinstance(forum_threads, list):
        support_count = 0
        challenge_count = 0
        for thread in forum_threads:
            position = thread.get("position", "").upper()
            if position in ("SUPPORT", "BULLISH"):
                support_count += 1
                if not bull_case:
                    bull_case = thread.get("content", "")[:300]
            elif position in ("CHALLENGE", "BEARISH"):
                challenge_count += 1
                if not bear_case:
                    bear_case = thread.get("content", "")[:300]

        total_forum = support_count + challenge_count
        if total_forum >= 5:
            intensity = "high"
        elif total_forum >= 2:
            intensity = "moderate"
        else:
            intensity = "low"

        # Forum contention can escalate status
        if total_forum >= 3:
            ratio = max(support_count, challenge_count) / total_forum
            if ratio < 0.6:
                status = "contentious"
            elif status == "resolved" and ratio < 0.8:
                status = "active"

    # If we still have no intensity from forum, derive from swarm data
    if intensity == "low" and swarm_brief:
        swarm_size = swarm_brief.get("swarm_size", 0)
        if swarm_size >= 10:
            intensity = "moderate"
        if swarm_size >= 20:
            intensity = "high"

    return {
        "bull_case": bull_case,
        "bear_case": bear_case,
        "archetype_breakdown": archetype_breakdown,
        "status": status,
        "intensity": intensity,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_age_minutes(timestamp_str: str) -> float | None:
    """Parse an ISO timestamp and return age in minutes, or None."""
    if not timestamp_str:
        return None
    try:
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        return delta.total_seconds() / 60.0
    except (ValueError, TypeError):
        return None


def _safe_json_field(signal: dict, key: str, default):
    """
    Return ``signal[key]`` parsed from JSON if it is a string,
    otherwise return it directly.  Falls back to *default*.
    """
    value = signal.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


def _build_evidence(consensus_signal: dict) -> list:
    """
    Break down evidence by contributing agent type.

    Uses the ``evidence_summary`` field if stored, otherwise synthesises
    a minimal list from ``contributing_agents`` and the signal direction.
    """
    stored = _safe_json_field(consensus_signal, "evidence_summary", None)
    if stored and isinstance(stored, list):
        return stored

    evidence = []
    agents = consensus_signal.get("contributing_agents", [])
    if isinstance(agents, str):
        try:
            agents = json.loads(agents)
        except (json.JSONDecodeError, TypeError):
            agents = []

    direction = consensus_signal.get("direction", "NEUTRAL")
    confidence = consensus_signal.get("adjusted_confidence", consensus_signal.get("raw_confidence", 50))

    # Map agent types to human-readable categories
    agent_categories = {
        "grok": "News & Event Analysis",
        "technical": "Technical Analysis",
        "thesis": "Thesis Research",
        "scout": "Market Scouting",
        "forum": "Forum Consensus",
        "swarm": "Swarm Intelligence",
        "valuation": "Valuation Analysis",
        "catalyst": "Catalyst Calendar",
        "macro": "Macro Context",
        "sentiment": "Sentiment Analysis",
    }

    for agent_type in agents:
        agent_lower = agent_type.lower() if isinstance(agent_type, str) else str(agent_type).lower()
        category = agent_categories.get(agent_lower, agent_type)
        evidence.append({
            "category": category,
            "text": f"{category} contributed to {direction} signal",
            "direction": direction,
            "confidence": confidence,
        })

    return evidence


def _extract_catalysts(consensus_signal: dict, ticker: str) -> list:
    """
    Pull catalyst data from the signal or from the catalyst calendar.
    """
    # Check if catalysts are embedded in the signal
    stored = _safe_json_field(consensus_signal, "catalysts", None)
    if stored and isinstance(stored, list):
        return stored

    # Try to fetch from catalyst_calendar
    try:
        from catalyst_calendar import get_signal_context
        time_horizon = consensus_signal.get("time_horizon_days", 5)
        ctx = get_signal_context(ticker, time_horizon)
        if ctx:
            catalysts = []
            if ctx.get("earnings_imminent"):
                catalysts.append({"type": "earnings", "description": "Earnings report imminent", "impact": "high"})
            if not ctx.get("trading_window_clear", True):
                catalysts.append({"type": "event", "description": "High-impact event within horizon", "impact": "high"})
            upcoming = ctx.get("upcoming_events", [])
            for evt in upcoming[:5]:
                catalysts.append({
                    "type": evt.get("type", "event"),
                    "description": evt.get("description", str(evt)),
                    "impact": evt.get("impact", "medium"),
                })
            return catalysts
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Catalyst extraction failed: %s", exc)

    return []


def _lookup_sector(ticker: str) -> str | None:
    """Look up the sector for *ticker* from the most recent consensus signal."""
    try:
        if not os.path.exists(KG_DB_PATH):
            return None
        conn = sqlite3.connect(KG_DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sector FROM consensus_signals WHERE ticker = ? AND sector IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
        conn.close()
        return row["sector"] if row else None
    except Exception:
        return None


def _find_conflicting_signals(ticker: str) -> list:
    """
    Check the consensus_signals table for recent signals on the same ticker
    with an opposite direction.
    """
    conflicts = []
    try:
        if not os.path.exists(KG_DB_PATH):
            return conflicts
        conn = sqlite3.connect(KG_DB_PATH)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT direction, consensus_score, timestamp, reasoning
            FROM consensus_signals
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (ticker.upper(),)).fetchall()
        conn.close()

        if len(rows) < 2:
            return conflicts

        latest_direction = rows[0]["direction"]
        for row in rows[1:]:
            if row["direction"] != latest_direction:
                conflicts.append({
                    "direction": row["direction"],
                    "consensus_score": row["consensus_score"],
                    "timestamp": row["timestamp"],
                    "summary": (row["reasoning"] or "")[:150],
                })
    except Exception as exc:
        logger.debug("Conflict check failed: %s", exc)

    return conflicts


def _load_forum_threads(ticker: str) -> list:
    """Try to load recent forum threads related to *ticker*."""
    try:
        if not os.path.exists(KG_DB_PATH):
            return []
        conn = sqlite3.connect(KG_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM forum_threads
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """, (ticker.upper(),)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []
