"""
FlowTrace ConsensusAgent

Aggregates outputs from multiple research agents and forum debate
into a final weighted consensus signal for the trader. Applies
learning config weights to produce calibrated confidence scores.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

from learning_config_manager import load_config, get_agent_weight
from node_identity import get_node_id, generate_agent_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")

# ConsensusAgent identity for this session
_consensus_agent_id = None


def _get_agent_id() -> str:
    global _consensus_agent_id
    if _consensus_agent_id is None:
        _consensus_agent_id = generate_agent_id("ConsensusAgent")
    return _consensus_agent_id


def _get_db():
    """Get connection to the knowledge graph database."""
    if not os.path.exists(KG_DB_PATH):
        return None
    conn = sqlite3.connect(KG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_consensus_table(conn):
    """Create the consensus_signals table if it doesn't exist, and add expanded columns."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consensus_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            consensus_score REAL NOT NULL,
            raw_confidence REAL,
            adjusted_confidence REAL,
            expected_move_pct REAL,
            time_horizon_days INTEGER DEFAULT 5,
            contributing_agents TEXT,
            event_type TEXT,
            reasoning TEXT,
            source_signal_ids TEXT,
            node_id TEXT,
            agent_id TEXT,
            resolved_at TEXT,
            actual_return REAL,
            outcome TEXT
        )
    """)
    # Expanded columns (V0.3.1-alpha)
    new_columns = [
        ("risk_factors", "TEXT"),
        ("sector", "TEXT"),
        ("market_regime", "TEXT"),
        ("market_snapshot", "TEXT"),
        ("evidence_summary", "TEXT"),
        ("debate_summary", "TEXT"),
        ("portfolio_context", "TEXT"),
        ("valuation_summary", "TEXT"),
        ("trade_plan", "TEXT"),
    ]
    for col_name, col_type in new_columns:
        # Note: DDL column names cannot be parameterized in SQLite.
        # These values are hardcoded above (not from user input), so no injection risk.
        try:
            cursor.execute(f"ALTER TABLE consensus_signals ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


def calculate_consensus(
    ticker: str,
    direction: str,
    raw_confidence: float,
    expected_move_pct: float,
    event_type: str,
    contributing_agents: list,
    reasoning: str,
    time_horizon_days: int = 5,
    source_signal_ids: list = None,
    forum_support_count: int = 0,
    forum_challenge_count: int = 0,
    forum_agent_scores: dict = None,
    swarm_consensus: dict = None,
) -> dict:
    """
    Calculate a weighted consensus score from agent outputs and forum debate.

    Args:
        ticker: The ticker symbol
        direction: BULLISH or BEARISH
        raw_confidence: The original confidence from Grok analysis (0-100)
        expected_move_pct: Expected % move
        event_type: Type of event (Earnings, Contract, etc.)
        contributing_agents: List of agent types that contributed
        reasoning: Combined reasoning text
        time_horizon_days: 1-5 trading days
        source_signal_ids: IDs of signals that fed into this consensus
        forum_support_count: Number of forum agents supporting this thesis
        forum_challenge_count: Number of forum agents challenging this thesis
        forum_agent_scores: Dict of {agent_id: reputation_score} from forum participants
        swarm_consensus: Optional SwarmConsensusSignal dict from the Trading Agent Swarm

    Returns:
        Consensus signal dict with adjusted scores
    """
    config = load_config()

    # 1. Apply agent weights to raw confidence
    if contributing_agents:
        weight_count = 0
        weighted_sum = 0.0
        for agent_type in contributing_agents:
            weight = get_agent_weight(agent_type)
            weighted_sum += weight
            weight_count += 1

        agent_multiplier = weighted_sum / max(1, weight_count)
    else:
        agent_multiplier = 1.0

    # 2. Apply event type weight
    event_weight = config["event_type_weights"].get(event_type, 1.0)

    # 3. Apply sector adjustment using market context
    sector_multiplier = 1.0
    ctx = None
    try:
        from market_context import capture_market_context_sync, classify_regime
        ctx = capture_market_context_sync(ticker)

        if ctx:
            sector_change = ctx.get("sector", {}).get("change_pct", 0) or 0
            spy_change = ctx.get("indices", {}).get("SPY", {}).get("change_pct", 0) or 0
            relative = sector_change - spy_change
            # Sector outperforming market = tailwind for bullish signals
            if direction == "BULLISH" and relative > 0.5:
                sector_multiplier = 1.05
            elif direction == "BULLISH" and relative < -0.5:
                sector_multiplier = 0.92
            elif direction == "BEARISH" and relative < -0.5:
                sector_multiplier = 1.05
            elif direction == "BEARISH" and relative > 0.5:
                sector_multiplier = 0.92
    except ImportError:
        pass

    # 4. Apply regime adjustment using VIX
    regime_multiplier = 1.0
    try:
        if ctx:
            vix_data = ctx.get("vix", {})
            vix_level = vix_data.get("level") if isinstance(vix_data, dict) else vix_data
            if vix_level and isinstance(vix_level, (int, float)):
                regime = classify_regime(vix_level)
                if regime == "low_vix":
                    regime_multiplier = 1.05
                elif regime == "high_vix":
                    regime_multiplier = 0.85
    except Exception:
        pass

    # 5. Apply forum consensus adjustment
    forum_multiplier = 1.0
    if forum_support_count + forum_challenge_count > 0:
        # Net forum sentiment
        net_support = forum_support_count - forum_challenge_count
        total_forum = forum_support_count + forum_challenge_count

        # Weight by participant reputation if available
        if forum_agent_scores:
            avg_supporter_score = 0.5
            avg_challenger_score = 0.5
            # Higher reputation supporters boost confidence more
            scores = list(forum_agent_scores.values())
            if scores:
                avg_score = sum(scores) / len(scores)
                forum_multiplier = 0.9 + (avg_score * 0.2)  # range: 0.9 to 1.1

        # Net support/challenge ratio
        if total_forum >= 3:
            support_ratio = forum_support_count / total_forum
            if support_ratio > 0.7:
                forum_multiplier *= 1.1  # strong network support
            elif support_ratio < 0.3:
                forum_multiplier *= 0.85  # strong network challenge

    # 6. Apply swarm consensus adjustment
    swarm_multiplier = 1.0
    if swarm_consensus:
        # Ensure swarm_consensus is a dict (may arrive as JSON string from DB)
        if isinstance(swarm_consensus, str):
            try:
                swarm_consensus = json.loads(swarm_consensus)
            except (json.JSONDecodeError, TypeError):
                swarm_consensus = None
    if swarm_consensus and isinstance(swarm_consensus, dict) and swarm_consensus.get("ticker") == ticker:
        swarm_direction_match = (swarm_consensus.get("direction") == direction)
        swarm_strength = swarm_consensus.get("consensus_strength", 0.5)
        swarm_weight = get_agent_weight("swarm")

        if swarm_direction_match:
            # Swarm agrees — boost proportional to consensus strength and swarm weight
            swarm_multiplier = 1.0 + (swarm_strength - 0.5) * 0.2 * swarm_weight
        else:
            # Swarm disagrees — penalize proportional to consensus strength
            swarm_multiplier = 1.0 - (swarm_strength - 0.5) * 0.15 * swarm_weight

        swarm_multiplier = max(0.80, min(1.15, swarm_multiplier))

    # 7. Apply valuation multiplier
    valuation_multiplier = 1.0
    try:
        from valuation import synthesize_valuation
        val = synthesize_valuation(ticker)
        if val and "error" not in val:
            verdict = val.get("valuation_verdict", "fairly_valued")
            if verdict == "undervalued" and direction == "BULLISH":
                valuation_multiplier = 1.05
            elif verdict == "overvalued" and direction == "BULLISH":
                valuation_multiplier = 0.90
            elif verdict == "overvalued" and direction == "BEARISH":
                valuation_multiplier = 1.05
            elif verdict == "undervalued" and direction == "BEARISH":
                valuation_multiplier = 0.90
    except ImportError:
        pass

    # 8. Apply catalyst multiplier
    catalyst_multiplier = 1.0
    try:
        from catalyst_calendar import get_signal_context
        cat_ctx = get_signal_context(ticker, time_horizon_days)
        if cat_ctx:
            if not cat_ctx.get("trading_window_clear", True):
                catalyst_multiplier = 0.90  # Caution: high-impact event within horizon
            if cat_ctx.get("earnings_imminent") and direction == "BULLISH":
                catalyst_multiplier = max(catalyst_multiplier, 1.05)  # Earnings catalyst for bullish
    except ImportError:
        pass

    # 9. Calculate final adjusted confidence
    raw_normalized = raw_confidence / 100.0  # normalize to 0-1

    adjusted = (
        raw_normalized
        * agent_multiplier
        * event_weight
        * sector_multiplier
        * regime_multiplier
        * forum_multiplier
        * swarm_multiplier
        * valuation_multiplier
        * catalyst_multiplier
    )

    # Clamp to [0, 1]
    adjusted = max(0.0, min(1.0, adjusted))

    # Consensus score combines adjusted confidence with expected move magnitude
    move_factor = min(abs(expected_move_pct) / 10.0, 1.0)  # normalize large moves
    consensus_score = (adjusted * 0.7) + (move_factor * 0.3)
    consensus_score = max(0.0, min(1.0, consensus_score))

    signal = {
        "ticker": ticker,
        "direction": direction,
        "consensus_score": round(consensus_score, 4),
        "raw_confidence": raw_confidence,
        "adjusted_confidence": round(adjusted * 100, 2),
        "expected_move_pct": expected_move_pct,
        "time_horizon_days": time_horizon_days,
        "event_type": event_type,
        "contributing_agents": contributing_agents,
        "reasoning": reasoning,
        "source_signal_ids": source_signal_ids or [],
        "forum_support": forum_support_count,
        "forum_challenge": forum_challenge_count,
        "weights_applied": {
            "agent_multiplier": round(agent_multiplier, 3),
            "event_weight": round(event_weight, 3),
            "sector_multiplier": round(sector_multiplier, 3),
            "regime_multiplier": round(regime_multiplier, 3),
            "forum_multiplier": round(forum_multiplier, 3),
            "swarm_multiplier": round(swarm_multiplier, 3),
            "valuation_multiplier": round(valuation_multiplier, 3),
            "catalyst_multiplier": round(catalyst_multiplier, 3),
        },
        "node_id": get_node_id(),
        "agent_id": _get_agent_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return signal


def emit_signal(signal: dict) -> Optional[int]:
    """
    Save a consensus signal to the local database and return its ID.
    Only emits if the consensus score meets the minimum threshold.
    """
    config = load_config()
    min_score = config.get("min_signal_consensus", 0.65)

    if signal["consensus_score"] < min_score:
        logger.info(
            f"Signal for {signal['ticker']} below threshold "
            f"({signal['consensus_score']:.2f} < {min_score}). Skipping."
        )
        return None

    conn = _get_db()
    if not conn:
        logger.error("Knowledge graph database not found.")
        return None

    _ensure_consensus_table(conn)
    cursor = conn.cursor()

    # Generate trade plan for high-confidence signals (HIGH bug #8 fix)
    if not signal.get("trade_plan"):
        try:
            from trade_plan import generate_trade_plan
            plan = generate_trade_plan(
                consensus_signal=signal,
                trader_profile=None,  # Will load from disk
            )
            if plan and "error" not in plan:
                signal["trade_plan"] = plan
        except Exception as e:
            logger.debug(f"Trade plan generation skipped: {e}")

    # Populate evidence_summary if not already set (HIGH bug #9 fix)
    # Aggregate from source signals' agent_data if available
    if not signal.get("evidence_summary"):
        try:
            evidence_parts = []
            for agent_type in signal.get("contributing_agents", []):
                evidence_parts.append({
                    "agent_type": agent_type,
                    "direction_bias": signal["direction"],
                    "confidence": signal["raw_confidence"] / 100.0,
                })
            if evidence_parts:
                signal["evidence_summary"] = evidence_parts
        except Exception:
            pass

    # Build expanded fields for V0.3.1-alpha
    trade_plan_json = json.dumps(signal.get("trade_plan"), default=str) if signal.get("trade_plan") else None
    risk_factors_json = json.dumps(signal.get("risk_factors")) if signal.get("risk_factors") else None
    market_snapshot_json = json.dumps(signal.get("market_snapshot"), default=str) if signal.get("market_snapshot") else None
    evidence_json = json.dumps(signal.get("evidence_summary"), default=str) if signal.get("evidence_summary") else None
    debate_json = json.dumps(signal.get("debate_summary"), default=str) if signal.get("debate_summary") else None
    portfolio_json = json.dumps(signal.get("portfolio_context"), default=str) if signal.get("portfolio_context") else None
    valuation_json = json.dumps(signal.get("valuation_summary"), default=str) if signal.get("valuation_summary") else None

    # Determine market regime
    market_regime = None
    try:
        from market_context import capture_market_context_sync, classify_regime
        ctx = capture_market_context_sync(signal["ticker"])
        if ctx:
            vix_data = ctx.get("vix", {})
            vix_level = vix_data.get("level") if isinstance(vix_data, dict) else vix_data
            if vix_level and isinstance(vix_level, (int, float)):
                market_regime = classify_regime(vix_level)
    except ImportError:
        pass

    cursor.execute("""
        INSERT INTO consensus_signals (
            timestamp, ticker, direction, consensus_score,
            raw_confidence, adjusted_confidence, expected_move_pct,
            time_horizon_days, contributing_agents, event_type,
            reasoning, source_signal_ids, node_id, agent_id,
            risk_factors, sector, market_regime, market_snapshot,
            evidence_summary, debate_summary, portfolio_context, valuation_summary,
            trade_plan
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal["timestamp"],
        signal["ticker"],
        signal["direction"],
        signal["consensus_score"],
        signal["raw_confidence"],
        signal["adjusted_confidence"],
        signal["expected_move_pct"],
        signal["time_horizon_days"],
        json.dumps(signal["contributing_agents"]),
        signal["event_type"],
        signal["reasoning"],
        json.dumps(signal.get("source_signal_ids", [])),
        signal["node_id"],
        signal["agent_id"],
        risk_factors_json,
        signal.get("sector"),
        market_regime,
        market_snapshot_json,
        evidence_json,
        debate_json,
        portfolio_json,
        valuation_json,
        trade_plan_json,
    ))

    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()

    logger.info(
        f"Consensus signal emitted: {signal['ticker']} {signal['direction']} "
        f"score={signal['consensus_score']:.2f} (id={signal_id})"
    )

    return signal_id


def get_recent_consensus_signals(limit: int = 20) -> list:
    """Fetch the most recent consensus signals from the database."""
    conn = _get_db()
    if not conn:
        return []

    _ensure_consensus_table(conn)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM consensus_signals
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Parse JSON fields
    for row in rows:
        try:
            row["contributing_agents"] = json.loads(row.get("contributing_agents") or "[]")
        except (json.JSONDecodeError, TypeError):
            row["contributing_agents"] = []
        try:
            row["source_signal_ids"] = json.loads(row.get("source_signal_ids") or "[]")
        except (json.JSONDecodeError, TypeError):
            row["source_signal_ids"] = []

    return rows


def get_open_consensus_signals() -> list:
    """Get consensus signals that haven't been resolved yet."""
    conn = _get_db()
    if not conn:
        return []

    _ensure_consensus_table(conn)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM consensus_signals
        WHERE outcome IS NULL
        ORDER BY timestamp DESC
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def process_raw_signals(raw_signals: list) -> list:
    """
    Take a batch of raw signals from the ingestion pipeline and produce
    consensus signals by grouping by ticker and aggregating.

    raw_signals: list of dicts from the signals table
    Returns: list of emitted consensus signal dicts
    """
    if not raw_signals:
        return []

    # Group by target ticker
    by_ticker = {}
    for sig in raw_signals:
        ticker = sig.get("target_ticker")
        if not ticker:
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(sig)

    emitted = []

    for ticker, signals in by_ticker.items():
        # Average confidence and expected move
        confidences = [s.get("confidence", 0) for s in signals if s.get("confidence")]
        moves = [s.get("expected_move_pct", 0) for s in signals if s.get("expected_move_pct")]

        if not confidences:
            continue

        avg_confidence = sum(confidences) / len(confidences)
        avg_move = sum(moves) / len(moves) if moves else 0

        # Determine direction from average expected move
        direction = "BULLISH" if avg_move >= 0 else "BEARISH"

        # Collect event types and reasoning
        event_types = [s.get("event_type", "Unknown") for s in signals]
        primary_event = max(set(event_types), key=event_types.count) if event_types else "Unknown"

        reasoning_parts = [s.get("reasoning", "") for s in signals if s.get("reasoning")]
        combined_reasoning = " | ".join(reasoning_parts[:3])  # top 3

        # Which agents contributed
        contributing = set()
        for s in signals:
            # Try multiple possible field names
            if s.get("contributing_agents"):
                agents = s["contributing_agents"]
                if isinstance(agents, list):
                    contributing.update(agents)
                elif isinstance(agents, str):
                    contributing.add(agents)
            elif s.get("agent_name"):
                contributing.add(s["agent_name"])
            elif s.get("source") == "trading_agent_swarm":
                contributing.add("swarm_aggregate")
            else:
                contributing.add(f"AnalysisEngine_{s.get('event_type', 'Unknown')}")
        contributing = list(contributing) if contributing else ["AnalysisEngine"]

        # Check if any signal carries swarm consensus data
        swarm_data = None
        for s in signals:
            if s.get("swarm_consensus"):
                swarm_data = s["swarm_consensus"]
                break

        signal = calculate_consensus(
            ticker=ticker,
            direction=direction,
            raw_confidence=avg_confidence,
            expected_move_pct=avg_move,
            event_type=primary_event,
            contributing_agents=contributing,
            reasoning=combined_reasoning,
            swarm_consensus=swarm_data,
        )

        signal_id = emit_signal(signal)
        if signal_id:
            signal["id"] = signal_id
            emitted.append(signal)

    return emitted
