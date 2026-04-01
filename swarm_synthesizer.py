"""
FlowTrace Swarm Synthesizer

Reads the swarm simulation state and produces two outputs:
1. SwarmBrief — structured summary for the Supervisor agent (path c)
2. SwarmConsensusSignals — weighted signals for ConsensusAgent (path b)
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")


def _get_conn(db_path: str = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or KG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Core Synthesis
# ---------------------------------------------------------------------------

def synthesize_cycle(db_path: str = None, cycle_number: int = 0) -> tuple:
    """
    Analyze a completed simulation cycle and produce:
    - swarm_brief (dict): for injection into Supervisor context
    - swarm_signals (list[dict]): for ConsensusAgent signal pipeline

    Returns (swarm_brief, swarm_signals).
    """
    db = db_path or KG_DB_PATH
    conn = _get_conn(db)

    # Determine round range for this cycle
    if cycle_number > 0:
        rows = conn.execute("""
            SELECT MIN(round_number) as min_r, MAX(round_number) as max_r
            FROM swarm_rounds WHERE cycle_number = ?
        """, (cycle_number,)).fetchone()
    else:
        rows = conn.execute("""
            SELECT MIN(round_number) as min_r, MAX(round_number) as max_r
            FROM swarm_rounds
        """).fetchone()

    if not rows or rows["min_r"] is None:
        conn.close()
        return _empty_brief(), []

    min_round = rows["min_r"]
    max_round = rows["max_r"]
    rounds_analyzed = max_round - min_round + 1

    # Get swarm size
    agent_count = conn.execute(
        "SELECT COUNT(*) FROM swarm_agents WHERE is_active = 1"
    ).fetchone()[0]

    # Aggregate per-ticker consensus
    ticker_data = _aggregate_ticker_consensus(conn, min_round, max_round)

    # Get top theses (highest consensus strength)
    top_theses = _extract_top_theses(conn, ticker_data, min_round, max_round)

    # Get active debates (contested tickers)
    active_debates = _extract_active_debates(conn, ticker_data, min_round, max_round)

    # Get swarm alerts (anomalies)
    alerts = _detect_anomalies(ticker_data, agent_count)

    # Get performance context
    performance = _get_performance_context(conn)

    # Get archetype breakdown
    archetype_perf = _get_archetype_performance(conn)

    conn.close()

    # Build SwarmBrief
    swarm_brief = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_number": cycle_number,
        "rounds_analyzed": rounds_analyzed,
        "swarm_size": agent_count,
        "top_theses": top_theses[:5],
        "active_debates": active_debates[:5],
        "swarm_alerts": alerts[:5],
        "performance_context": performance,
        "archetype_performance": archetype_perf,
    }

    # Build SwarmConsensusSignals (one per ticker with strong consensus)
    swarm_signals = _build_consensus_signals(ticker_data, rounds_analyzed, agent_count)

    logger.info(
        f"[Synthesizer] Cycle {cycle_number}: {len(top_theses)} theses, "
        f"{len(active_debates)} debates, {len(swarm_signals)} signals"
    )

    return swarm_brief, swarm_signals


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _empty_brief() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycle_number": 0,
        "rounds_analyzed": 0,
        "swarm_size": 0,
        "top_theses": [],
        "active_debates": [],
        "swarm_alerts": [],
        "performance_context": {},
        "archetype_performance": {},
    }


def _aggregate_ticker_consensus(conn, min_round, max_round) -> dict:
    """Aggregate direction votes and confidence per ticker across the cycle."""
    rows = conn.execute("""
        SELECT ticker, direction, COUNT(*) as cnt, AVG(confidence) as avg_conf,
               MAX(confidence) as max_conf
        FROM swarm_posts
        WHERE round_number BETWEEN ? AND ?
          AND channel IN ('theses', 'challenges')
          AND ticker IS NOT NULL
        GROUP BY ticker, direction
    """, (min_round, max_round)).fetchall()

    ticker_data = {}
    for r in rows:
        ticker = r["ticker"]
        if ticker not in ticker_data:
            ticker_data[ticker] = {
                "votes": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
                "avg_conf": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
                "max_conf": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
                "total_votes": 0,
            }
        ticker_data[ticker]["votes"][r["direction"]] = r["cnt"]
        ticker_data[ticker]["avg_conf"][r["direction"]] = round(r["avg_conf"], 3)
        ticker_data[ticker]["max_conf"][r["direction"]] = round(r["max_conf"], 3)
        ticker_data[ticker]["total_votes"] += r["cnt"]

    # Calculate consensus strength per ticker
    for ticker, data in ticker_data.items():
        total = data["total_votes"]
        if total > 0:
            dominant = max(data["votes"], key=data["votes"].get)
            data["dominant_direction"] = dominant
            data["consensus_strength"] = data["votes"][dominant] / total
        else:
            data["dominant_direction"] = "NEUTRAL"
            data["consensus_strength"] = 0

    return ticker_data


def _extract_top_theses(conn, ticker_data, min_round, max_round) -> list:
    """Extract the strongest thesis per ticker."""
    theses = []

    for ticker, data in ticker_data.items():
        if data["consensus_strength"] < 0.5:
            continue

        dominant = data["dominant_direction"]
        opposite = "BEARISH" if dominant == "BULLISH" else "BULLISH"

        # Get the best thesis content for this ticker
        top_post = conn.execute("""
            SELECT content FROM swarm_posts
            WHERE ticker = ? AND direction = ? AND channel = 'theses'
              AND round_number BETWEEN ? AND ?
            ORDER BY confidence DESC LIMIT 1
        """, (ticker, dominant, min_round, max_round)).fetchone()

        # Get the strongest challenge
        top_challenge = conn.execute("""
            SELECT content FROM swarm_posts
            WHERE ticker = ? AND direction = ? AND channel = 'challenges'
              AND round_number BETWEEN ? AND ?
            ORDER BY confidence DESC LIMIT 1
        """, (ticker, opposite, min_round, max_round)).fetchone()

        # Get archetype breakdown for this ticker
        arch_rows = conn.execute("""
            SELECT sa.archetype, sp.direction, COUNT(*) as cnt
            FROM swarm_posts sp
            JOIN swarm_agents sa ON sp.agent_id = sa.agent_id
            WHERE sp.ticker = ? AND sp.round_number BETWEEN ? AND ?
              AND sp.channel IN ('theses', 'challenges')
            GROUP BY sa.archetype, sp.direction
        """, (ticker, min_round, max_round)).fetchall()

        archetype_breakdown = {}
        for r in arch_rows:
            arch = r["archetype"]
            if arch not in archetype_breakdown:
                archetype_breakdown[arch] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
            archetype_breakdown[arch][r["direction"]] = r["cnt"]

        # Simplify: dominant direction per archetype
        arch_simple = {}
        for arch, dirs in archetype_breakdown.items():
            arch_simple[arch] = max(dirs, key=dirs.get)

        theses.append({
            "ticker": ticker,
            "direction": dominant,
            "consensus_strength": round(data["consensus_strength"], 3),
            "avg_confidence": round(data["avg_conf"].get(dominant, 0), 3),
            "top_argument": top_post["content"][:300] if top_post else "N/A",
            "strongest_challenge": top_challenge["content"][:300] if top_challenge else "N/A",
            "debate_resolved": data["consensus_strength"] > 0.8,
            "archetype_breakdown": arch_simple,
        })

    theses.sort(key=lambda t: t["consensus_strength"], reverse=True)
    return theses


def _extract_active_debates(conn, ticker_data, min_round, max_round) -> list:
    """Find tickers where the swarm is deeply divided."""
    debates = []

    for ticker, data in ticker_data.items():
        bull = data["votes"]["BULLISH"]
        bear = data["votes"]["BEARISH"]
        neutral = data["votes"]["NEUTRAL"]
        total = data["total_votes"]

        if total < 3:
            continue

        # A "debate" is where no direction has >70% consensus
        if data["consensus_strength"] > 0.70:
            continue

        # Get key arguments
        bull_arg = conn.execute("""
            SELECT content FROM swarm_posts
            WHERE ticker = ? AND direction = 'BULLISH' AND channel IN ('theses', 'challenges')
              AND round_number BETWEEN ? AND ?
            ORDER BY confidence DESC LIMIT 1
        """, (ticker, min_round, max_round)).fetchone()

        bear_arg = conn.execute("""
            SELECT content FROM swarm_posts
            WHERE ticker = ? AND direction = 'BEARISH' AND channel IN ('theses', 'challenges')
              AND round_number BETWEEN ? AND ?
            ORDER BY confidence DESC LIMIT 1
        """, (ticker, min_round, max_round)).fetchone()

        # Suggest investigation
        if bull > bear:
            rec = f"TechnicalAgent should confirm bullish setup for {ticker}"
        elif bear > bull:
            rec = f"FundamentalAgent should check for deteriorating metrics on {ticker}"
        else:
            rec = f"ResearchAgent should gather more data on {ticker} — swarm is evenly split"

        debates.append({
            "ticker": ticker,
            "bull_count": bull,
            "bear_count": bear,
            "neutral_count": neutral,
            "key_bull_argument": bull_arg["content"][:200] if bull_arg else "N/A",
            "key_bear_argument": bear_arg["content"][:200] if bear_arg else "N/A",
            "recommended_investigation": rec,
        })

    debates.sort(key=lambda d: d["bull_count"] + d["bear_count"], reverse=True)
    return debates


def _detect_anomalies(ticker_data: dict, agent_count: int) -> list:
    """Detect herding, unusual consensus, or anomalous patterns."""
    alerts = []

    for ticker, data in ticker_data.items():
        strength = data["consensus_strength"]
        dominant = data["dominant_direction"]
        total = data["total_votes"]

        if strength > 0.90 and total >= 5:
            alerts.append(
                f"Unusual consensus on {ticker} ({strength:.0%} {dominant}) "
                f"— possible herding, needs external validation"
            )

        if total > agent_count * 0.5:
            alerts.append(
                f"High activity on {ticker} ({total} posts) — "
                f"swarm is unusually focused, check for catalyst"
            )

    return alerts


def _get_performance_context(conn) -> dict:
    """Get swarm-wide performance metrics."""
    row = conn.execute("""
        SELECT
            SUM(lifetime_wins) as total_wins,
            SUM(lifetime_losses) as total_losses,
            SUM(lifetime_trades) as total_trades,
            AVG(win_rate) as avg_win_rate,
            AVG(reputation_score) as avg_reputation
        FROM swarm_agents WHERE is_active = 1
    """).fetchone()

    if not row or row["total_trades"] is None:
        return {"swarm_win_rate": 0, "total_trades": 0}

    total_trades = row["total_trades"] or 0
    total_wins = row["total_wins"] or 0

    return {
        "swarm_win_rate": round(total_wins / total_trades, 3) if total_trades > 0 else 0,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": row["total_losses"] or 0,
        "avg_reputation": round(row["avg_reputation"], 3) if row["avg_reputation"] else 0.5,
    }


def _get_archetype_performance(conn) -> dict:
    """Win rate per archetype."""
    rows = conn.execute("""
        SELECT archetype,
               SUM(lifetime_wins) as wins,
               SUM(lifetime_trades) as trades,
               AVG(reputation_score) as avg_rep
        FROM swarm_agents WHERE is_active = 1
        GROUP BY archetype
    """).fetchall()

    result = {}
    for r in rows:
        trades = r["trades"] or 0
        wins = r["wins"] or 0
        result[r["archetype"]] = {
            "win_rate": round(wins / trades, 3) if trades > 0 else 0,
            "trades": trades,
            "avg_reputation": round(r["avg_rep"], 3) if r["avg_rep"] else 0.5,
        }
    return result


def _build_consensus_signals(ticker_data: dict, rounds_analyzed: int, agent_count: int) -> list:
    """
    Build SwarmConsensusSignal dicts formatted for the ConsensusAgent pipeline.
    Only emit signals for tickers with meaningful consensus (strength >= 0.55).
    """
    signals = []

    for ticker, data in ticker_data.items():
        strength = data["consensus_strength"]
        dominant = data["dominant_direction"]
        total = data["total_votes"]

        if strength < 0.55 or total < 3:
            continue

        avg_conf = data["avg_conf"].get(dominant, 0.5)

        # Estimate expected move based on confidence and direction
        expected_move = avg_conf * 100 * (1.0 if dominant == "BULLISH" else -1.0 if dominant == "BEARISH" else 0)
        expected_move = round(expected_move * 0.05, 2)  # Scale to reasonable %

        # Build reasoning string
        votes = data["votes"]
        reasoning = (
            f"{strength:.0%} of {total} swarm agents are {dominant} "
            f"(bull: {votes['BULLISH']}, bear: {votes['BEARISH']}, neutral: {votes['NEUTRAL']}). "
            f"Average confidence: {avg_conf:.0%}."
        )

        signals.append({
            "source": "trading_agent_swarm",
            "ticker": ticker,
            "direction": dominant,
            "confidence": round(avg_conf * 100, 1),
            "expected_move_pct": expected_move,
            "time_horizon_days": 3,
            "consensus_strength": round(strength, 3),
            "contributing_agents": ["swarm_aggregate"],
            "reasoning": reasoning,
            "event_type": "SwarmConsensus",
            "metadata": {
                "swarm_size": agent_count,
                "rounds_analyzed": rounds_analyzed,
                "votes": votes,
                "debate_intensity": "high" if (total / max(1, agent_count)) > 0.3 else "moderate",
            },
        })

    signals.sort(key=lambda s: s["consensus_strength"], reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Formatting for Integration
# ---------------------------------------------------------------------------

def format_swarm_brief_for_supervisor(brief: dict) -> str:
    """
    Convert a SwarmBrief dict into natural language for injection
    into the Supervisor's system prompt.
    """
    lines = []
    lines.append(f"Swarm Size: {brief.get('swarm_size', 0)} agents | "
                 f"Rounds Analyzed: {brief.get('rounds_analyzed', 0)}")

    perf = brief.get("performance_context", {})
    if perf.get("total_trades", 0) > 0:
        lines.append(f"Swarm Win Rate: {perf.get('swarm_win_rate', 0):.0%} "
                     f"({perf.get('total_trades', 0)} trades)")

    theses = brief.get("top_theses", [])
    if theses:
        lines.append("\nTOP SWARM THESES:")
        for t in theses[:3]:
            lines.append(
                f"  {t['ticker']}: {t['direction']} "
                f"(consensus: {t['consensus_strength']:.0%}, "
                f"confidence: {t['avg_confidence']:.0%})"
            )
            lines.append(f"    Argument: {t['top_argument'][:150]}")
            if t.get("strongest_challenge") and t["strongest_challenge"] != "N/A":
                lines.append(f"    Challenge: {t['strongest_challenge'][:150]}")

    debates = brief.get("active_debates", [])
    if debates:
        lines.append("\nACTIVE DEBATES (contested tickers):")
        for d in debates[:3]:
            lines.append(
                f"  {d['ticker']}: Bull={d['bull_count']} vs Bear={d['bear_count']}"
            )
            lines.append(f"    Recommendation: {d['recommended_investigation']}")

    alerts = brief.get("swarm_alerts", [])
    if alerts:
        lines.append("\nSWARM ALERTS:")
        for a in alerts[:3]:
            lines.append(f"  - {a}")

    return "\n".join(lines)


def format_swarm_signals_for_consensus(signals: list) -> list:
    """
    Format swarm signals to match the structure expected by
    agent_consensus.process_raw_signals().
    """
    formatted = []
    for sig in signals:
        formatted.append({
            "source_ticker": sig["ticker"],
            "target_ticker": sig["ticker"],
            "direction": sig["direction"],
            "confidence": sig["confidence"],
            "expected_move_pct": sig["expected_move_pct"],
            "time_horizon_days": sig["time_horizon_days"],
            "event_type": sig.get("event_type", "SwarmConsensus"),
            "contributing_agents": sig.get("contributing_agents", ["swarm_aggregate"]),
            "reasoning": sig["reasoning"],
            "summary": f"Swarm consensus: {sig['direction']} on {sig['ticker']} ({sig['consensus_strength']:.0%})",
            "swarm_consensus": sig,  # Pass full signal for swarm multiplier
        })
    return formatted
