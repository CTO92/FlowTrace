"""
FlowTrace LearningAgent

The meta-cognitive agent that analyzes trade outcomes to identify
what works and adjust the system's behavior over time. It reads
resolved signals from the local database, calculates win rates
per agent/sector/event type, and updates learning_config.json.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import yfinance as yf

from learning_config_manager import load_config, save_config, update_agent_weight, update_performance
from node_identity import get_node_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")
PERFORMANCE_LOG_FILE = os.path.join(BASE_DIR, "agent_performance.log")


def _get_signals_db():
    """Connect to the knowledge graph database (signals table)."""
    if not os.path.exists(KG_DB_PATH):
        return None
    conn = sqlite3.connect(KG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_outcome_columns(conn):
    """Add outcome tracking columns to the signals table if they don't exist."""
    cursor = conn.cursor()
    for col, col_type in [
        ("resolved_at", "TEXT"),
        ("actual_return", "REAL"),
        ("outcome", "TEXT"),
        ("time_horizon_days", "INTEGER DEFAULT 5"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


def resolve_open_signals(lookback_days: int = 30) -> dict:
    """
    Check all unresolved signals and resolve those whose time horizon has elapsed.
    Fetches actual price data to determine WIN/LOSS/NEUTRAL.

    Returns summary: {resolved: int, wins: int, losses: int, neutral: int}
    """
    conn = _get_signals_db()
    if not conn:
        return {"resolved": 0, "wins": 0, "losses": 0, "neutral": 0}

    _ensure_outcome_columns(conn)
    cursor = conn.cursor()

    # Find unresolved signals within lookback window
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    cursor.execute("""
        SELECT rowid, timestamp, source_ticker, target_ticker, event_type,
               expected_move_pct, confidence, unified_score, time_horizon_days
        FROM signals
        WHERE outcome IS NULL AND timestamp >= ?
    """, (cutoff,))

    signals = [dict(row) for row in cursor.fetchall()]

    if not signals:
        conn.close()
        return {"resolved": 0, "wins": 0, "losses": 0, "neutral": 0}

    summary = {"resolved": 0, "wins": 0, "losses": 0, "neutral": 0}

    for signal in signals:
        try:
            signal_time = datetime.fromisoformat(str(signal["timestamp"]))
        except (ValueError, TypeError):
            continue

        horizon = signal.get("time_horizon_days") or 5
        resolution_date = signal_time + timedelta(days=horizon + 2)  # +2 for weekends

        if datetime.now() < resolution_date:
            continue  # not yet time to resolve

        ticker = signal["target_ticker"]
        if not ticker:
            continue

        # Fetch price data around the signal date
        try:
            start_date = (signal_time - timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = (signal_time + timedelta(days=horizon + 3)).strftime("%Y-%m-%d")

            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 2:
                continue

            # Use 'Close' or 'Adj Close'
            if "Adj Close" in data.columns:
                prices = data["Adj Close"]
            else:
                prices = data["Close"]

            # Handle MultiIndex columns from yfinance
            if hasattr(prices, 'columns'):
                prices = prices.iloc[:, 0] if len(prices.columns) == 1 else prices

            entry_price = float(prices.iloc[0])
            exit_price = float(prices.iloc[-1])

            if entry_price == 0:
                continue

            actual_return = (exit_price - entry_price) / entry_price * 100  # percentage

            expected_move = signal.get("expected_move_pct") or 0

            # Determine outcome
            if expected_move > 0:  # bullish thesis
                if actual_return > 0:
                    outcome = "WIN"
                elif actual_return > -abs(expected_move) * 0.25:
                    outcome = "NEUTRAL"
                else:
                    outcome = "LOSS"
            elif expected_move < 0:  # bearish thesis
                if actual_return < 0:
                    outcome = "WIN"
                elif actual_return < abs(expected_move) * 0.25:
                    outcome = "NEUTRAL"
                else:
                    outcome = "LOSS"
            else:
                outcome = "NEUTRAL"

            # Update signal record
            cursor.execute("""
                UPDATE signals SET resolved_at = ?, actual_return = ?, outcome = ?
                WHERE rowid = ?
            """, (
                datetime.now(timezone.utc).isoformat(),
                round(actual_return, 4),
                outcome,
                signal["rowid"],
            ))

            summary["resolved"] += 1
            summary[outcome.lower() + "s" if outcome != "NEUTRAL" else "neutral"] += 1

        except Exception as e:
            logger.warning(f"Failed to resolve signal for {ticker}: {e}")
            continue

    conn.commit()
    conn.close()
    return summary


def calculate_agent_performance() -> dict:
    """
    Parse the agent performance log to calculate per-agent success rates.
    Returns: {agent_type: {success: int, fail: int, avg_duration: float, fail_rate: float}}
    """
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return {}

    stats = {}
    try:
        with open(PERFORMANCE_LOG_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    name = entry["agent_name"]
                    if name not in stats:
                        stats[name] = {"success": 0, "fail": 0, "total_time": 0, "count": 0}

                    if entry["success"]:
                        stats[name]["success"] += 1
                    else:
                        stats[name]["fail"] += 1

                    stats[name]["total_time"] += entry.get("duration", 0)
                    stats[name]["count"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        return {}

    # Calculate derived metrics
    for name, s in stats.items():
        s["avg_duration"] = round(s["total_time"] / max(1, s["count"]), 2)
        s["fail_rate"] = round(s["fail"] / max(1, s["count"]), 4)

    return stats


def calculate_signal_stats() -> dict:
    """
    Analyze resolved signals to calculate win rates by various dimensions.
    Returns nested dict with stats per event_type, per sector, per time_horizon.
    """
    conn = _get_signals_db()
    if not conn:
        return {}

    _ensure_outcome_columns(conn)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_type, outcome, actual_return, confidence, time_horizon_days
        FROM signals
        WHERE outcome IS NOT NULL
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    if not rows:
        return {"total": 0}

    # Overall stats
    wins = sum(1 for r in rows if r["outcome"] == "WIN")
    losses = sum(1 for r in rows if r["outcome"] == "LOSS")
    neutrals = sum(1 for r in rows if r["outcome"] == "NEUTRAL")
    returns = [r["actual_return"] for r in rows if r["actual_return"] is not None]
    avg_return = float(np.mean(returns)) if returns else 0.0

    # By event type
    event_stats = {}
    for row in rows:
        et = row.get("event_type") or "Unknown"
        if et not in event_stats:
            event_stats[et] = {"wins": 0, "losses": 0, "neutral": 0, "returns": []}
        if row["outcome"] == "WIN":
            event_stats[et]["wins"] += 1
        elif row["outcome"] == "LOSS":
            event_stats[et]["losses"] += 1
        else:
            event_stats[et]["neutral"] += 1
        if row["actual_return"] is not None:
            event_stats[et]["returns"].append(row["actual_return"])

    for et, s in event_stats.items():
        total = s["wins"] + s["losses"]
        s["win_rate"] = round(s["wins"] / max(1, total), 4)
        s["avg_return"] = round(float(np.mean(s["returns"])) if s["returns"] else 0.0, 4)
        del s["returns"]

    return {
        "total": len(rows),
        "wins": wins,
        "losses": losses,
        "neutral": neutrals,
        "win_rate": round(wins / max(1, wins + losses), 4),
        "avg_return": round(avg_return, 4),
        "by_event_type": event_stats,
    }


def adjust_weights() -> dict:
    """
    The core learning function. Analyzes all available data and adjusts
    learning_config.json weights accordingly.

    Returns a summary of adjustments made.
    """
    config = load_config()
    adjustments = {}

    # 1. Adjust agent weights based on execution performance
    agent_perf = calculate_agent_performance()
    for agent_type, stats in agent_perf.items():
        if stats["count"] < 5:
            continue  # not enough data

        current_weight = config["agent_weights"].get(agent_type, 1.0)

        # Agents with high failure rates get penalized
        if stats["fail_rate"] > 0.3:
            new_weight = current_weight * 0.95  # 5% penalty
            adjustments[agent_type] = f"fail_rate={stats['fail_rate']:.1%}, weight {current_weight:.3f} -> {new_weight:.3f}"
            update_agent_weight(agent_type, new_weight)
        elif stats["fail_rate"] < 0.1 and stats["count"] >= 10:
            new_weight = current_weight * 1.02  # 2% bonus for reliable agents
            adjustments[agent_type] = f"fail_rate={stats['fail_rate']:.1%}, weight {current_weight:.3f} -> {new_weight:.3f}"
            update_agent_weight(agent_type, new_weight)

    # 2. Adjust event type weights based on signal outcomes
    signal_stats = calculate_signal_stats()
    event_stats = signal_stats.get("by_event_type", {})

    for event_type, stats in event_stats.items():
        total = stats["wins"] + stats["losses"]
        if total < 3:
            continue

        current_weight = config["event_type_weights"].get(event_type, 1.0)

        if stats["win_rate"] > 0.6:
            new_weight = min(1.5, current_weight * 1.05)
            config["event_type_weights"][event_type] = round(new_weight, 3)
            adjustments[f"event:{event_type}"] = f"win_rate={stats['win_rate']:.1%}, weight -> {new_weight:.3f}"
        elif stats["win_rate"] < 0.4:
            new_weight = max(0.5, current_weight * 0.95)
            config["event_type_weights"][event_type] = round(new_weight, 3)
            adjustments[f"event:{event_type}"] = f"win_rate={stats['win_rate']:.1%}, weight -> {new_weight:.3f}"

    # 3. Update overall performance stats
    if signal_stats.get("total", 0) > 0:
        update_performance(
            wins=signal_stats["wins"],
            losses=signal_stats["losses"],
            neutral=signal_stats["neutral"],
            avg_return=signal_stats["avg_return"],
        )

    # 4. Confidence calibration (Phase 4)
    calibration = calibrate_confidence()
    if calibration:
        adjustments["confidence_calibration"] = calibration

    # 5. Market regime detection (Phase 4)
    regime = detect_market_regime()
    if regime:
        config["regime_adjustments"] = regime["adjustments"]
        adjustments["regime"] = regime["current_regime"]

    # 6. Save updated config
    save_config(config)

    logger.info(f"Learning review complete. Adjustments: {len(adjustments)}")
    return adjustments


def get_learning_summary() -> dict:
    """
    Produce a comprehensive summary for the dashboard / weekly report.
    """
    config = load_config()
    signal_stats = calculate_signal_stats()
    agent_perf = calculate_agent_performance()

    # Identify best and worst performing agent types
    sorted_agents = sorted(
        config["agent_weights"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Market regime
    regime = detect_market_regime()

    # Prompt improvement suggestions
    prompt_suggestions = generate_prompt_improvement_suggestions()

    # Confidence calibration
    calibration = calibrate_confidence()

    return {
        "node_id": get_node_id(),
        "config_version": config["version"],
        "participation_intensity": config.get("participation_intensity", "medium"),
        "performance": config["performance_history"],
        "signal_stats": signal_stats,
        "agent_weights": config["agent_weights"],
        "top_agents": sorted_agents[:5],
        "bottom_agents": sorted_agents[-5:],
        "agent_execution_stats": agent_perf,
        "event_type_weights": config["event_type_weights"],
        "market_regime": regime,
        "confidence_calibration": calibration,
        "prompt_suggestions": prompt_suggestions,
        "updated_at": config.get("updated_at"),
    }


def calibrate_confidence() -> dict:
    """
    Phase 4: Confidence Calibration

    Analyzes whether stated confidence levels match actual win rates.
    If we say "80% confident" but only win 50% of the time at that level,
    we need to recalibrate downward.

    Returns calibration adjustments or empty dict if not enough data.
    """
    conn = _get_signals_db()
    if not conn:
        return {}

    _ensure_outcome_columns(conn)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT confidence, outcome FROM signals
        WHERE outcome IS NOT NULL AND confidence IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 20:
        return {}  # not enough data

    # Bucket signals by confidence ranges
    buckets = {
        "50-60": {"predicted": 55, "wins": 0, "total": 0},
        "60-70": {"predicted": 65, "wins": 0, "total": 0},
        "70-80": {"predicted": 75, "wins": 0, "total": 0},
        "80-90": {"predicted": 85, "wins": 0, "total": 0},
        "90-100": {"predicted": 95, "wins": 0, "total": 0},
    }

    for row in rows:
        conf = row[0]
        outcome = row[1]

        if 50 <= conf < 60:
            bucket = "50-60"
        elif 60 <= conf < 70:
            bucket = "60-70"
        elif 70 <= conf < 80:
            bucket = "70-80"
        elif 80 <= conf < 90:
            bucket = "80-90"
        elif conf >= 90:
            bucket = "90-100"
        else:
            continue

        buckets[bucket]["total"] += 1
        if outcome == "WIN":
            buckets[bucket]["wins"] += 1

    # Calculate calibration errors
    calibration = {}
    for bucket_name, data in buckets.items():
        if data["total"] < 5:
            continue

        actual_win_rate = data["wins"] / data["total"] * 100
        predicted = data["predicted"]
        error = predicted - actual_win_rate

        calibration[bucket_name] = {
            "predicted_confidence": predicted,
            "actual_win_rate": round(actual_win_rate, 1),
            "calibration_error": round(error, 1),
            "sample_size": data["total"],
        }

    return calibration


def detect_market_regime() -> dict:
    """
    Phase 4: Market Regime Detection

    Classifies the current market regime based on VIX levels
    and adjusts confidence multipliers accordingly.

    Returns: {current_regime: str, vix_level: float, adjustments: dict}
    """
    try:
        vix_data = yf.download("^VIX", period="5d", progress=False)
        if vix_data.empty:
            return {}

        if "Close" in vix_data.columns:
            vix_level = float(vix_data["Close"].iloc[-1])
        elif "Adj Close" in vix_data.columns:
            vix_level = float(vix_data["Adj Close"].iloc[-1])
        else:
            # Handle MultiIndex columns
            close_cols = [c for c in vix_data.columns if "Close" in str(c)]
            if close_cols:
                vix_level = float(vix_data[close_cols[0]].iloc[-1])
            else:
                return {}

        # Classify regime
        if vix_level < 15:
            regime = "low_vix"
            confidence_multiplier = 1.05  # calmer markets, slightly more confident
        elif vix_level < 25:
            regime = "normal_vix"
            confidence_multiplier = 1.0
        else:
            regime = "high_vix"
            confidence_multiplier = 0.85  # volatile markets, less confident

        return {
            "current_regime": regime,
            "vix_level": round(vix_level, 2),
            "adjustments": {
                "low_vix": {"confidence_multiplier": 1.05},
                "normal_vix": {"confidence_multiplier": 1.0},
                "high_vix": {"confidence_multiplier": 0.85},
            },
        }
    except Exception as e:
        logger.warning(f"Failed to detect market regime: {e}")
        return {}


def generate_prompt_improvement_suggestions() -> dict:
    """
    Phase 4: Agent Prompt Evolution

    Analyzes agent performance to suggest system prompt modifications
    for consistently underperforming agents. Extends prompt_optimizer.py.

    Returns: {agent_type: suggestion_text}
    """
    agent_perf = calculate_agent_performance()
    config = load_config()
    suggestions = {}

    for agent_type, stats in agent_perf.items():
        if stats["count"] < 10:
            continue

        weight = config["agent_weights"].get(agent_type, 1.0)

        # Agents with high failure rate AND low weight
        if stats["fail_rate"] > 0.3 and weight < 0.9:
            suggestions[agent_type] = (
                f"Agent '{agent_type}' has a {stats['fail_rate']:.0%} failure rate "
                f"across {stats['count']} runs (trust weight: {weight:.2f}). "
                f"Consider: (1) simplifying the system prompt, "
                f"(2) reducing the number of tools, "
                f"(3) adding explicit error handling instructions."
            )

        # Agents that are consistently slow
        if stats["avg_duration"] > 45:
            existing = suggestions.get(agent_type, "")
            suggestions[agent_type] = (
                existing +
                f" Agent '{agent_type}' averages {stats['avg_duration']:.0f}s per run. "
                f"Consider: breaking the task into smaller steps or caching repeated lookups."
            )

    return suggestions


async def update_network_trust() -> dict:
    """
    Phase 4: Cross-Node Learning

    Fetches leaderboard data from the forum and updates network_node_trust
    in learning_config.json so ConsensusAgent can weight remote node inputs.
    """
    from forum_config import is_forum_configured
    if not is_forum_configured():
        return {}

    try:
        from forum_client import get_forum_client
        client = get_forum_client()
        leaderboard = await client.get_leaderboard()

        if not leaderboard or not isinstance(leaderboard, list):
            return {}

        config = load_config()
        trust_updates = {}

        for entry in leaderboard:
            alias = entry.get("node_alias")
            reputation = entry.get("reputation", 0.5)
            if alias:
                config["network_node_trust"][alias] = round(reputation, 4)
                trust_updates[alias] = reputation

        save_config(config)
        logger.info(f"Updated trust scores for {len(trust_updates)} network nodes")
        return trust_updates

    except Exception as e:
        logger.warning(f"Failed to update network trust: {e}")
        return {}


async def run_learning_review() -> dict:
    """
    Full learning review cycle:
    1. Resolve any open signals that have matured
    2. Recalculate stats
    3. Adjust weights
    4. Update network trust (Phase 4)
    5. Return summary
    """
    logger.info("Starting learning review...")

    # Step 1: Resolve matured signals
    resolution = resolve_open_signals()
    logger.info(f"Resolved {resolution['resolved']} signals: "
                f"{resolution['wins']} wins, {resolution['losses']} losses, "
                f"{resolution['neutral']} neutral")

    # Step 2 & 3: Analyze and adjust
    adjustments = adjust_weights()

    # Step 4: Cross-node learning
    trust_updates = await update_network_trust()
    if trust_updates:
        adjustments["network_trust_updates"] = len(trust_updates)

    # Step 5: Summary
    summary = get_learning_summary()
    summary["latest_resolution"] = resolution
    summary["latest_adjustments"] = adjustments

    return summary
