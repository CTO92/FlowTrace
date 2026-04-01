"""
FlowTrace Signal Intelligence

Cross-signal analysis module: clustering related signals, portfolio
cross-referencing, freshness classification, and contradiction detection.
"""

import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_kg_db():
    """Get connection to the knowledge graph database."""
    if not os.path.exists(KG_DB_PATH):
        return None
    conn = sqlite3.connect(KG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_related_tickers(ticker: str, conn) -> set:
    """Query knowledge graph relationships for tickers related to the given one."""
    related = set()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT target_ticker FROM relationships WHERE source_ticker = ?",
            (ticker,),
        )
        for row in cursor.fetchall():
            related.add(row["target_ticker"])
        cursor.execute(
            "SELECT source_ticker FROM relationships WHERE target_ticker = ?",
            (ticker,),
        )
        for row in cursor.fetchall():
            related.add(row["source_ticker"])
    except Exception as exc:
        logger.debug("KG relationship lookup failed for %s: %s", ticker, exc)
    return related


def _get_sector_for_ticker(ticker: str, conn) -> Optional[str]:
    """Look up sector from the companies table."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sector FROM companies WHERE ticker = ?", (ticker,)
        )
        row = cursor.fetchone()
        return row["sector"] if row else None
    except Exception:
        return None


def _parse_timestamp(ts) -> Optional[datetime]:
    """Parse an ISO-format timestamp string into a timezone-aware datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    try:
        # Handle common ISO formats including those with and without tz info
        ts_str = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 1. cluster_signals
# ---------------------------------------------------------------------------

def cluster_signals(signals: list, time_window_hours: int = 6) -> list:
    """Group related signals into thematic clusters.

    Clustering criteria (in priority order):
    1. Same sector (from the signal's ``sector`` field).
    2. Same event chain -- signals within *time_window_hours* that share
       related assets via the knowledge graph.
    3. Knowledge-graph connections (companies / relationships tables).

    Returns a list of cluster dicts::

        {
            "signals": [...],
            "theme": str,
            "tickers": [str, ...],
            "combined_confidence": float,
            "dominant_direction": str,
        }

    If clustering is not possible every signal is returned as its own cluster.
    """
    if not signals:
        return []

    conn = _get_kg_db()

    # Build a lookup of related tickers per signal ticker from the KG
    kg_relations: dict[str, set] = {}
    if conn:
        for sig in signals:
            ticker = sig.get("ticker", "")
            if ticker and ticker not in kg_relations:
                kg_relations[ticker] = _get_related_tickers(ticker, conn)

    # Union-Find helpers for merging clusters
    parent: list[int] = list(range(len(signals)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Pairwise comparison
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            si, sj = signals[i], signals[j]

            # Criterion 1: same sector
            sec_i = si.get("sector") or (
                _get_sector_for_ticker(si.get("ticker", ""), conn) if conn else None
            )
            sec_j = sj.get("sector") or (
                _get_sector_for_ticker(sj.get("ticker", ""), conn) if conn else None
            )
            if sec_i and sec_j and sec_i == sec_j:
                union(i, j)
                continue

            # Criterion 2: event chain -- within time window and shared related assets
            ts_i = _parse_timestamp(si.get("timestamp"))
            ts_j = _parse_timestamp(sj.get("timestamp"))
            if ts_i and ts_j:
                delta = abs((ts_i - ts_j).total_seconds())
                if delta <= time_window_hours * 3600:
                    ti = si.get("ticker", "")
                    tj = sj.get("ticker", "")
                    # Direct ticker match
                    if ti and tj and ti == tj:
                        union(i, j)
                        continue
                    # Related via KG
                    related_i = kg_relations.get(ti, set())
                    related_j = kg_relations.get(tj, set())
                    if (tj in related_i) or (ti in related_j) or (related_i & related_j):
                        union(i, j)
                        continue

            # Criterion 3: KG connections (regardless of time)
            ti = si.get("ticker", "")
            tj = sj.get("ticker", "")
            if ti and tj:
                related_i = kg_relations.get(ti, set())
                if tj in related_i:
                    union(i, j)

    if conn:
        conn.close()

    # Build cluster groups
    groups: dict[int, list[int]] = {}
    for idx in range(len(signals)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    clusters = []
    for indices in groups.values():
        cluster_signals_list = [signals[i] for i in indices]
        tickers = list({s.get("ticker", "") for s in cluster_signals_list if s.get("ticker")})

        # Combined confidence: average of individual confidences
        confidences = [
            s.get("confidence", s.get("consensus_score", 0.0))
            for s in cluster_signals_list
        ]
        combined = sum(confidences) / len(confidences) if confidences else 0.0

        # Dominant direction: majority vote
        direction_counts: dict[str, int] = {}
        for s in cluster_signals_list:
            d = (s.get("direction") or "NEUTRAL").upper()
            direction_counts[d] = direction_counts.get(d, 0) + 1
        dominant = max(direction_counts, key=direction_counts.get) if direction_counts else "NEUTRAL"

        # Theme: sector if uniform, otherwise generic description
        sectors = list({s.get("sector", "") for s in cluster_signals_list if s.get("sector")})
        if len(sectors) == 1:
            theme = sectors[0]
        elif tickers:
            theme = f"Multi-signal cluster: {', '.join(tickers[:5])}"
        else:
            theme = "Unclassified cluster"

        clusters.append({
            "signals": cluster_signals_list,
            "theme": theme,
            "tickers": tickers,
            "combined_confidence": round(combined, 4),
            "dominant_direction": dominant,
        })

    return clusters


# ---------------------------------------------------------------------------
# 2. check_portfolio_conflicts
# ---------------------------------------------------------------------------

def check_portfolio_conflicts(ticker: str, direction: str) -> dict:
    """Check portfolio for existing positions, sector exposure, and contradictions.

    Returns::

        {
            "existing_position": {"shares": int, "avg_price": float, "unrealized_pnl": float} | None,
            "sector_exposure": float,
            "contradictions": [{"ticker": str, "direction": str, "consensus_score": float, "timestamp": str}, ...],
            "concentration_warning": bool,
        }
    """
    result = {
        "existing_position": None,
        "sector_exposure": 0.0,
        "contradictions": [],
        "concentration_warning": False,
    }

    ticker = ticker.upper()
    direction = direction.upper()

    # --- Portfolio position & concentration ---
    try:
        import portfolio_manager

        summary = portfolio_manager.get_portfolio_summary()
        cash = summary.get("cash", 0.0)
        positions = summary.get("positions", [])

        total_invested = sum(p["quantity"] * p["avg_price"] for p in positions)
        total_value = cash + total_invested

        # Existing position
        for pos in positions:
            if pos["ticker"] == ticker:
                shares = pos["quantity"]
                avg_price = pos["avg_price"]
                # Approximate unrealized PnL -- use avg_price as current (no live price)
                unrealized = 0.0
                result["existing_position"] = {
                    "shares": shares,
                    "avg_price": avg_price,
                    "unrealized_pnl": unrealized,
                }
                break

        # Sector exposure (approximate)
        kg_conn = _get_kg_db()
        target_sector = None
        if kg_conn:
            target_sector = _get_sector_for_ticker(ticker, kg_conn)

        if target_sector and total_value > 0:
            sector_value = 0.0
            for pos in positions:
                pos_sector = _get_sector_for_ticker(pos["ticker"], kg_conn) if kg_conn else None
                if pos_sector == target_sector:
                    sector_value += pos["quantity"] * pos["avg_price"]
            result["sector_exposure"] = round(sector_value / total_value * 100, 2)

        if kg_conn:
            kg_conn.close()

        # Concentration warning
        if total_value > 0:
            ticker_exposure = 0.0
            for pos in positions:
                if pos["ticker"] == ticker:
                    ticker_exposure = (pos["quantity"] * pos["avg_price"]) / total_value * 100
                    break
            if ticker_exposure > 10.0 or result["sector_exposure"] > 25.0:
                result["concentration_warning"] = True

    except ImportError:
        logger.debug("portfolio_manager not available; skipping position checks")
    except Exception as exc:
        logger.warning("Error checking portfolio conflicts: %s", exc)

    # --- Contradicting consensus signals ---
    try:
        conn = _get_kg_db()
        if conn:
            cursor = conn.cursor()
            opposite = "BEARISH" if direction == "BULLISH" else "BULLISH"
            try:
                cursor.execute(
                    """SELECT ticker, direction, consensus_score, timestamp
                       FROM consensus_signals
                       WHERE ticker = ? AND direction = ?
                       ORDER BY timestamp DESC
                       LIMIT 10""",
                    (ticker, opposite),
                )
                for row in cursor.fetchall():
                    result["contradictions"].append({
                        "ticker": row["ticker"],
                        "direction": row["direction"],
                        "consensus_score": row["consensus_score"],
                        "timestamp": row["timestamp"],
                    })
            except sqlite3.OperationalError:
                pass  # consensus_signals table may not exist yet
            conn.close()
    except Exception as exc:
        logger.debug("Error looking up contradictions: %s", exc)

    return result


# ---------------------------------------------------------------------------
# 3. classify_freshness
# ---------------------------------------------------------------------------

# Thresholds in minutes: (fresh_max, aging_max).  >= aging_max is stale.
_FRESHNESS_THRESHOLDS = {
    "day_trader":         (30,   120),
    "swing_single_week":  (240,  720),
    "swing_multi_week":   (720,  2880),
    "value_investor":     (2880, 10080),
}


def classify_freshness(signal_timestamp: str, trader_profile: dict = None) -> str:
    """Classify a signal as ``'fresh'``, ``'aging'``, or ``'stale'``.

    Thresholds depend on the trader's style:

    ============= ============= ============== ==============
    Style         fresh (min)   aging (min)    stale (min)
    ============= ============= ============== ==============
    day_trader       < 30         < 120          >= 120
    swing_single     < 240        < 720          >= 720
    swing_multi      < 720        < 2880         >= 2880
    value_investor   < 2880       < 10080        >= 10080
    ============= ============= ============== ==============

    *trader_profile* should contain a ``trading_style`` key.  If ``None`` or
    unrecognised, defaults to ``swing_multi_week``.
    """
    dt = _parse_timestamp(signal_timestamp)
    if dt is None:
        return "stale"

    now = datetime.now(timezone.utc)
    age_minutes = (now - dt).total_seconds() / 60.0

    style = "swing_multi_week"
    if trader_profile:
        raw_style = trader_profile.get("trading_style", "")
        # Normalise: the profile may store "swing_trader" with a sub-key
        if raw_style == "swing_trader":
            horizon = trader_profile.get("swing_horizon", "multi_week")
            style = f"swing_{horizon}" if horizon else "swing_multi_week"
        elif raw_style in _FRESHNESS_THRESHOLDS:
            style = raw_style

    fresh_max, aging_max = _FRESHNESS_THRESHOLDS.get(style, (720, 2880))

    if age_minutes < fresh_max:
        return "fresh"
    elif age_minutes < aging_max:
        return "aging"
    return "stale"


# ---------------------------------------------------------------------------
# 4. find_contradictions
# ---------------------------------------------------------------------------

def find_contradictions(signal: dict, recent_signals: list) -> list:
    """Find signals in *recent_signals* that contradict *signal*.

    A contradiction exists when:
    - Both signals share the same ticker.
    - Their directions are opposite (BULLISH vs BEARISH).
    - Both signals are still within their ``time_horizon`` (not expired).

    Returns a list of contradicting signal dicts from *recent_signals*.
    """
    if not signal or not recent_signals:
        return []

    ticker = (signal.get("ticker") or "").upper()
    direction = (signal.get("direction") or "").upper()
    if not ticker or not direction:
        return []

    opposite = "BEARISH" if direction == "BULLISH" else "BULLISH"
    if direction not in ("BULLISH", "BEARISH"):
        return []

    now = datetime.now(timezone.utc)

    def _is_within_horizon(sig: dict) -> bool:
        ts = _parse_timestamp(sig.get("timestamp"))
        if ts is None:
            return False
        horizon_days = sig.get("time_horizon_days", sig.get("time_horizon", 5))
        try:
            horizon_days = float(horizon_days)
        except (TypeError, ValueError):
            horizon_days = 5
        expiry = ts + timedelta(days=horizon_days)
        return now <= expiry

    # Check that the source signal itself is still within horizon
    if not _is_within_horizon(signal):
        return []

    contradictions = []
    for other in recent_signals:
        other_ticker = (other.get("ticker") or "").upper()
        other_direction = (other.get("direction") or "").upper()
        if other_ticker != ticker:
            continue
        if other_direction != opposite:
            continue
        if _is_within_horizon(other):
            contradictions.append(other)

    return contradictions
