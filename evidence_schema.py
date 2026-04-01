"""
FlowTrace Structured Evidence Schema

Defines the standard evidence format that all agents produce, replacing
raw text strings. Enables downstream consumers (ConsensusAgent, swarm,
forum, recommendation card, learning system) to parse, weight, and cite
specific findings.

Each agent type has a defined key_data schema so structured data flows
consistently from ingestion to final trader output.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-Agent-Type Key Data Schemas (defines expected fields)
# ---------------------------------------------------------------------------

KEY_DATA_SCHEMAS = {
    "TechnicalAgent": [
        "RSI_14", "MACD_signal", "MACD_histogram", "SMA_50", "SMA_200",
        "price_vs_SMA50", "ADX", "support_level", "resistance_level",
        "pattern", "VWAP", "OBV_trend",
    ],
    "FundamentalAgent": [
        "trailing_pe", "forward_pe", "price_to_book", "price_to_sales",
        "gross_margin", "operating_margin", "net_margin", "roe", "roa",
        "current_ratio", "debt_to_equity", "free_cash_flow", "fcf_yield",
        "revenue_growth", "earnings_growth", "peg_ratio",
    ],
    "MacroAgent": [
        "fed_funds_rate", "treasury_10y", "treasury_2y", "yield_curve_spread",
        "vix", "vix_regime", "dxy", "sector_performance",
    ],
    "SentimentAgent": [
        "finbert_score", "finbert_label", "reddit_sentiment", "reddit_volume",
        "insider_net_activity", "insider_buy_count", "insider_sell_count",
    ],
    "EarningsAgent": [
        "days_to_earnings", "earnings_date", "eps_estimate", "revenue_estimate",
        "last_4_surprises", "avg_surprise_pct",
    ],
    "ShortInterestAgent": [
        "short_pct_float", "days_to_cover", "short_interest_trend",
        "short_interest_change_pct",
    ],
    "BusinessModelAgent": [
        "revenue_model_type", "top_segment", "top_segment_pct",
        "customer_concentration_pct", "top_customer",
        "moat_type", "moat_strength", "tam_bn", "tam_growth_rate",
        "market_share_pct", "business_quality_score",
    ],
    "ResearchAgent": [
        "key_finding", "data_source", "relevance_score",
    ],
    "ScoutAgent": [
        "web_traffic_trend", "app_ranking", "job_posting_trend",
        "google_trends_direction",
    ],
    "StrategyAgent": [
        "recommended_strategy", "strategy_legs", "max_risk", "max_reward",
        "breakeven", "probability_of_profit",
    ],
    "ValidationAgent": [
        "backtest_win_rate", "backtest_avg_return", "backtest_max_drawdown",
        "backtest_sample_size", "signal_type_tested",
    ],
    "NewsSentimentAgent": [
        "finbert_score", "finbert_label", "finbert_confidence",
    ],
    "RiskManagerAgent": [
        "current_exposure_pct", "sector_exposure_pct", "concentration_risk",
        "portfolio_beta", "var_95",
    ],
    "VolatilityAgent": [
        "vix_spot", "vix_3m", "vix_6m", "term_structure", "contango_ratio",
    ],
    "CorrelationMatrixAgent": [
        "correlation_vs_spy", "correlation_vs_sector", "rolling_avg",
        "decoupling_detected",
    ],
    "SeasonalityAgent": [
        "current_month_avg_return", "current_month_win_rate",
        "best_month", "worst_month",
    ],
    "SectorRotationAgent": [
        "sector_rank", "sector_momentum", "relative_strength_vs_spy",
        "recommended_allocation",
    ],
    "PeerComparisonAgent": [
        "pe_percentile", "margin_percentile", "growth_percentile",
        "overall_rank_in_peers",
    ],
}

# Agent types that are valid evidence producers
VALID_AGENT_TYPES = set(KEY_DATA_SCHEMAS.keys()) | {
    "ExecutionAgent", "PortfolioOptimizerAgent",
    "SupplyChainVisualizerAgent", "SECFilingsAgent",
    "NewsAggregatorAgent",
}


# ---------------------------------------------------------------------------
# Evidence Creation & Validation
# ---------------------------------------------------------------------------

def create_evidence(
    agent_type: str,
    ticker: str,
    findings: dict,
    data_sources: list = None,
) -> dict:
    """
    Create a validated evidence record.

    Args:
        agent_type: The agent that produced this evidence
        ticker: The ticker this evidence relates to
        findings: Dict with keys: summary, direction_bias, confidence, key_data, signals, warnings
        data_sources: List of data source names (e.g., ["yfinance", "technical_indicators.py"])

    Returns:
        Validated evidence dict
    """
    if agent_type not in VALID_AGENT_TYPES:
        logger.warning(f"Unknown agent type '{agent_type}' in evidence creation")

    # Validate findings is a dict
    if not isinstance(findings, dict):
        logger.warning(f"findings must be a dict, got {type(findings).__name__}")
        findings = {"summary": str(findings) if findings else "", "confidence": 0.5}

    # Ensure required fields
    evidence = {
        "agent_type": agent_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "findings": {
            "summary": findings.get("summary", ""),
            "direction_bias": findings.get("direction_bias", "NEUTRAL"),
            "confidence": max(0.0, min(1.0, findings.get("confidence", 0.5))),
            "key_data": findings.get("key_data", {}),
            "signals": findings.get("signals", []),
            "warnings": findings.get("warnings", []),
        },
        "data_sources": data_sources or [],
    }

    return evidence


def validate_evidence(evidence: dict) -> bool:
    """Check that an evidence record has the required structure."""
    required = ["agent_type", "ticker", "findings"]
    for field in required:
        if field not in evidence:
            return False

    findings = evidence.get("findings", {})
    if not isinstance(findings, dict):
        return False

    if "summary" not in findings:
        return False

    confidence = findings.get("confidence", 0)
    if not (0.0 <= confidence <= 1.0):
        return False

    return True


# ---------------------------------------------------------------------------
# Evidence Merging & Aggregation
# ---------------------------------------------------------------------------

def merge_evidence(evidence_list: list) -> dict:
    """
    Merge evidence from multiple agents into a unified view per ticker.

    Returns:
    {
        "ticker": "SWKS",
        "agent_count": 5,
        "direction_consensus": "BULLISH",
        "avg_confidence": 0.72,
        "by_agent": {
            "TechnicalAgent": { ... },
            "FundamentalAgent": { ... },
        },
        "all_signals": ["MACD bullish crossover", "P/E below sector avg", ...],
        "all_warnings": ["RSI approaching overbought", ...],
        "key_data_merged": {
            "RSI_14": 42.3,
            "trailing_pe": 18.2,
            ...
        },
    }
    """
    if not evidence_list:
        return {"ticker": None, "agent_count": 0, "by_agent": {}}

    ticker = evidence_list[0].get("ticker")
    by_agent = {}
    all_signals = []
    all_warnings = []
    key_data_merged = {}
    direction_votes = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    confidences = []

    for ev in evidence_list:
        if not validate_evidence(ev):
            continue

        agent_type = ev["agent_type"]
        findings = ev["findings"]

        by_agent[agent_type] = findings

        # Aggregate signals and warnings
        all_signals.extend(findings.get("signals", []))
        all_warnings.extend(findings.get("warnings", []))

        # Merge key_data
        key_data = findings.get("key_data", {})
        for k, v in key_data.items():
            if v is not None:
                key_data_merged[k] = v

        # Direction voting
        bias = findings.get("direction_bias", "NEUTRAL")
        if bias in direction_votes:
            direction_votes[bias] += 1

        confidences.append(findings.get("confidence", 0.5))

    # Determine consensus direction
    if direction_votes:
        direction_consensus = max(direction_votes, key=direction_votes.get)
    else:
        direction_consensus = "NEUTRAL"

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    return {
        "ticker": ticker,
        "agent_count": len(by_agent),
        "direction_consensus": direction_consensus,
        "direction_votes": direction_votes,
        "avg_confidence": round(avg_confidence, 3),
        "by_agent": by_agent,
        "all_signals": all_signals,
        "all_warnings": all_warnings,
        "key_data_merged": key_data_merged,
    }


# ---------------------------------------------------------------------------
# Evidence Formatting for Display & Communication
# ---------------------------------------------------------------------------

def evidence_to_debate_text(evidence: dict) -> str:
    """
    Convert structured evidence into natural language suitable for
    forum/swarm posts. Agents use this to cite specific data when
    publishing theses or challenging others.
    """
    findings = evidence.get("findings", {})
    agent_type = evidence.get("agent_type", "Agent")
    summary = findings.get("summary", "")
    key_data = findings.get("key_data", {})
    signals = findings.get("signals", [])
    confidence = findings.get("confidence", 0.5)

    parts = [f"[{agent_type}] {summary} (confidence: {confidence:.0%})"]

    # Add key data points
    data_points = []
    for k, v in list(key_data.items())[:5]:
        if v is not None:
            if isinstance(v, float):
                data_points.append(f"{k}={v:.2f}")
            else:
                data_points.append(f"{k}={v}")

    if data_points:
        parts.append(f"Data: {', '.join(data_points)}")

    if signals:
        parts.append(f"Signals: {'; '.join(signals[:3])}")

    return " | ".join(parts)


def evidence_summary_for_display(merged_evidence: dict) -> list:
    """
    Convert merged evidence into bullet points for the recommendation card.

    Returns list of dicts: [{category, text, direction, confidence}, ...]
    """
    bullets = []
    by_agent = merged_evidence.get("by_agent", {})

    # Map agent types to display categories
    category_map = {
        "MacroAgent": "Macro",
        "TechnicalAgent": "Technical",
        "FundamentalAgent": "Fundamental",
        "SentimentAgent": "Sentiment",
        "NewsSentimentAgent": "Sentiment",
        "EarningsAgent": "Earnings",
        "ShortInterestAgent": "Short Interest",
        "BusinessModelAgent": "Business Model",
        "VolatilityAgent": "Volatility",
        "ScoutAgent": "Alt Data",
        "ResearchAgent": "Research",
        "ValidationAgent": "Validation",
        "SeasonalityAgent": "Seasonality",
        "CorrelationMatrixAgent": "Correlation",
        "SectorRotationAgent": "Sector Rotation",
        "PeerComparisonAgent": "Peer Comparison",
        "StrategyAgent": "Options Strategy",
    }

    for agent_type, findings in by_agent.items():
        category = category_map.get(agent_type, agent_type.replace("Agent", ""))
        summary = findings.get("summary", "")
        confidence = findings.get("confidence", 0.5)
        direction = findings.get("direction_bias", "NEUTRAL")

        # Build concise bullet text
        key_data = findings.get("key_data", {})
        data_snippet = ""
        top_data = [(k, v) for k, v in list(key_data.items())[:3] if v is not None]
        if top_data:
            snippets = []
            for k, v in top_data:
                if isinstance(v, float):
                    snippets.append(f"{k}={v:.2f}")
                else:
                    snippets.append(f"{k}={v}")
            data_snippet = f" ({', '.join(snippets)})"

        bullets.append({
            "category": category,
            "text": f"{summary}{data_snippet}",
            "direction": direction,
            "confidence": confidence,
        })

    return bullets


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------

def evidence_to_json(evidence: dict) -> str:
    """Serialize evidence to JSON string for database storage."""
    return json.dumps(evidence, default=str)


def evidence_from_json(json_str: str) -> Optional[dict]:
    """Deserialize evidence from JSON string."""
    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def evidence_list_to_json(evidence_list: list) -> str:
    """Serialize a list of evidence records to JSON."""
    return json.dumps(evidence_list, default=str)


def evidence_list_from_json(json_str: str) -> list:
    """Deserialize a list of evidence records from JSON."""
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []
