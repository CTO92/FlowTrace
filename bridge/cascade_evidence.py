"""Converts TidalShift pathway signals into FlowTrace-compatible context.

Provides functions to:
- Build cascade_context section for signal renderer
- Assess pathway alignment with a FlowTrace trade signal
- Format pathway data for Streamlit display
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta


def build_cascade_context(adapter, ticker: str, sector: str = "") -> dict:
    """Build the cascade_context section for signal_renderer.

    Args:
        adapter: FlowTraceBridgeAdapter instance
        ticker: The ticker symbol
        sector: Optional sector name

    Returns a dict suitable for embedding in render_recommendation() output.
    """
    context = adapter.get_cascade_context_for_ticker(ticker, sector)

    if not context.get("available"):
        return {
            "available": False,
            "source": "TidalShift",
            "relevant_pathways": [],
            "cascade_warnings": [],
            "macro_regime": None,
        }

    # Format pathways for display
    formatted_pathways = []
    for pw in context.get("relevant_pathways", []):
        formatted_pathways.append({
            "pathway_id": pw.get("pathway_id"),
            "label": pw.get("label", "Unknown pathway"),
            "plausibility": pw.get("plausibility", 0),
            "composite": pw.get("composite", 0),
            "domains_involved": pw.get("domains_involved", []),
            "affected_sectors": pw.get("affected_sectors", []),
            "leading_indicators": [
                ind.get("name", ind.get("description", ""))
                for ind in pw.get("leading_indicators", [])[:5]
            ],
            "feedback_loops": pw.get("feedback_loops", []),
            "summary": pw.get("summary", "")[:500],
            "time_horizon": pw.get("time_horizon", "weeks"),
        })

    # Determine overall cascade risk from warnings
    warnings = context.get("cascade_warnings", [])
    if warnings:
        severities = [w.get("severity", "low") for w in warnings]
        if "critical" in severities:
            cascade_risk = "critical"
        elif "high" in severities:
            cascade_risk = "high"
        elif "medium" in severities or "moderate" in severities:
            cascade_risk = "medium"
        else:
            cascade_risk = "low"
    else:
        cascade_risk = "none"

    # Format warnings
    formatted_warnings = []
    for w in warnings[:3]:
        formatted_warnings.append({
            "type": w.get("warning_type", "unknown"),
            "severity": w.get("severity", "unknown"),
            "summary": w.get("summary", ""),
            "recommended_action": w.get("recommended_action", ""),
        })

    return {
        "available": True,
        "source": "TidalShift",
        "relevant_pathways": formatted_pathways,
        "cascade_risk": cascade_risk,
        "cascade_warnings": formatted_warnings,
        "leading_indicators": [
            ind.get("name", ind.get("description", ""))
            for ind in context.get("leading_indicators", [])[:5]
        ],
        "last_updated": context.get("last_updated"),
        "freshness": _compute_freshness(context.get("last_updated")),
    }


def _compute_freshness(last_updated: str | None) -> str:
    """Compute freshness of cascade context."""
    if not last_updated:
        return "stale"
    try:
        ts = datetime.fromisoformat(last_updated)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        if age < timedelta(minutes=30):
            return "fresh"
        elif age < timedelta(hours=4):
            return "aging"
        return "stale"
    except (ValueError, TypeError):
        return "stale"


def format_pathways_for_streamlit(cascade_cache: list[dict]) -> list[dict]:
    """Format pathway cache for Streamlit display."""
    formatted = []
    for pw in reversed(cascade_cache):
        formatted.append({
            "Label": pw.get("label", "Unknown"),
            "Plausibility": f"{pw.get('plausibility', 0):.0%}",
            "Sectors": ", ".join(pw.get("affected_sectors", [])),
            "Entities": ", ".join(pw.get("affected_entities", [])[:5]),
            "Domains": ", ".join(pw.get("domains_involved", [])),
            "Horizon": pw.get("time_horizon", "weeks"),
            "Received": pw.get("received_at", "")[:19],
        })
    return formatted


def format_warnings_for_streamlit(warnings: list[dict]) -> list[dict]:
    """Format cascade warnings for Streamlit display."""
    formatted = []
    for w in warnings:
        formatted.append({
            "Type": w.get("warning_type", "unknown"),
            "Severity": w.get("severity", "unknown"),
            "Summary": w.get("summary", "")[:100],
            "Action": w.get("recommended_action", "")[:100],
            "Received": w.get("received_at", "")[:19],
        })
    return formatted
