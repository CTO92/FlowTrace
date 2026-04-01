"""
report_generator.py — Institutional-grade research report generator for FlowTrace.

LLM writes narrative sections; structured data renders numbers directly
(no hallucination risk on data).  Supports HTML and PDF output.
"""

import os
import json
import logging
import datetime
from typing import Optional

from fpdf import FPDF
from llm_config import async_chat_completion, is_llm_configured

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# ---------------------------------------------------------------------------
# HTML Template & CSS
# ---------------------------------------------------------------------------

_CSS = """
body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: #0e1117;
    color: #fafafa;
    margin: 0;
    padding: 24px 40px;
    line-height: 1.6;
}
h1 {
    color: #4fc3f7;
    border-bottom: 2px solid #4fc3f7;
    padding-bottom: 8px;
    margin-top: 0;
}
h2 {
    color: #81d4fa;
    margin-top: 36px;
    margin-bottom: 8px;
    border-bottom: 1px solid #263238;
    padding-bottom: 4px;
}
.section { margin-bottom: 28px; }
.meta { color: #90a4ae; font-size: 0.85em; margin-bottom: 20px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}
th, td {
    text-align: left;
    padding: 8px 12px;
    border: 1px solid #37474f;
}
th { background: #1a237e; color: #e3f2fd; }
td { background: #1c1c2e; }
tr:nth-child(even) td { background: #212138; }
.badge-bull { color: #66bb6a; font-weight: bold; }
.badge-bear { color: #ef5350; font-weight: bold; }
.badge-neutral { color: #ffa726; font-weight: bold; }
ul { padding-left: 20px; }
li { margin-bottom: 4px; }
.risk-high { color: #ef5350; }
.risk-medium { color: #ffa726; }
.risk-low { color: #66bb6a; }
.signal-box {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 1.1em;
}
.signal-bullish { background: #1b5e20; color: #a5d6a7; }
.signal-bearish { background: #b71c1c; color: #ef9a9a; }
.signal-neutral { background: #e65100; color: #ffcc80; }
"""


def _html_wrap(title: str, body: str, generated_at: str) -> str:
    """Wrap section HTML in a full document."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body>"
        f"<h1>FlowTrace Research Report — {title}</h1>"
        f"<div class='meta'>Generated {generated_at}</div>"
        f"{body}</body></html>"
    )


def _section_html(heading: str, content: str) -> str:
    return f"<div class='section'><h2>{heading}</h2>{content}</div>"


def _dict_table(data: dict, col_key: str = "Metric", col_val: str = "Value") -> str:
    """Render a flat dict as a two-column HTML table."""
    if not data:
        return "<p><em>No data available.</em></p>"
    rows = "".join(
        f"<tr><td>{_esc(str(k))}</td><td>{_esc(str(v))}</td></tr>"
        for k, v in data.items()
        if v is not None
    )
    return (
        f"<table><thead><tr><th>{col_key}</th><th>{col_val}</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _list_html(items: list) -> str:
    if not items:
        return "<p><em>None listed.</em></p>"
    lis = "".join(f"<li>{_esc(str(i))}</li>" for i in items)
    return f"<ul>{lis}</ul>"


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _signal_badge(direction: str) -> str:
    d = (direction or "").upper()
    if d in ("BULLISH", "BUY", "LONG"):
        cls = "signal-bullish"
    elif d in ("BEARISH", "SELL", "SHORT"):
        cls = "signal-bearish"
    else:
        cls = "signal-neutral"
    return f"<span class='signal-box {cls}'>{_esc(direction or 'N/A')}</span>"


def _safe_get(d: dict, *keys, default=None):
    """Traverse nested dicts safely."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

async def _llm_narrative(system_prompt: str, user_prompt: str) -> str:
    """Call LLM for a narrative section.  Returns empty string on failure."""
    if not is_llm_configured():
        return ""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = await async_chat_completion(messages, temperature=0.3)
        return result.strip()
    except Exception as e:
        logger.warning("LLM narrative generation failed: %s", e)
        return ""


_REPORT_SYSTEM = (
    "You are a senior equity research analyst writing an institutional-grade "
    "investment report. Be concise, specific, and data-driven. "
    "Do NOT invent numbers — only reference data provided to you. "
    "Write in professional prose suitable for a hedge fund audience."
)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

async def _build_executive_summary(
    ticker: str, consensus_signal: dict, evidence: dict
) -> str:
    """Section 1 — LLM-generated 3-5 sentence summary."""
    data_blob = json.dumps(
        {
            "ticker": ticker,
            "direction": consensus_signal.get("direction"),
            "confidence": consensus_signal.get("confidence"),
            "expected_return": consensus_signal.get("expected_return"),
            "risk_factors": consensus_signal.get("risk_factors", [])[:3],
            "catalysts": consensus_signal.get("catalysts", [])[:3],
            "evidence_summary": {
                k: v
                for k, v in (evidence or {}).items()
                if k in (
                    "finbert_sentiment",
                    "analyst_consensus",
                    "insider_activity",
                    "short_interest",
                )
            },
        },
        default=str,
    )
    prompt = (
        f"Write an executive summary (3-5 sentences) for {ticker}. "
        "Cover: key thesis, primary catalyst, expected return, and main risk. "
        f"Data:\n{data_blob}"
    )
    narrative = await _llm_narrative(_REPORT_SYSTEM, prompt)
    if not narrative:
        direction = consensus_signal.get("direction", "N/A")
        confidence = consensus_signal.get("confidence", "N/A")
        narrative = (
            f"<p>Consensus signal: <strong>{_esc(str(direction))}</strong> "
            f"with confidence {_esc(str(confidence))}%.</p>"
        )
    else:
        narrative = f"<p>{_esc(narrative)}</p>"
    return _section_html(
        "1. Executive Summary",
        f"{_signal_badge(consensus_signal.get('direction', ''))}<br><br>{narrative}",
    )


async def _build_investment_thesis(ticker: str, consensus_signal: dict) -> str:
    """Section 2 — LLM narrative: core argument, mispricing."""
    data_blob = json.dumps(
        {
            "ticker": ticker,
            "direction": consensus_signal.get("direction"),
            "confidence": consensus_signal.get("confidence"),
            "expected_return": consensus_signal.get("expected_return"),
            "catalysts": consensus_signal.get("catalysts", []),
            "risk_factors": consensus_signal.get("risk_factors", []),
        },
        default=str,
    )
    prompt = (
        f"Write a concise investment thesis for {ticker} (2-3 paragraphs). "
        "Explain the core argument and what the market is mispricing. "
        f"Data:\n{data_blob}"
    )
    narrative = await _llm_narrative(_REPORT_SYSTEM, prompt)
    if not narrative:
        return _section_html(
            "2. Investment Thesis",
            "<p><em>LLM not configured — narrative unavailable.</em></p>",
        )
    return _section_html("2. Investment Thesis", f"<p>{_esc(narrative)}</p>")


async def _build_business_overview(ticker: str, business_model: dict) -> str:
    """Section 3 — LLM from business_model dict.  Skip if no data."""
    if not business_model:
        return ""
    data_blob = json.dumps(business_model, default=str)
    prompt = (
        f"Write a business overview for {ticker} covering: revenue model, "
        "competitive moat, customer concentration, and TAM. "
        f"Business model data:\n{data_blob}"
    )
    narrative = await _llm_narrative(_REPORT_SYSTEM, prompt)
    if not narrative:
        return _section_html("3. Business Overview", _dict_table(business_model))
    return _section_html("3. Business Overview", f"<p>{_esc(narrative)}</p>")


def _build_financial_analysis(consensus_signal: dict, evidence: dict) -> str:
    """Section 4 — DATA only: key metrics table."""
    metrics = {}

    # Pull from consensus_signal
    for key in (
        "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "ev_ebitda",
        "roe", "roa", "profit_margin", "operating_margin",
        "debt_to_equity", "current_ratio", "quick_ratio",
        "revenue_growth", "earnings_growth", "free_cash_flow",
    ):
        val = consensus_signal.get(key) or _safe_get(evidence or {}, "fundamentals", key)
        if val is not None:
            label = key.replace("_", " ").title()
            metrics[label] = val

    # Pull from evidence fundamentals block
    fundamentals = (evidence or {}).get("fundamentals", {})
    if isinstance(fundamentals, dict):
        for k, v in fundamentals.items():
            label = k.replace("_", " ").title()
            if label not in metrics and v is not None:
                metrics[label] = v

    if not metrics:
        return _section_html(
            "4. Financial Analysis",
            "<p><em>No financial metrics available.</em></p>",
        )
    return _section_html("4. Financial Analysis", _dict_table(metrics))


async def _build_valuation(ticker: str, valuation: dict) -> str:
    """Section 5 — DATA + LLM: fair-value numbers + LLM interpretation."""
    if not valuation:
        return _section_html(
            "5. Valuation",
            "<p><em>Valuation data not available.</em></p>",
        )

    # Data table
    display = {}
    for key in (
        "dcf_fair_value", "relative_fair_value", "technical_fair_value",
        "composite_fair_value", "current_price", "upside_pct",
        "margin_of_safety",
    ):
        val = valuation.get(key)
        if val is not None:
            display[key.replace("_", " ").title()] = val
    # Include any other keys
    for k, v in valuation.items():
        label = k.replace("_", " ").title()
        if label not in display and v is not None and not isinstance(v, (dict, list)):
            display[label] = v

    table_html = _dict_table(display)

    # LLM interpretation
    data_blob = json.dumps(valuation, default=str)
    prompt = (
        f"Briefly interpret these valuation metrics for {ticker}. "
        "What does the fair-value range imply? 2-3 sentences. "
        f"Data:\n{data_blob}"
    )
    narrative = await _llm_narrative(_REPORT_SYSTEM, prompt)
    interp = f"<p>{_esc(narrative)}</p>" if narrative else ""
    return _section_html("5. Valuation", table_html + interp)


def _build_technical_analysis(technical_data: dict) -> str:
    """Section 6 — DATA: indicator readings and signals."""
    if not technical_data:
        return _section_html(
            "6. Technical Analysis",
            "<p><em>Technical data not available.</em></p>",
        )

    parts = []

    # Indicator readings
    indicators = {
        k: v
        for k, v in technical_data.items()
        if k not in ("signals", "signals_detected", "pattern_signals")
        and not isinstance(v, (dict, list))
    }
    if indicators:
        parts.append("<h3>Indicator Readings</h3>")
        parts.append(_dict_table(indicators, "Indicator", "Reading"))

    # Signals detected
    signals = (
        technical_data.get("signals")
        or technical_data.get("signals_detected")
        or technical_data.get("pattern_signals")
        or []
    )
    if isinstance(signals, list) and signals:
        parts.append("<h3>Signals Detected</h3>")
        parts.append(_list_html(signals))
    elif isinstance(signals, dict) and signals:
        parts.append("<h3>Signals Detected</h3>")
        parts.append(_dict_table(signals, "Signal", "Detail"))

    # Nested dicts (e.g. moving_averages, oscillators)
    for key, val in technical_data.items():
        if isinstance(val, dict) and key not in ("signals", "signals_detected", "pattern_signals"):
            parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>")
            parts.append(_dict_table(val))

    if not parts:
        parts.append("<p><em>No indicator data parsed.</em></p>")

    return _section_html("6. Technical Analysis", "\n".join(parts))


def _build_sentiment(evidence: dict) -> str:
    """Section 7 — DATA: FinBERT, Reddit, insider, short interest, analyst."""
    if not evidence:
        return _section_html(
            "7. Sentiment & Alt Data",
            "<p><em>No sentiment/evidence data available.</em></p>",
        )

    parts = []
    sections_map = {
        "finbert_sentiment": "FinBERT Sentiment",
        "reddit_sentiment": "Reddit Sentiment",
        "insider_activity": "Insider Activity",
        "short_interest": "Short Interest",
        "analyst_consensus": "Analyst Consensus",
        "social_sentiment": "Social Sentiment",
        "news_sentiment": "News Sentiment",
    }

    for key, label in sections_map.items():
        val = evidence.get(key)
        if val is None:
            continue
        parts.append(f"<h3>{label}</h3>")
        if isinstance(val, dict):
            parts.append(_dict_table(val))
        elif isinstance(val, list):
            parts.append(_list_html(val))
        else:
            parts.append(f"<p>{_esc(str(val))}</p>")

    # Catch any other evidence keys
    shown = set(sections_map.keys())
    for key, val in evidence.items():
        if key in shown or key == "fundamentals":
            continue
        if val is not None:
            parts.append(f"<h3>{key.replace('_', ' ').title()}</h3>")
            if isinstance(val, dict):
                parts.append(_dict_table(val))
            elif isinstance(val, list):
                parts.append(_list_html(val))
            else:
                parts.append(f"<p>{_esc(str(val))}</p>")

    if not parts:
        parts.append("<p><em>No sentiment data found in evidence.</em></p>")

    return _section_html("7. Sentiment & Alt Data", "\n".join(parts))


async def _build_risk_factors(consensus_signal: dict) -> str:
    """Section 8 — DATA + LLM: risk factors with probability/impact."""
    risks = consensus_signal.get("risk_factors", [])
    if not risks:
        return _section_html(
            "8. Risk Factors",
            "<p><em>No risk factors identified.</em></p>",
        )

    # Data list
    risk_html = _list_html(risks)

    # LLM probability/impact assessment
    data_blob = json.dumps({"risk_factors": risks}, default=str)
    prompt = (
        "For each risk factor below, briefly assess probability (high/medium/low) "
        "and potential impact (high/medium/low) in 1 sentence each. "
        f"Risk factors:\n{data_blob}"
    )
    narrative = await _llm_narrative(_REPORT_SYSTEM, prompt)
    assessment = f"<h3>Probability &amp; Impact Assessment</h3><p>{_esc(narrative)}</p>" if narrative else ""

    return _section_html("8. Risk Factors", risk_html + assessment)


def _build_agent_consensus(consensus_signal: dict, swarm_brief: dict) -> str:
    """Section 9 — DATA: contributing agents, swarm debate, confidence waterfall."""
    parts = []

    # Contributing agents
    agents = consensus_signal.get("contributing_agents") or consensus_signal.get("agents", [])
    if agents:
        parts.append("<h3>Contributing Agents</h3>")
        if isinstance(agents, list):
            parts.append(_list_html(agents))
        elif isinstance(agents, dict):
            parts.append(_dict_table(agents, "Agent", "Signal"))

    # Swarm debate summary
    if swarm_brief:
        parts.append("<h3>Swarm Debate Summary</h3>")
        if isinstance(swarm_brief, dict):
            summary = swarm_brief.get("summary") or swarm_brief.get("debate_summary")
            if summary:
                parts.append(f"<p>{_esc(str(summary))}</p>")
            outcome = swarm_brief.get("outcome") or swarm_brief.get("consensus")
            if outcome:
                parts.append(f"<p><strong>Outcome:</strong> {_esc(str(outcome))}</p>")
            # Show full dict if no known keys found
            remaining = {
                k: v for k, v in swarm_brief.items()
                if k not in ("summary", "debate_summary", "outcome", "consensus")
                and v is not None
            }
            if remaining:
                parts.append(_dict_table(remaining))
        else:
            parts.append(f"<p>{_esc(str(swarm_brief))}</p>")

    # Forum signal
    forum = consensus_signal.get("forum_signal") or consensus_signal.get("forum")
    if forum:
        parts.append("<h3>Forum Signal</h3>")
        if isinstance(forum, dict):
            parts.append(_dict_table(forum))
        else:
            parts.append(f"<p>{_esc(str(forum))}</p>")

    # Confidence waterfall / weights
    weights = consensus_signal.get("weights_applied") or consensus_signal.get("weights", {})
    if weights and isinstance(weights, dict):
        parts.append("<h3>Confidence Waterfall</h3>")
        parts.append(_dict_table(weights, "Component", "Weight"))

    if not parts:
        parts.append("<p><em>No agent consensus data available.</em></p>")

    return _section_html("9. Agent Consensus", "\n".join(parts))


def _build_trade_plan(trade_plan: dict) -> str:
    """Section 10 — DATA: entry/stop/targets/sizing."""
    if not trade_plan:
        return _section_html(
            "10. Trade Plan",
            "<p><em>No trade plan generated.</em></p>",
        )

    display = {}
    ordered_keys = [
        "entry_price", "stop_loss", "target_1", "target_2", "target_3",
        "risk_reward_ratio", "position_size", "position_pct",
        "max_loss", "timeframe", "strategy",
    ]
    for key in ordered_keys:
        val = trade_plan.get(key)
        if val is not None:
            display[key.replace("_", " ").title()] = val

    # Include remaining keys
    for k, v in trade_plan.items():
        label = k.replace("_", " ").title()
        if label not in display and v is not None and not isinstance(v, (dict, list)):
            display[label] = v

    parts = [_dict_table(display)]

    # Nested targets list
    targets = trade_plan.get("targets")
    if isinstance(targets, list) and targets:
        parts.append("<h3>Price Targets</h3>")
        parts.append(_list_html(targets))

    return _section_html("10. Trade Plan", "\n".join(parts))


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    """Custom FPDF subclass with header/footer."""

    def __init__(self, ticker: str, generated_at: str):
        super().__init__()
        self._ticker = ticker
        self._generated_at = generated_at

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, f"FlowTrace Research Report - {self._ticker}", 0, 1, "C")
        self.set_font("Arial", "I", 8)
        self.cell(0, 5, f"Generated {self._generated_at}", 0, 1, "C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} | FlowTrace", 0, 0, "C")


def _strip_html(html_text: str) -> str:
    """Crude HTML tag removal for PDF text."""
    import re
    text = re.sub(r"<br\s*/?>", "\n", html_text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()


def _write_pdf(ticker: str, sections: dict, generated_at: str) -> Optional[str]:
    """Generate a PDF and save to reports/ directory.  Returns path or None."""
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        pdf = _ReportPDF(ticker, generated_at)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        for title, html_content in sections.items():
            if not html_content:
                continue
            # Section heading
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, title, 0, 1)
            pdf.ln(2)

            # Body text
            body = _strip_html(html_content)
            if body:
                pdf.set_font("Arial", "", 10)
                # encode to latin-1 safe
                safe_body = body.encode("latin-1", errors="replace").decode("latin-1")
                pdf.multi_cell(0, 5, safe_body)
            pdf.ln(4)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_{timestamp}.pdf"
        filepath = os.path.join(REPORTS_DIR, filename)
        pdf.output(filepath)
        logger.info("PDF report saved: %s", filepath)
        return filepath
    except Exception as e:
        logger.warning("PDF generation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_research_report(
    ticker: str,
    consensus_signal: dict,
    evidence: dict = None,
    business_model: dict = None,
    valuation: dict = None,
    technical_data: dict = None,
    catalyst_data: dict = None,
    trade_plan: dict = None,
    swarm_brief: dict = None,
    format: str = "html",
) -> dict:
    """
    Generate an institutional-grade research report.

    Returns:
        {
            "html": str,           # Full HTML document
            "pdf_path": str|None,  # Path to PDF if format includes pdf
            "sections": dict,      # Section title -> HTML fragment
            "generated_at": str,
        }
    """
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    consensus_signal = consensus_signal or {}
    evidence = evidence or {}

    # Build sections concurrently where possible
    import asyncio

    # LLM sections (can run in parallel)
    (
        exec_summary,
        investment_thesis,
        business_overview,
        valuation_section,
        risk_factors,
    ) = await asyncio.gather(
        _build_executive_summary(ticker, consensus_signal, evidence),
        _build_investment_thesis(ticker, consensus_signal),
        _build_business_overview(ticker, business_model),
        _build_valuation(ticker, valuation),
        _build_risk_factors(consensus_signal),
    )

    # Data-only sections (synchronous, fast)
    financial_analysis = _build_financial_analysis(consensus_signal, evidence)
    technical_analysis = _build_technical_analysis(technical_data)
    sentiment = _build_sentiment(evidence)
    agent_consensus = _build_agent_consensus(consensus_signal, swarm_brief)
    trade_plan_section = _build_trade_plan(trade_plan)

    # Ordered sections dict
    sections = {}
    section_list = [
        ("1. Executive Summary", exec_summary),
        ("2. Investment Thesis", investment_thesis),
        ("3. Business Overview", business_overview),
        ("4. Financial Analysis", financial_analysis),
        ("5. Valuation", valuation_section),
        ("6. Technical Analysis", technical_analysis),
        ("7. Sentiment & Alt Data", sentiment),
        ("8. Risk Factors", risk_factors),
        ("9. Agent Consensus", agent_consensus),
        ("10. Trade Plan", trade_plan_section),
    ]
    body_parts = []
    for title, html_fragment in section_list:
        if html_fragment:  # skip empty (e.g. business overview with no data)
            sections[title] = html_fragment
            body_parts.append(html_fragment)

    full_html = _html_wrap(ticker, "\n".join(body_parts), generated_at)

    # PDF
    pdf_path = None
    fmt = (format or "html").lower()
    if "pdf" in fmt:
        pdf_path = _write_pdf(ticker, sections, generated_at)

    return {
        "html": full_html,
        "pdf_path": pdf_path,
        "sections": sections,
        "generated_at": generated_at,
    }


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

def generate_signal_report(signals_df):
    """
    Legacy function for backward compatibility with existing app.py calls.

    Generates a PDF report for the provided signals DataFrame.
    Returns the binary content of the PDF.
    """
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 15)
    pdf.cell(0, 10, "FlowTrace - High Conviction Signal Report", 0, 1, "C")
    pdf.ln(5)

    # Metadata
    pdf.set_font("Arial", "B", 12)
    pdf.cell(
        0, 10,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        0, 1,
    )
    pdf.cell(0, 10, f"Total Signals: {len(signals_df)}", 0, 1)
    pdf.ln(5)

    if signals_df.empty:
        pdf.set_font("Arial", "I", 12)
        pdf.cell(0, 10, "No high-conviction signals found matching criteria.", 0, 1)
        return pdf.output(dest="S").encode("latin-1")

    # Footer setup
    pdf.set_auto_page_break(auto=True, margin=15)

    for _index, row in signals_df.iterrows():
        # Signal header
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(200, 220, 255)
        title = (
            f"{row.get('source_ticker', 'N/A')} -> "
            f"{row.get('target_ticker', 'N/A')} "
            f"({row.get('event_type', 'N/A')})"
        )
        pdf.cell(0, 10, title, 0, 1, "L", fill=True)

        # Metrics
        pdf.set_font("Arial", size=10)
        metrics = (
            f"Confidence: {row.get('confidence', 'N/A')}% | "
            f"Unified Score: {row.get('unified_score', 'N/A')} | "
            f"Exp Move: {row.get('expected_move_pct', 'N/A')}%"
        )
        pdf.cell(0, 8, metrics, 0, 1)

        # Summary
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Summary:", 0, 1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, str(row.get("summary", "")))
        pdf.ln(2)

        # Reasoning
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Reasoning:", 0, 1)
        pdf.set_font("Arial", size=10)
        reasoning = str(row.get("reasoning", "")).replace("\n", " ")
        pdf.multi_cell(0, 5, reasoning)

        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    return pdf.output(dest="S").encode("latin-1")
