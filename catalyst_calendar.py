"""
FlowTrace Catalyst Calendar

Aggregates forward-looking company, macro, and market events into a
unified catalyst calendar used by signal generators and risk checks.
"""

import os
import json
import logging
from datetime import datetime, date, timedelta

import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_macro_cache = None


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _load_macro_calendar() -> dict:
    """Load and cache macro_calendar.json from the FlowTrace base directory."""
    global _macro_cache
    if _macro_cache is not None:
        return _macro_cache

    path = os.path.join(BASE_DIR, "macro_calendar.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            _macro_cache = json.load(fh)
            logger.info("Loaded macro calendar from %s", path)
    except FileNotFoundError:
        logger.error("macro_calendar.json not found at %s", path)
        _macro_cache = {
            "fed_meetings": [],
            "cpi_releases": [],
            "jobs_reports": [],
            "index_rebalance": [],
        }
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse macro_calendar.json: %s", exc)
        _macro_cache = {
            "fed_meetings": [],
            "cpi_releases": [],
            "jobs_reports": [],
            "index_rebalance": [],
        }
    return _macro_cache


def _next_monthly_opex() -> str:
    """Calculate the next monthly options expiration (3rd Friday of the month).

    Returns an ISO-format date string (YYYY-MM-DD).
    """
    today = date.today()
    year, month = today.year, today.month

    # Find the 3rd Friday of the current month
    first_day = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    third_friday = first_friday + timedelta(weeks=2)

    # If the 3rd Friday has already passed, move to next month
    if third_friday <= today:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        first_day = date(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)

    return third_friday.isoformat()


def _next_weekly_opex() -> str:
    """Calculate the next weekly options expiration (next Friday).

    Returns an ISO-format date string (YYYY-MM-DD).
    """
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        # Today is Friday; if market is still open this counts, but for
        # forward-looking purposes we pick next Friday.
        days_ahead = 7
    next_friday = today + timedelta(days=days_ahead)
    return next_friday.isoformat()


def _days_away(date_str: str) -> int:
    """Return the number of calendar days from today to *date_str*."""
    target = date.fromisoformat(date_str)
    return (target - date.today()).days


def _upcoming(events: list[dict], horizon_days: int = 90) -> list[dict]:
    """Filter a list of dated events to those within *horizon_days* from today."""
    today_iso = date.today().isoformat()
    cutoff = (date.today() + timedelta(days=horizon_days)).isoformat()
    return [e for e in events if today_iso <= e["date"] <= cutoff]


# ---------------------------------------------------------------------------
# Company Events (yfinance)
# ---------------------------------------------------------------------------

def _fetch_company_events(ticker: str) -> list[dict]:
    """Retrieve earnings and ex-dividend dates from yfinance."""
    events: list[dict] = []
    try:
        tkr = yf.Ticker(ticker)

        # --- Earnings ---
        try:
            cal = tkr.calendar
            if cal is not None:
                # yfinance may return a dict or DataFrame depending on version
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date:
                        # May be a list of dates or a single value
                        if isinstance(earnings_date, list):
                            for ed in earnings_date:
                                ed_str = _to_date_str(ed)
                                if ed_str:
                                    da = _days_away(ed_str)
                                    events.append({
                                        "type": "earnings",
                                        "date": ed_str,
                                        "days_away": da,
                                        "details": f"{ticker} earnings release",
                                        "impact": "high",
                                    })
                        else:
                            ed_str = _to_date_str(earnings_date)
                            if ed_str:
                                da = _days_away(ed_str)
                                events.append({
                                    "type": "earnings",
                                    "date": ed_str,
                                    "days_away": da,
                                    "details": f"{ticker} earnings release",
                                    "impact": "high",
                                })
                else:
                    # DataFrame path (older yfinance versions)
                    if hasattr(cal, "columns"):
                        for col in cal.columns:
                            if "Earnings Date" in str(cal[col].values):
                                pass  # best-effort; dict path is primary
                    # Also try .index based approach
                    if "Earnings Date" in getattr(cal, "index", []):
                        raw = cal.loc["Earnings Date"]
                        for val in (raw if hasattr(raw, "__iter__") else [raw]):
                            ed_str = _to_date_str(val)
                            if ed_str:
                                da = _days_away(ed_str)
                                events.append({
                                    "type": "earnings",
                                    "date": ed_str,
                                    "days_away": da,
                                    "details": f"{ticker} earnings release",
                                    "impact": "high",
                                })
        except Exception as exc:
            logger.debug("Could not fetch earnings calendar for %s: %s", ticker, exc)

        # --- Ex-dividend ---
        try:
            info = tkr.info or {}
            ex_div = info.get("exDividendDate")
            if ex_div:
                ed_str = _to_date_str(ex_div)
                if ed_str:
                    da = _days_away(ed_str)
                    events.append({
                        "type": "ex_dividend",
                        "date": ed_str,
                        "days_away": da,
                        "details": f"{ticker} ex-dividend date",
                        "impact": "medium",
                    })
        except Exception as exc:
            logger.debug("Could not fetch dividend info for %s: %s", ticker, exc)

    except Exception as exc:
        logger.warning("yfinance lookup failed for %s: %s", ticker, exc)

    return events


def _to_date_str(value) -> str | None:
    """Best-effort conversion of various date representations to YYYY-MM-DD."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value).strftime("%Y-%m-%d")
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Macro Events
# ---------------------------------------------------------------------------

def _build_macro_events(ticker: str, horizon_days: int = 90) -> list[dict]:
    """Build a list of upcoming macro events relevant to equities."""
    cal = _load_macro_calendar()
    events: list[dict] = []

    # Fed meetings
    for entry in _upcoming(cal.get("fed_meetings", []), horizon_days):
        da = _days_away(entry["date"])
        has_projections = entry["type"] == "rate_decision_projections"
        events.append({
            "type": "fed_meeting",
            "date": entry["date"],
            "days_away": da,
            "details": "FOMC rate decision" + (" with projections" if has_projections else ""),
            "impact": "high" if has_projections else "medium",
            "relevance": (
                f"Rate decisions affect cost of capital and discount rates for {ticker}. "
                "Projection meetings carry higher vol impact."
            ),
        })

    # CPI releases
    for entry in _upcoming(cal.get("cpi_releases", []), horizon_days):
        da = _days_away(entry["date"])
        events.append({
            "type": "cpi_release",
            "date": entry["date"],
            "days_away": da,
            "details": "BLS Consumer Price Index release",
            "impact": "high",
            "relevance": (
                f"CPI surprises drive rate expectations and broad equity repricing affecting {ticker}."
            ),
        })

    # Jobs reports
    for entry in _upcoming(cal.get("jobs_reports", []), horizon_days):
        da = _days_away(entry["date"])
        events.append({
            "type": "jobs_report",
            "date": entry["date"],
            "days_away": da,
            "details": "BLS Nonfarm Payrolls release",
            "impact": "high",
            "relevance": (
                f"Employment data influences Fed policy trajectory and consumer spending outlook for {ticker}."
            ),
        })

    return events


# ---------------------------------------------------------------------------
# Market / Structure Events
# ---------------------------------------------------------------------------

def _build_market_events(horizon_days: int = 90) -> list[dict]:
    """Build a list of upcoming market-structure events."""
    cal = _load_macro_calendar()
    events: list[dict] = []

    # Monthly opex
    monthly_opex = _next_monthly_opex()
    da = _days_away(monthly_opex)
    if 0 < da <= horizon_days:
        events.append({
            "type": "monthly_options_expiry",
            "date": monthly_opex,
            "days_away": da,
            "details": "Monthly options expiration (3rd Friday)",
            "impact": "medium",
        })

    # Weekly opex
    weekly_opex = _next_weekly_opex()
    da = _days_away(weekly_opex)
    if 0 < da <= horizon_days:
        events.append({
            "type": "weekly_options_expiry",
            "date": weekly_opex,
            "days_away": da,
            "details": "Weekly options expiration (Friday)",
            "impact": "low",
        })

    # Index rebalances
    for entry in _upcoming(cal.get("index_rebalance", []), horizon_days):
        da = _days_away(entry["date"])
        label = (
            "Russell reconstitution" if entry["type"] == "russell_reconstitution"
            else "S&P 500 quarterly rebalance"
        )
        events.append({
            "type": entry["type"],
            "date": entry["date"],
            "days_away": da,
            "details": label,
            "impact": "medium",
        })

    return events


# ---------------------------------------------------------------------------
# Signal Context
# ---------------------------------------------------------------------------

def _compute_signal_context(
    company_events: list[dict],
    macro_events: list[dict],
    market_events: list[dict],
    horizon_days: int = 5,
) -> dict:
    """Derive a concise signal-context summary from event lists."""
    all_events = company_events + macro_events + market_events
    high_impact = [e for e in all_events if e.get("impact") == "high" and 0 < e.get("days_away", 999) <= horizon_days]

    # Nearest high-impact event
    nearest = None
    if high_impact:
        nearest_evt = min(high_impact, key=lambda e: e["days_away"])
        nearest = f"{nearest_evt['type']} on {nearest_evt['date']} ({nearest_evt['days_away']}d away)"

    # Earnings imminent?
    earnings_imminent = any(
        e["type"] == "earnings" and 0 < e.get("days_away", 999) <= horizon_days
        for e in company_events
    )

    # Trading window clear = no high-impact events within horizon
    trading_window_clear = len(high_impact) == 0

    # Warning
    warning = None
    if earnings_imminent:
        warning = f"Earnings within {horizon_days} days -- elevated vol expected."
    elif not trading_window_clear:
        warning = f"High-impact macro/market event within {horizon_days} days."

    return {
        "nearest_high_impact_event": nearest,
        "earnings_imminent": earnings_imminent,
        "trading_window_clear": trading_window_clear,
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_catalyst_calendar(ticker: str, horizon_days: int = 90) -> dict:
    """Aggregate all forward-looking events for *ticker* into a single dict.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (e.g. ``"SWKS"``).
    horizon_days : int
        How many calendar days ahead to scan for macro/market events.

    Returns
    -------
    dict
        Unified catalyst calendar with ``company_events``, ``macro_events``,
        ``market_events``, and ``signal_context`` keys.
    """
    logger.info("Building catalyst calendar for %s (horizon=%dd)", ticker, horizon_days)

    company_events = _fetch_company_events(ticker)
    macro_events = _build_macro_events(ticker, horizon_days)
    market_events = _build_market_events(horizon_days)
    signal_context = _compute_signal_context(company_events, macro_events, market_events)

    return {
        "ticker": ticker.upper(),
        "generated_at": datetime.utcnow().isoformat(),
        "company_events": company_events,
        "macro_events": macro_events,
        "market_events": market_events,
        "signal_context": signal_context,
    }


def get_signal_context(ticker: str, horizon_days: int = 5) -> dict:
    """Return only the signal-context portion for quick risk checks.

    This is a lightweight wrapper around :func:`get_catalyst_calendar` that
    fetches the full calendar and extracts the ``signal_context`` block.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.
    horizon_days : int
        Look-ahead window in calendar days (default 5).

    Returns
    -------
    dict
        Signal context dict with keys ``nearest_high_impact_event``,
        ``earnings_imminent``, ``trading_window_clear``, and ``warning``.
    """
    company_events = _fetch_company_events(ticker)
    macro_events = _build_macro_events(ticker, horizon_days)
    market_events = _build_market_events(horizon_days)
    return _compute_signal_context(company_events, macro_events, market_events, horizon_days)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    test_ticker = os.getenv("FLOWTRACE_TEST_TICKER", "SWKS")
    print(f"--- Catalyst Calendar for {test_ticker} ---")
    pprint.pprint(get_catalyst_calendar(test_ticker))
    print(f"\n--- Signal Context (5d) for {test_ticker} ---")
    pprint.pprint(get_signal_context(test_ticker))
