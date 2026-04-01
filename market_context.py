"""
FlowTrace Market Context

Captures a snapshot of the current market environment at signal time.
Used by the signal pipeline to enrich signals with market context,
and to activate sector and regime multipliers in agent_consensus.py.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Sector ETF mapping (GICS sectors -> SPDR ETFs)
# ---------------------------------------------------------------------------

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# ---------------------------------------------------------------------------
# Simple in-memory cache: key -> (timestamp, data)
# ---------------------------------------------------------------------------

_cache: dict = {}


def _get_cache_ttl() -> int:
    """Return cache TTL in seconds. Day traders get 1-minute cache."""
    try:
        from trader_profile import get_trading_style
        style = get_trading_style()
        if style and "day" in style.lower():
            return 60
    except Exception:
        pass
    return 300  # default 5 minutes


def _get_cached(key: str) -> Optional[dict]:
    """Return cached value if still fresh, else None."""
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < _get_cache_ttl():
            return data
    return None


def _set_cached(key: str, data: dict) -> None:
    _cache[key] = (time.time(), data)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(vix: float) -> str:
    """Classify VIX into a regime label.

    Returns one of ``"low_vix"``, ``"normal_vix"``, or ``"high_vix"``.
    """
    if vix < 15:
        return "low_vix"
    elif vix <= 25:
        return "normal_vix"
    else:
        return "high_vix"


# ---------------------------------------------------------------------------
# Earnings proximity
# ---------------------------------------------------------------------------

def is_earnings_imminent(ticker: str) -> dict:
    """Check how many days until the next earnings date for *ticker*.

    Returns a dict with keys ``days_to_earnings``, ``date``, and
    ``eps_estimate`` (each may be ``None`` if unavailable).
    """
    result = {"days_to_earnings": None, "date": None, "eps_estimate": None}
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or (hasattr(cal, "empty") and cal.empty):
            return result

        # yfinance calendar can be a DataFrame or a dict depending on version
        earnings_date = None
        eps_estimate = None

        if isinstance(cal, dict):
            # Newer yfinance versions return a dict
            raw = cal.get("Earnings Date") or cal.get("Earnings Dates")
            if raw:
                if isinstance(raw, list) and len(raw) > 0:
                    earnings_date = raw[0]
                elif isinstance(raw, datetime):
                    earnings_date = raw
            eps_estimate = cal.get("EPS Estimate")
        else:
            # Older versions return a DataFrame
            try:
                if "Earnings Date" in cal.columns:
                    earnings_date = cal["Earnings Date"].iloc[0]
                elif "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"]
                    if hasattr(val, "iloc"):
                        earnings_date = val.iloc[0]
                    else:
                        earnings_date = val
            except Exception:
                pass
            try:
                if "EPS Estimate" in getattr(cal, "index", []):
                    val = cal.loc["EPS Estimate"]
                    eps_estimate = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
            except Exception:
                pass

        if earnings_date is not None:
            if isinstance(earnings_date, str):
                earnings_date = datetime.fromisoformat(earnings_date)
            if hasattr(earnings_date, "tzinfo") and earnings_date.tzinfo is None:
                earnings_date = earnings_date.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (earnings_date - now).days
            result["days_to_earnings"] = max(delta, 0)
            result["date"] = earnings_date.strftime("%Y-%m-%d")

        if eps_estimate is not None:
            try:
                result["eps_estimate"] = float(eps_estimate)
            except (TypeError, ValueError):
                pass

    except Exception as exc:
        logger.debug("Could not fetch earnings calendar for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_daily_change(tk_obj) -> Optional[float]:
    """Return the most recent 1-day percentage change, or None."""
    try:
        hist = tk_obj.history(period="2d")
        if hist is not None and len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            last_close = hist["Close"].iloc[-1]
            if prev_close and prev_close != 0:
                return round((last_close - prev_close) / prev_close * 100, 3)
    except Exception:
        pass
    return None


def _safe_price(tk_obj) -> Optional[float]:
    """Return the most recent closing price, or None."""
    try:
        hist = tk_obj.history(period="1d")
        if hist is not None and len(hist) > 0:
            return round(float(hist["Close"].iloc[-1]), 4)
    except Exception:
        pass
    return None


def _get_ticker_sector(ticker: str) -> Optional[str]:
    """Attempt to determine the GICS sector for *ticker*."""
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("sector")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main capture function
# ---------------------------------------------------------------------------

async def capture_market_context(ticker: str = None) -> dict:
    """Capture a snapshot of the current market environment.

    Results are cached for 5 minutes (or 1 minute for day-trading
    profiles).  If *ticker* is provided, ticker-specific data and
    sector information are included.

    Returns a dict with keys: ``vix``, ``indices``, ``sector``,
    ``yields``, ``ticker_detail``, ``earnings``, ``timestamp``.
    """
    cache_key = f"market_context:{ticker or '__none__'}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    context: dict = {
        "vix": None,
        "indices": {},
        "sector": None,
        "yields": None,
        "ticker_detail": None,
        "earnings": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ---- VIX ----------------------------------------------------------
    try:
        vix_tk = yf.Ticker("^VIX")
        vix_price = _safe_price(vix_tk)
        vix_change = _safe_daily_change(vix_tk)
        context["vix"] = {
            "level": vix_price,
            "regime": classify_regime(vix_price) if vix_price is not None else None,
            "change_1d": vix_change,
        }
    except Exception as exc:
        logger.warning("Failed to fetch VIX data: %s", exc)

    # ---- Major indices ------------------------------------------------
    for symbol in ("SPY", "QQQ", "IWM"):
        try:
            idx_tk = yf.Ticker(symbol)
            context["indices"][symbol] = {
                "price": _safe_price(idx_tk),
                "change_pct": _safe_daily_change(idx_tk),
            }
        except Exception as exc:
            logger.warning("Failed to fetch %s data: %s", symbol, exc)
            context["indices"][symbol] = {"price": None, "change_pct": None}

    # ---- Treasury yields ---------------------------------------------
    try:
        tnx = yf.Ticker("^TNX")
        irx = yf.Ticker("^IRX")
        yield_10y = _safe_price(tnx)
        yield_2y = _safe_price(irx)
        spread = None
        if yield_10y is not None and yield_2y is not None:
            spread = round(yield_10y - yield_2y, 4)
        context["yields"] = {
            "10y": yield_10y,
            "2y": yield_2y,
            "spread_10y_2y": spread,
        }
    except Exception as exc:
        logger.warning("Failed to fetch treasury yield data: %s", exc)

    # ---- Ticker-specific data -----------------------------------------
    if ticker:
        # Sector ETF
        try:
            sector_name = _get_ticker_sector(ticker)
            if sector_name:
                etf_symbol = SECTOR_ETF_MAP.get(sector_name)
                if etf_symbol:
                    etf_tk = yf.Ticker(etf_symbol)
                    etf_change = _safe_daily_change(etf_tk)
                    spy_change = (context["indices"].get("SPY") or {}).get("change_pct")

                    relative_strength = None
                    if etf_change is not None and spy_change is not None:
                        relative_strength = round(etf_change - spy_change, 3)

                    context["sector"] = {
                        "name": sector_name,
                        "etf": etf_symbol,
                        "change_pct": etf_change,
                        "relative_strength_vs_spy": relative_strength,
                    }
                else:
                    context["sector"] = {"name": sector_name, "etf": None,
                                         "change_pct": None,
                                         "relative_strength_vs_spy": None}
        except Exception as exc:
            logger.warning("Failed to fetch sector data for %s: %s", ticker, exc)

        # Ticker price + volume
        try:
            tk = yf.Ticker(ticker)
            hist_20 = tk.history(period="1mo")
            current_price = None
            volume = None
            avg_volume_20d = None
            volume_ratio = None

            if hist_20 is not None and len(hist_20) > 0:
                current_price = round(float(hist_20["Close"].iloc[-1]), 4)
                volume = int(hist_20["Volume"].iloc[-1])

                if len(hist_20) >= 20:
                    avg_volume_20d = int(hist_20["Volume"].iloc[-20:].mean())
                else:
                    avg_volume_20d = int(hist_20["Volume"].mean())

                if avg_volume_20d and avg_volume_20d > 0:
                    volume_ratio = round(volume / avg_volume_20d, 3)

            context["ticker_detail"] = {
                "ticker": ticker,
                "price": current_price,
                "volume": volume,
                "avg_volume_20d": avg_volume_20d,
                "volume_ratio": volume_ratio,
            }
        except Exception as exc:
            logger.warning("Failed to fetch ticker detail for %s: %s", ticker, exc)

        # Earnings
        try:
            context["earnings"] = is_earnings_imminent(ticker)
        except Exception as exc:
            logger.warning("Failed to fetch earnings for %s: %s", ticker, exc)

    _set_cached(cache_key, context)
    return context


# ---------------------------------------------------------------------------
# Synchronous wrapper (safe to call from any context)
# ---------------------------------------------------------------------------

_sync_executor = None


def capture_market_context_sync(ticker: str = None) -> dict:
    """
    Synchronous wrapper for capture_market_context().

    Handles the async/sync boundary safely:
    - If no event loop is running: uses asyncio.run()
    - If an event loop IS running (e.g., inside ContinuousMonitorAgent):
      runs the async function in a separate thread via a shared executor

    This is the function that agent_consensus.py and ingestion_listener.py
    should call instead of trying to manage the event loop themselves.
    """
    import asyncio
    import concurrent.futures
    global _sync_executor

    # Check cache first (avoids thread overhead on cache hit)
    cache_key = f"market_context:{ticker or '__none__'}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(capture_market_context(ticker))
    else:
        # Event loop is running — run in a shared thread pool (reuse executor)
        if _sync_executor is None:
            _sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        future = _sync_executor.submit(asyncio.run, capture_market_context(ticker))
        try:
            return future.result(timeout=30)
        except (concurrent.futures.TimeoutError, Exception) as e:
            logger.warning(f"Sync market context capture failed: {e}")
            return {}
