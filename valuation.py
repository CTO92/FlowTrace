"""
FlowTrace Multi-Method Valuation Engine

Computes fair value ranges for stocks using three complementary approaches:
  - DCF (Discounted Cash Flow) intrinsic value
  - Relative valuation (comparable company multiples)
  - Technical fair value (support/resistance/VWAP range)

Results are synthesized into a weighted composite with conviction scoring.
All market data sourced from yfinance (already a project dependency).
"""

import logging
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector -> well-known peer map (fallback when API peers unavailable)
# ---------------------------------------------------------------------------

_SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "CRM", "ADBE"],
    "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS"],
    "Industrials": ["CAT", "HON", "UNP", "BA", "RTX", "DE", "GE", "LMT"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ", "PM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA", "WELL"],
    "Basic Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DD"],
}

# Default risk-free rate (approximate 10Y Treasury yield)
_DEFAULT_RISK_FREE_RATE = 0.043
_EQUITY_RISK_PREMIUM = 0.055
_TERMINAL_GROWTH = 0.025
_GDP_GROWTH_PROXY = 0.03


# ---------------------------------------------------------------------------
# 1. DCF Valuation
# ---------------------------------------------------------------------------

def dcf_valuation(ticker: str) -> dict:
    """
    5-year Discounted Cash Flow model.

    Uses trailing FCF, analyst growth consensus, CAPM-derived WACC,
    and Gordon Growth terminal value. Returns base/bull/bear sensitivity.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # --- Gather inputs ---
        fcf = info.get("freeCashflow")
        if fcf is None or fcf <= 0:
            return {"method": "DCF", "error": f"No positive free cash flow available for {ticker}"}

        shares = info.get("sharesOutstanding")
        if not shares or shares <= 0:
            return {"method": "DCF", "error": f"Shares outstanding unavailable for {ticker}"}

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            return {"method": "DCF", "error": f"Current price unavailable for {ticker}"}

        # Growth rate: prefer earningsGrowth, fall back to revenueGrowth, then 5%
        growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.05
        # Clamp to reasonable range
        growth = max(-0.10, min(growth, 0.40))

        beta = info.get("beta") or 1.0
        beta = max(0.3, min(beta, 3.0))

        wacc = _DEFAULT_RISK_FREE_RATE + beta * _EQUITY_RISK_PREMIUM

        assumptions = {
            "base_fcf": fcf,
            "growth_years_1_3": round(growth, 4),
            "growth_years_4_5": round(_GDP_GROWTH_PROXY, 4),
            "wacc": round(wacc, 4),
            "terminal_growth": _TERMINAL_GROWTH,
            "beta": round(beta, 2),
            "shares_outstanding": shares,
        }

        def _run_dcf(g: float, w: float) -> float:
            """Project FCF, discount, add terminal value, return per-share value."""
            projected_fcf = []
            cf = float(fcf)
            for year in range(1, 6):
                rate = g if year <= 3 else _GDP_GROWTH_PROXY
                cf *= (1 + rate)
                projected_fcf.append(cf)

            # Terminal value (Gordon Growth) — guard against WACC <= terminal growth
            if w <= _TERMINAL_GROWTH + 0.005:
                w = _TERMINAL_GROWTH + 0.01  # Floor WACC at terminal growth + 1%
            terminal_value = projected_fcf[-1] * (1 + _TERMINAL_GROWTH) / (w - _TERMINAL_GROWTH)

            # Discount everything back to present
            pv_fcf = sum(cf_i / (1 + w) ** yr for yr, cf_i in enumerate(projected_fcf, 1))
            pv_terminal = terminal_value / (1 + w) ** 5

            enterprise_value = pv_fcf + pv_terminal

            # Subtract net debt if available
            total_debt = info.get("totalDebt", 0) or 0
            cash = info.get("totalCash", 0) or 0
            equity_value = enterprise_value - total_debt + cash

            return max(equity_value / max(shares, 1), 0)

        base_case = _run_dcf(growth, wacc)
        bull_case = _run_dcf(growth + 0.02, max(wacc - 0.01, 0.04))
        bear_case = _run_dcf(growth - 0.02, wacc + 0.01)

        upside_pct = ((base_case - current_price) / current_price) * 100

        return {
            "method": "DCF",
            "fair_value_per_share": round(base_case, 2),
            "upside_pct": round(upside_pct, 2),
            "assumptions": assumptions,
            "sensitivity": {
                "bull_case": round(bull_case, 2),
                "base_case": round(base_case, 2),
                "bear_case": round(bear_case, 2),
            },
        }

    except Exception as e:
        logger.error(f"DCF valuation failed for {ticker}: {e}")
        return {"method": "DCF", "error": str(e)}


# ---------------------------------------------------------------------------
# 2. Relative Valuation
# ---------------------------------------------------------------------------

def _get_peers(ticker: str, info: dict) -> List[str]:
    """Resolve a list of peer tickers from sector/industry."""
    sector = info.get("sector", "")
    peers = list(_SECTOR_PEERS.get(sector, []))

    # Remove the ticker itself from peers
    upper_ticker = ticker.upper()
    peers = [p for p in peers if p.upper() != upper_ticker]

    if not peers:
        # Fallback: broad large-cap set
        peers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "PG"]
        peers = [p for p in peers if p.upper() != upper_ticker]

    return peers[:8]


def _safe_metric(info: dict, key: str) -> Optional[float]:
    """Extract a numeric metric, returning None if missing or non-positive."""
    val = info.get(key)
    if val is not None and isinstance(val, (int, float)) and val > 0:
        return float(val)
    return None


def relative_valuation(ticker: str) -> dict:
    """
    Comparable company analysis using P/E, EV/EBITDA, P/S, and P/B multiples.

    Calculates peer median multiples and applies them to the company's
    fundamentals to derive implied share prices.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            return {"method": "Relative", "error": f"Current price unavailable for {ticker}"}

        shares = info.get("sharesOutstanding")
        if not shares or shares <= 0:
            return {"method": "Relative", "error": f"Shares outstanding unavailable for {ticker}"}

        # Company fundamentals
        eps = _safe_metric(info, "trailingEps")
        ebitda = _safe_metric(info, "ebitda")
        revenue = _safe_metric(info, "totalRevenue")
        book_value = _safe_metric(info, "bookValue")

        ebitda_per_share = ebitda / shares if ebitda else None
        revenue_per_share = revenue / shares if revenue else None

        # Gather peer multiples
        peers = _get_peers(ticker, info)
        peer_pe, peer_ev_ebitda, peer_ps, peer_pb = [], [], [], []
        peers_used = []

        for peer_ticker in peers:
            try:
                p_info = yf.Ticker(peer_ticker).info or {}
                peers_used.append(peer_ticker)

                val = _safe_metric(p_info, "trailingPE")
                if val and val < 200:
                    peer_pe.append(val)

                val = _safe_metric(p_info, "enterpriseToEbitda")
                if val and val < 100:
                    peer_ev_ebitda.append(val)

                val = _safe_metric(p_info, "priceToSalesTrailing12Months")
                if val and val < 100:
                    peer_ps.append(val)

                val = _safe_metric(p_info, "priceToBook")
                if val and val < 100:
                    peer_pb.append(val)

            except Exception as e:
                logger.debug(f"Skipping peer {peer_ticker}: {e}")

        if not peers_used:
            return {"method": "Relative", "error": "No peer data could be retrieved"}

        # Compute implied prices from peer median multiples
        implied = {}
        prices_for_avg = []

        if peer_pe and eps:
            median_pe = float(np.median(peer_pe))
            implied["pe_implied"] = round(median_pe * eps, 2)
            prices_for_avg.append(implied["pe_implied"])
        else:
            implied["pe_implied"] = None

        if peer_ev_ebitda and ebitda_per_share:
            median_ev_ebitda = float(np.median(peer_ev_ebitda))
            implied["ev_ebitda_implied"] = round(median_ev_ebitda * ebitda_per_share, 2)
            prices_for_avg.append(implied["ev_ebitda_implied"])
        else:
            implied["ev_ebitda_implied"] = None

        if peer_ps and revenue_per_share:
            median_ps = float(np.median(peer_ps))
            implied["ps_implied"] = round(median_ps * revenue_per_share, 2)
            prices_for_avg.append(implied["ps_implied"])
        else:
            implied["ps_implied"] = None

        if peer_pb and book_value:
            median_pb = float(np.median(peer_pb))
            implied["pb_implied"] = round(median_pb * book_value, 2)
            prices_for_avg.append(implied["pb_implied"])
        else:
            implied["pb_implied"] = None

        if not prices_for_avg:
            return {"method": "Relative", "error": "Insufficient data to compute implied prices"}

        fair_value = float(np.mean(prices_for_avg))
        upside_pct = ((fair_value - current_price) / current_price) * 100

        # Percentile ranking vs peers
        company_pe = _safe_metric(info, "trailingPE")
        company_ev_ebitda = _safe_metric(info, "enterpriseToEbitda")

        def _percentile_rank(value, peer_list):
            if value is None or not peer_list:
                return None
            all_vals = sorted(peer_list + [value])
            rank = all_vals.index(value)
            return int(round(rank / len(all_vals) * 100))

        return {
            "method": "Relative",
            "fair_value_per_share": round(fair_value, 2),
            "upside_pct": round(upside_pct, 2),
            "peers_used": peers_used,
            "implied_prices": implied,
            "current_vs_peers": {
                "pe_percentile": _percentile_rank(company_pe, peer_pe),
                "ev_ebitda_percentile": _percentile_rank(company_ev_ebitda, peer_ev_ebitda),
            },
        }

    except Exception as e:
        logger.error(f"Relative valuation failed for {ticker}: {e}")
        return {"method": "Relative", "error": str(e)}


# ---------------------------------------------------------------------------
# 3. Technical Fair Value
# ---------------------------------------------------------------------------

def technical_fair_value(ticker: str) -> dict:
    """
    Support/resistance range from 1 year of daily price data.

    Combines SMA-200, 60-day extremes, Fibonacci levels, and
    volume-weighted price center to define a technical fair value range.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y", interval="1d")

        if hist.empty or len(hist) < 60:
            return {"method": "Technical", "error": f"Insufficient price history for {ticker}"}

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]

        # SMA 200
        sma_200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())

        # 60-day window
        recent_close = close.iloc[-60:]
        recent_high = high.iloc[-60:]
        recent_low = low.iloc[-60:]
        recent_volume = volume.iloc[-60:]

        lowest_low_60 = float(recent_low.min())
        highest_high_60 = float(recent_high.max())

        # Fibonacci levels based on 1-year range
        year_high = float(high.max())
        year_low = float(low.min())
        fib_range = year_high - year_low

        fib_618_retracement = year_high - fib_range * 0.618  # 61.8% retracement (support)
        fib_1618_extension = year_low + fib_range * 1.618    # 161.8% extension (resistance)

        # Key support: max of the three support indicators
        key_support = max(sma_200, lowest_low_60, fib_618_retracement)

        # Key resistance: min of the two resistance indicators
        key_resistance = min(highest_high_60, fib_1618_extension)

        # Ensure support < resistance; if not, widen the range
        if key_support >= key_resistance:
            key_support = min(sma_200, lowest_low_60, fib_618_retracement)
            key_resistance = max(highest_high_60, fib_1618_extension)

        # Volume-weighted center (approximate VWAP over last 60 days)
        typical_price_60 = (recent_high + recent_low + recent_close) / 3
        total_volume = recent_volume.sum()
        if total_volume > 0:
            vwap_60 = float((typical_price_60 * recent_volume).sum() / total_volume)
        else:
            vwap_60 = float(recent_close.mean())

        return {
            "method": "Technical",
            "fair_value_range": [round(key_support, 2), round(key_resistance, 2)],
            "key_support": round(key_support, 2),
            "key_resistance": round(key_resistance, 2),
            "volume_weighted_center": round(vwap_60, 2),
        }

    except Exception as e:
        logger.error(f"Technical fair value failed for {ticker}: {e}")
        return {"method": "Technical", "error": str(e)}


# ---------------------------------------------------------------------------
# 4. Synthesized Valuation
# ---------------------------------------------------------------------------

# Weight presets per trading style
_STYLE_WEIGHTS = {
    "value_investor": {"dcf": 0.50, "relative": 0.35, "technical": 0.15},
    "swing_trader":   {"dcf": 0.20, "relative": 0.30, "technical": 0.50},
    "day_trader":     {"dcf": 0.05, "relative": 0.15, "technical": 0.80},
}

_DEFAULT_WEIGHTS = {"dcf": 0.30, "relative": 0.35, "technical": 0.35}


def _get_style_weights() -> dict:
    """Load weights based on active trader profile, or use defaults."""
    try:
        from trader_profile import get_trading_style
        style = get_trading_style()
        weights = _STYLE_WEIGHTS.get(style, _DEFAULT_WEIGHTS)
        logger.info(f"Valuation weights from trader profile ({style}): {weights}")
        return weights
    except Exception:
        logger.info("Trader profile unavailable, using default valuation weights")
        return dict(_DEFAULT_WEIGHTS)


_valuation_cache = {}  # {ticker: (timestamp, result)}
_VALUATION_CACHE_TTL = 600  # 10 minutes


def synthesize_valuation(ticker: str) -> dict:
    """
    Run all three valuation methods and produce a weighted composite.

    Weights are determined by the active trader profile. Returns a
    comprehensive result with fair value range, verdict, and conviction.

    Results are cached for 10 minutes to avoid excessive yfinance API calls
    when called in a loop (e.g., per-ticker in process_raw_signals).
    """
    import time as _time

    # Check cache
    cached = _valuation_cache.get(ticker)
    if cached:
        cache_ts, cache_result = cached
        if _time.time() - cache_ts < _VALUATION_CACHE_TTL:
            return cache_result

    result = _synthesize_valuation_impl(ticker)

    # Store in cache
    _valuation_cache[ticker] = (_time.time(), result)
    return result


def _synthesize_valuation_impl(ticker: str) -> dict:
    """Internal implementation of synthesize_valuation (uncached)."""
    try:
        # Fetch current price
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            return {"ticker": ticker, "error": "Current price unavailable"}

        weights = _get_style_weights()

        # Run all three methods
        dcf_result = dcf_valuation(ticker)
        rel_result = relative_valuation(ticker)
        tech_result = technical_fair_value(ticker)

        methods = {
            "dcf": dcf_result,
            "relative": rel_result,
            "technical": tech_result,
        }

        # Collect valid fair values
        values_and_weights = []

        if "error" not in dcf_result:
            fv = dcf_result["fair_value_per_share"]
            values_and_weights.append((fv, fv, fv, weights["dcf"]))
            # For range: use bear/bull from sensitivity
            sens = dcf_result.get("sensitivity", {})
            bear = sens.get("bear_case", fv)
            bull = sens.get("bull_case", fv)
            values_and_weights[-1] = (bear, fv, bull, weights["dcf"])

        if "error" not in rel_result:
            fv = rel_result["fair_value_per_share"]
            # Use implied prices spread for range
            implied = rel_result.get("implied_prices", {})
            valid_implied = [v for v in implied.values() if v is not None and v > 0]
            low_rel = min(valid_implied) if valid_implied else fv
            high_rel = max(valid_implied) if valid_implied else fv
            values_and_weights.append((low_rel, fv, high_rel, weights["relative"]))

        if "error" not in tech_result:
            support = tech_result["key_support"]
            resistance = tech_result["key_resistance"]
            center = tech_result["volume_weighted_center"]
            values_and_weights.append((support, center, resistance, weights["technical"]))

        if not values_and_weights:
            return {
                "ticker": ticker,
                "current_price": current_price,
                "error": "All valuation methods failed",
                "methods": methods,
            }

        # Renormalize weights to sum to 1.0
        total_weight = sum(w for _, _, _, w in values_and_weights)
        if total_weight <= 0:
            total_weight = 1.0

        fair_value_low = sum(lo * w / total_weight for lo, _, _, w in values_and_weights)
        fair_value_mid = sum(mid * w / total_weight for _, mid, _, w in values_and_weights)
        fair_value_high = sum(hi * w / total_weight for _, _, hi, w in values_and_weights)

        upside_to_mid = ((fair_value_mid - current_price) / current_price) * 100

        # Verdict
        if upside_to_mid > 10:
            verdict = "undervalued"
        elif upside_to_mid < -10:
            verdict = "overvalued"
        else:
            verdict = "fairly_valued"

        # Conviction based on range tightness
        spread = fair_value_high - fair_value_low
        if fair_value_mid > 0:
            tightness = spread / fair_value_mid
        else:
            tightness = 1.0

        if tightness < 0.10:
            conviction = "high"
        elif tightness < 0.20:
            conviction = "moderate"
        else:
            conviction = "low"

        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "fair_value_low": round(fair_value_low, 2),
            "fair_value_mid": round(fair_value_mid, 2),
            "fair_value_high": round(fair_value_high, 2),
            "upside_to_mid": round(upside_to_mid, 2),
            "valuation_verdict": verdict,
            "conviction": conviction,
            "methods": methods,
        }

    except Exception as e:
        logger.error(f"Synthesized valuation failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}
