"""
FlowTrace Trade Plan Generator

Generates specific, actionable trade plans from consensus signals.
Takes in analysis data and produces entry/stop/targets/sizing/scaling
recommendations with full risk management context.
"""

import os
import math
import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _safe_get(data: Optional[dict], *keys, default=None):
    """Safely traverse nested dicts, returning default on any miss."""
    current = data
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is None:
            return default
    return current


def _fetch_current_price(ticker: str) -> Optional[float]:
    """Fetch current market price via yfinance."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("Could not fetch price for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# 1. Entry Strategy
# ---------------------------------------------------------------------------

def _build_entry_strategy(
    direction: str,
    current_price: float,
    technical_data: Optional[dict],
    catalyst_data: Optional[dict],
) -> dict:
    """Determine entry type, price, reasoning, and urgency."""
    entry_type = "market"
    entry_price = current_price
    reasoning = "No actionable technical level nearby; market order for immediate entry."
    urgency = "medium"

    try:
        support = _safe_get(technical_data, "support_level")
        resistance = _safe_get(technical_data, "resistance_level")
        macd_cross = _safe_get(technical_data, "macd_cross_bullish")

        if direction == "LONG" and support is not None:
            distance_pct = abs(current_price - support) / current_price
            if distance_pct <= 0.03:
                entry_type = "limit_at_support"
                entry_price = round(support, 2)
                reasoning = (
                    f"Price is within {distance_pct:.1%} of support at "
                    f"${support:.2f}; limit order at support for better fill."
                )

        if direction == "SHORT" and resistance is not None:
            distance_pct = abs(current_price - resistance) / current_price
            if distance_pct <= 0.03:
                entry_type = "limit_at_resistance"
                entry_price = round(resistance, 2)
                reasoning = (
                    f"Price is within {distance_pct:.1%} of resistance at "
                    f"${resistance:.2f}; limit order at resistance for better fill."
                )

        if macd_cross is True and entry_type == "market":
            entry_type = "breakout_confirmation"
            reasoning = "MACD just crossed bullish; enter on breakout confirmation above recent high."
    except Exception as exc:
        logger.debug("Entry strategy fallback to market: %s", exc)

    # Urgency based on catalyst proximity
    try:
        days_to_catalyst = _safe_get(catalyst_data, "days_to_next_event")
        if days_to_catalyst is not None:
            if days_to_catalyst <= 3:
                urgency = "high"
            elif days_to_catalyst <= 10:
                urgency = "medium"
            else:
                urgency = "low"
    except Exception:
        pass

    return {
        "type": entry_type,
        "price": round(entry_price, 2),
        "reasoning": reasoning,
        "urgency": urgency,
    }


# ---------------------------------------------------------------------------
# 2. Stop Loss
# ---------------------------------------------------------------------------

def _build_stop_loss(
    direction: str,
    entry_price: float,
    technical_data: Optional[dict],
) -> dict:
    """Calculate stop loss using ATR when available, else 5% default."""
    method = "percent_default"
    atr = _safe_get(technical_data, "atr")
    multiplier = 2.0

    try:
        if atr is not None and atr > 0:
            method = "atr_based"
            if direction == "LONG":
                stop_price = entry_price - (multiplier * atr)
            else:
                stop_price = entry_price + (multiplier * atr)
            reasoning = f"Stop set at {multiplier}x ATR (${atr:.2f}) from entry."
        else:
            raise ValueError("ATR unavailable")
    except Exception:
        pct = 0.05
        if direction == "LONG":
            stop_price = entry_price * (1 - pct)
        else:
            stop_price = entry_price * (1 + pct)
        reasoning = f"ATR unavailable; using {pct:.0%} default stop from entry."

    stop_price = round(stop_price, 2)
    risk_per_share = round(abs(entry_price - stop_price), 2)
    risk_pct = round(risk_per_share / entry_price * 100, 2) if entry_price and entry_price > 0 else 0.0

    return {
        "price": stop_price,
        "method": method,
        "risk_per_share": risk_per_share,
        "risk_pct": risk_pct,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# 3. Profit Targets
# ---------------------------------------------------------------------------

def _build_targets(
    direction: str,
    entry_price: float,
    risk_per_share: float,
    valuation_data: Optional[dict],
) -> list:
    """Build three profit targets from valuation data or percentage defaults."""
    fair_low = _safe_get(valuation_data, "fair_value_low")
    fair_mid = _safe_get(valuation_data, "fair_value_mid")
    fair_high = _safe_get(valuation_data, "fair_value_high")

    if direction == "LONG":
        t1_price = fair_low if fair_low is not None else round(entry_price * 1.03, 2)
        t2_price = fair_mid if fair_mid is not None else round(entry_price * 1.06, 2)
        t3_price = fair_high if fair_high is not None else round(entry_price * 1.10, 2)
    else:
        t1_price = fair_low if fair_low is not None else round(entry_price * 0.97, 2)
        t2_price = fair_mid if fair_mid is not None else round(entry_price * 0.94, 2)
        t3_price = fair_high if fair_high is not None else round(entry_price * 0.90, 2)

    def _rr(target_price: float) -> float:
        reward = abs(target_price - entry_price)
        if risk_per_share > 0:
            return round(reward / risk_per_share, 2)
        return 0.0

    targets = [
        {
            "label": "T1",
            "price": round(t1_price, 2),
            "reward_risk_ratio": _rr(t1_price),
            "reasoning": (
                f"Fair value low (${t1_price:.2f}) from valuation model."
                if fair_low is not None
                else f"Default +3% target at ${t1_price:.2f}."
            ),
            "action": "Take 33% profit",
        },
        {
            "label": "T2",
            "price": round(t2_price, 2),
            "reward_risk_ratio": _rr(t2_price),
            "reasoning": (
                f"Fair value mid (${t2_price:.2f}) from valuation model."
                if fair_mid is not None
                else f"Default +6% target at ${t2_price:.2f}."
            ),
            "action": "Take 33% profit",
        },
        {
            "label": "T3",
            "price": round(t3_price, 2),
            "reward_risk_ratio": _rr(t3_price),
            "reasoning": (
                f"Fair value high (${t3_price:.2f}) from valuation model."
                if fair_high is not None
                else f"Default +10% target at ${t3_price:.2f}."
            ),
            "action": "Take 33% profit",
        },
    ]

    return targets


# ---------------------------------------------------------------------------
# 4. Position Sizing
# ---------------------------------------------------------------------------

def _build_position_sizing(
    entry_price: float,
    risk_per_share: float,
    confidence: float,
    portfolio: Optional[dict],
    trader_profile: Optional[dict],
) -> dict:
    """Size the position based on risk budget and conviction level."""
    # Portfolio cash balance
    try:
        cash = _safe_get(portfolio, "cash_balance") or 100_000
    except Exception:
        cash = 100_000

    # Conviction-based risk budget
    if confidence >= 80:
        conviction = "high"
        risk_budget_pct = 2.0
    elif confidence >= 60:
        conviction = "moderate"
        risk_budget_pct = 1.5
    else:
        conviction = "low"
        risk_budget_pct = 0.75

    # Method selection
    win_rate = _safe_get(trader_profile, "win_rate")
    method = "fixed_risk"

    if conviction == "low":
        method = "equal_weight"
    elif win_rate is not None and confidence > 85:
        method = "half_kelly"

    # Position size calculation
    risk_budget_dollars = cash * (risk_budget_pct / 100.0)
    if risk_per_share > 0:
        position_size_shares = int(risk_budget_dollars / risk_per_share)
    else:
        position_size_shares = 0

    position_size_dollars = round(position_size_shares * entry_price, 2)
    position_pct = round((position_size_dollars / cash) * 100, 2) if cash > 0 else 0.0

    reasoning = (
        f"{conviction.capitalize()} conviction ({confidence:.0f}%); "
        f"risking {risk_budget_pct}% of ${cash:,.0f} portfolio "
        f"(${risk_budget_dollars:,.0f}) with ${risk_per_share:.2f}/share risk. "
        f"Method: {method}."
    )

    return {
        "risk_budget_pct": risk_budget_pct,
        "position_size_shares": position_size_shares,
        "position_size_dollars": position_size_dollars,
        "position_pct_of_portfolio": position_pct,
        "method": method,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# 5. Scaling Plan
# ---------------------------------------------------------------------------

def _build_scaling_plan(
    confidence: float,
    entry_price: float,
    technical_data: Optional[dict],
) -> dict:
    """Determine entry scaling strategy based on confidence."""
    support = _safe_get(technical_data, "support_level")

    if confidence > 80:
        plan = "enter_100"
        tranches = [
            {"pct": 100, "trigger": "immediate", "price": round(entry_price, 2)},
        ]
        reasoning = f"High confidence ({confidence:.0f}%); enter full position immediately."
    elif confidence >= 60:
        pullback_price = round(support, 2) if support else round(entry_price * 0.98, 2)
        plan = "enter_50_add_50"
        tranches = [
            {"pct": 50, "trigger": "immediate", "price": round(entry_price, 2)},
            {"pct": 50, "trigger": "pullback_to_support", "price": pullback_price},
        ]
        reasoning = (
            f"Moderate confidence ({confidence:.0f}%); enter 50% now, "
            f"add 50% on pullback to ${pullback_price:.2f}."
        )
    else:
        pullback_1 = round(entry_price * 0.98, 2)
        pullback_2 = round(entry_price * 0.96, 2)
        plan = "enter_33_add_33_add_33"
        tranches = [
            {"pct": 33, "trigger": "immediate", "price": round(entry_price, 2)},
            {"pct": 33, "trigger": "pullback_1", "price": pullback_1},
            {"pct": 34, "trigger": "pullback_2", "price": pullback_2},
        ]
        reasoning = (
            f"Low confidence ({confidence:.0f}%); scale in with three tranches "
            f"at ${entry_price:.2f}, ${pullback_1:.2f}, ${pullback_2:.2f}."
        )

    return {
        "plan": plan,
        "tranches": tranches,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# 6. Risk Management
# ---------------------------------------------------------------------------

def _build_risk_management(
    ticker: str,
    direction: str,
    portfolio: Optional[dict],
    catalyst_data: Optional[dict],
    consensus_signal: dict,
) -> dict:
    """Compile risk management checks: sector exposure, correlations, catalysts, thesis killers."""
    risk = {
        "sector_exposure": None,
        "correlated_positions": [],
        "catalyst_warning": None,
        "thesis_killers": [],
    }

    # Sector exposure check
    try:
        sector = _safe_get(consensus_signal, "sector")
        positions = _safe_get(portfolio, "positions") or []
        if sector and positions:
            same_sector = [
                p["ticker"] for p in positions
                if p.get("sector") == sector and p.get("ticker") != ticker
            ]
            if same_sector:
                risk["correlated_positions"] = same_sector
                risk["sector_exposure"] = (
                    f"Already holding {len(same_sector)} position(s) in {sector}: "
                    f"{', '.join(same_sector)}. Monitor aggregate sector risk."
                )
    except Exception as exc:
        logger.debug("Sector exposure check skipped: %s", exc)

    # Catalyst warnings
    try:
        next_event = _safe_get(catalyst_data, "next_event")
        days_to = _safe_get(catalyst_data, "days_to_next_event")
        if next_event and days_to is not None and days_to <= 7:
            risk["catalyst_warning"] = (
                f"Upcoming catalyst in {days_to} day(s): {next_event}. "
                f"Consider reducing size or waiting until after the event."
            )
    except Exception:
        pass

    # Thesis killers (template-based)
    try:
        reasoning = _safe_get(consensus_signal, "reasoning") or ""
        if direction == "LONG":
            risk["thesis_killers"] = [
                f"Price closes below stop loss level on increasing volume.",
                f"Material negative earnings revision or guidance cut for {ticker}.",
                f"Sector-wide selloff driven by macro deterioration (e.g., rate shock).",
            ]
        else:
            risk["thesis_killers"] = [
                f"Price closes above stop loss level on increasing volume.",
                f"Unexpected positive catalyst (earnings beat, buyout rumor) for {ticker}.",
                f"Broad market rally or sector rotation into {_safe_get(consensus_signal, 'sector') or 'this sector'}.",
            ]
    except Exception:
        risk["thesis_killers"] = [
            "Stop loss hit on above-average volume.",
            "Fundamental thesis invalidated by new data.",
            "Correlated macro event moves market against position.",
        ]

    return risk


# ---------------------------------------------------------------------------
# 7. Time Horizon
# ---------------------------------------------------------------------------

def _build_time_horizon(
    consensus_signal: dict,
    trader_profile: Optional[dict],
) -> dict:
    """Determine holding period from consensus signal or trader profile."""
    days = _safe_get(consensus_signal, "time_horizon_days")
    source = "consensus_signal"

    if days is None:
        days = _safe_get(trader_profile, "preferred_holding_period_days")
        source = "trader_profile"

    if days is None:
        days = 10
        source = "default"

    return {
        "days": days,
        "source": source,
        "reasoning": f"Holding period of {days} day(s) derived from {source}.",
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def generate_trade_plan(
    consensus_signal: dict,
    technical_data: dict = None,
    valuation_data: dict = None,
    portfolio: dict = None,
    catalyst_data: dict = None,
    trader_profile: dict = None,
) -> dict:
    """
    Generate a complete, actionable trade plan from a consensus signal.

    Parameters
    ----------
    consensus_signal : dict
        Must contain at minimum ``ticker`` and ``direction`` (LONG/SHORT).
        Optional keys: ``consensus_score``, ``adjusted_confidence``,
        ``time_horizon_days``, ``sector``, ``reasoning``.
    technical_data : dict, optional
        Keys: ``support_level``, ``resistance_level``, ``atr``,
        ``macd_cross_bullish``, ``current_price``.
    valuation_data : dict, optional
        Keys: ``fair_value_low``, ``fair_value_mid``, ``fair_value_high``.
    portfolio : dict, optional
        Keys: ``cash_balance``, ``positions`` (list of dicts with ticker/sector).
    catalyst_data : dict, optional
        Keys: ``next_event``, ``days_to_next_event``.
    trader_profile : dict, optional
        Keys: ``win_rate``, ``preferred_holding_period_days``.

    Returns
    -------
    dict
        Full trade plan with entry, stop_loss, targets, position_sizing,
        scaling, risk_management, time_horizon sections.
    """
    ticker = consensus_signal.get("ticker", "UNKNOWN")
    direction = consensus_signal.get("direction", "LONG").upper()
    confidence = consensus_signal.get("adjusted_confidence") or consensus_signal.get("consensus_score", 50)

    logger.info("Generating trade plan for %s (%s, confidence=%.1f%%)", ticker, direction, confidence)

    # Resolve current price
    current_price = _safe_get(technical_data, "current_price")
    if current_price is None:
        current_price = _fetch_current_price(ticker)
    if current_price is None:
        logger.error("Cannot determine current price for %s; aborting plan.", ticker)
        return {
            "error": f"Unable to fetch current price for {ticker}.",
            "ticker": ticker,
            "direction": direction,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # 1. Entry
    entry = _build_entry_strategy(direction, current_price, technical_data, catalyst_data)

    # 2. Stop Loss
    stop_loss = _build_stop_loss(direction, entry["price"], technical_data)

    # 3. Targets
    targets = _build_targets(direction, entry["price"], stop_loss["risk_per_share"], valuation_data)

    # 4. Position Sizing
    position_sizing = _build_position_sizing(
        entry["price"], stop_loss["risk_per_share"], confidence, portfolio, trader_profile,
    )

    # 5. Scaling
    scaling = _build_scaling_plan(confidence, entry["price"], technical_data)

    # 6. Risk Management
    risk_management = _build_risk_management(
        ticker, direction, portfolio, catalyst_data, consensus_signal,
    )

    # 7. Time Horizon
    time_horizon = _build_time_horizon(consensus_signal, trader_profile)

    plan = {
        "ticker": ticker,
        "direction": direction,
        "confidence": round(confidence, 2),
        "current_price": round(current_price, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry": entry,
        "stop_loss": stop_loss,
        "targets": targets,
        "position_sizing": position_sizing,
        "scaling": scaling,
        "risk_management": risk_management,
        "time_horizon": time_horizon,
    }

    logger.info(
        "Trade plan generated for %s: entry=%s @ $%.2f, stop=$%.2f, %d targets, %d shares",
        ticker, entry["type"], entry["price"], stop_loss["price"],
        len(targets), position_sizing["position_size_shares"],
    )

    return plan
