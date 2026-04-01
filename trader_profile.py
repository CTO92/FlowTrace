"""
FlowTrace Trader Profile System

One-time configuration where the trader declares their trading style.
Cascades through every system layer: indicator selection, timeframes,
fundamental vs technical weighting, swarm LLM budget, news relevance,
and signal processing intervals.

Four presets: value_investor, swing_multi_week, swing_single_week, day_trader.
All settings are overridable after preset selection.
"""

import os
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(BASE_DIR, "trader_profile.json")

_profile_cache = None


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "value_investor": {
        "trading_style": "value_investor",
        "swing_horizon": None,
        "analysis_timeframes": ["1d", "1wk", "1mo"],
        "primary_chart_timeframe": "1wk",
        "technical_indicator_set": [
            "SMA_200", "SMA_50", "MACD", "RSI", "OBV",
        ],
        "fundamental_weight": 0.8,
        "technical_weight": 0.2,
        "news_relevance_window_hours": 720,
        "signal_time_horizon_days": [10, 15, 20, 30, 60, 90],
        "preferred_holding_period_days": 90,
        "swarm_llm_recommendation": "auto",
        "description": (
            "Focus on fundamentals, long-term value. Technical indicators "
            "confirm entry timing only. Low LLM costs -- 'auto' mode provides "
            "excellent analysis given the longer time horizon."
        ),
    },
    "swing_multi_week": {
        "trading_style": "swing_trader",
        "swing_horizon": "multi_week",
        "analysis_timeframes": ["1h", "1d"],
        "primary_chart_timeframe": "1d",
        "technical_indicator_set": [
            "RSI", "MACD", "STOCH", "SMA", "EMA", "ADX",
            "OBV", "CMF", "ALLIGATOR", "AO",
        ],
        "fundamental_weight": 0.4,
        "technical_weight": 0.6,
        "news_relevance_window_hours": 168,
        "signal_time_horizon_days": [3, 5, 7, 10],
        "preferred_holding_period_days": 10,
        "swarm_llm_recommendation": "auto",
        "description": (
            "Balance of technical and fundamental. Moderate LLM costs -- "
            "'auto' mode works well."
        ),
    },
    "swing_single_week": {
        "trading_style": "swing_trader",
        "swing_horizon": "single_week",
        "analysis_timeframes": ["15m", "1h", "1d"],
        "primary_chart_timeframe": "1h",
        "technical_indicator_set": [
            "RSI", "MACD", "STOCH", "STOCH_RSI", "EMA", "SMA",
            "VWAP", "OBV", "MFI", "CCI", "SUPERTREND",
        ],
        "fundamental_weight": 0.2,
        "technical_weight": 0.8,
        "news_relevance_window_hours": 48,
        "signal_time_horizon_days": [1, 2, 3, 5],
        "preferred_holding_period_days": 5,
        "swarm_llm_recommendation": 50,
        "description": (
            "Technically driven. More frequent analysis needed. "
            "Consider increasing LLM budget for swarm."
        ),
    },
    "day_trader": {
        "trading_style": "day_trader",
        "swing_horizon": None,
        "analysis_timeframes": ["1m", "5m", "15m", "1h"],
        "primary_chart_timeframe": "5m",
        "technical_indicator_set": [
            "RSI", "STOCH", "MACD", "EMA", "VWAP", "OBV",
            "MFI", "SUPERTREND", "PSAR", "CCI", "WILLR",
            "ELDER", "FRACTALS",
        ],
        "fundamental_weight": 0.05,
        "technical_weight": 0.95,
        "news_relevance_window_hours": 4,
        "signal_time_horizon_days": [1],
        "preferred_holding_period_days": 1,
        "swarm_llm_recommendation": "all",
        "description": (
            "WARNING: Day trading requires maximum swarm intensity. Set LLM "
            "budget to 'all' or a high number (250+). This will significantly "
            "increase API costs but is necessary for the rapid analysis cycles "
            "day trading demands. Auto mode is NOT recommended -- by the time "
            "auto-mode produces consensus, the opportunity may have passed."
        ),
    },
}

# Signal freshness thresholds per style
FRESHNESS_THRESHOLDS = {
    "day_trader": {"fresh": 30, "aging": 120, "stale": 120},           # minutes
    "swing_single_week": {"fresh": 240, "aging": 720, "stale": 720},   # minutes
    "swing_multi_week": {"fresh": 720, "aging": 2880, "stale": 2880},  # minutes
    "value_investor": {"fresh": 2880, "aging": 10080, "stale": 10080}, # minutes
}


# ---------------------------------------------------------------------------
# Cost Warnings
# ---------------------------------------------------------------------------

COST_WARNINGS = {
    "value_investor": {
        "level": "low",
        "message": (
            "Value investing runs efficiently on 'auto' mode. The longer time "
            "horizon gives the swarm plenty of time to reach consensus with "
            "minimal LLM calls. Estimated cost: $1-5/day."
        ),
    },
    "swing_multi_week": {
        "level": "low_medium",
        "message": (
            "Multi-week swing trading works well on 'auto' mode. Moderate "
            "analysis frequency. Estimated cost: $3-10/day."
        ),
    },
    "swing_single_week": {
        "level": "medium",
        "message": (
            "Single-week swing trading benefits from a higher LLM budget "
            "(50+ calls per round). More frequent analysis cycles are needed "
            "to catch shorter-duration opportunities. Estimated cost: $8-20/day."
        ),
    },
    "day_trader": {
        "level": "high",
        "message": (
            "WARNING: Day trading requires maximum swarm intensity. Set LLM "
            "budget to 'all' or 250+ calls per round. This will significantly "
            "increase API costs ($15-50/day) but is necessary for the rapid "
            "analysis cycles day trading demands. On shorter time horizons, "
            "the swarm needs more LLM-driven agents to produce actionable "
            "consensus before the trading window closes. 'Auto' mode (10-15 "
            "calls) is designed for multi-day horizons and will be too slow "
            "for intraday."
        ),
    },
}


# ---------------------------------------------------------------------------
# Profile Loading / Saving
# ---------------------------------------------------------------------------

def _default_profile() -> dict:
    """Return default profile (swing multi-week)."""
    profile = dict(PRESETS["swing_multi_week"])
    profile["version"] = 1
    profile["created_at"] = None
    profile["updated_at"] = None
    return profile


def load_profile() -> dict:
    """Load trader profile from disk, or return default."""
    global _profile_cache
    if _profile_cache is not None:
        return _profile_cache

    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            _profile_cache = json.load(f)
    else:
        _profile_cache = _default_profile()

    return _profile_cache


def save_profile(profile: dict) -> None:
    """Persist trader profile to disk."""
    global _profile_cache
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    if not profile.get("created_at"):
        profile["created_at"] = profile["updated_at"]
    profile["version"] = 1
    _profile_cache = profile
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info(f"Trader profile saved: {profile.get('trading_style')}")


def reload_profile() -> dict:
    """Force reload from disk."""
    global _profile_cache
    _profile_cache = None
    return load_profile()


def profile_exists() -> bool:
    """Check if a trader profile has been configured."""
    return os.path.exists(PROFILE_FILE)


# ---------------------------------------------------------------------------
# Preset Application
# ---------------------------------------------------------------------------

def apply_preset(preset_name: str) -> dict:
    """
    Apply a preset and save. Returns the new profile.

    Valid presets: value_investor, swing_multi_week, swing_single_week, day_trader
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Valid: {list(PRESETS.keys())}")

    profile = dict(PRESETS[preset_name])
    profile["version"] = 1
    profile["created_at"] = datetime.now(timezone.utc).isoformat()
    save_profile(profile)

    # Cascade to other config files
    _cascade_profile(profile)

    return profile


def _cascade_profile(profile: dict) -> None:
    """
    Update other configuration files based on the trader profile.
    This is the integration cascade described in the upgrade plan.
    """
    # Update learning_config.json preferred_time_horizons
    try:
        from learning_config_manager import load_config, save_config
        config = load_config()
        config["preferred_time_horizons"] = profile.get(
            "signal_time_horizon_days", [1, 2, 3, 4, 5]
        )
        save_config(config)
    except Exception as e:
        logger.warning(f"Could not cascade to learning_config: {e}")

    # Update swarm_config.json llm_calls_per_round recommendation
    try:
        from swarm_config import load_swarm_config, save_swarm_config
        swarm_cfg = load_swarm_config()
        rec = profile.get("swarm_llm_recommendation", "auto")
        swarm_cfg["llm_calls_per_round"] = rec
        save_swarm_config(swarm_cfg)
    except Exception as e:
        logger.warning(f"Could not cascade to swarm_config: {e}")


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_trading_style() -> str:
    """Return the current trading style (value_investor, swing_trader, day_trader)."""
    return load_profile().get("trading_style", "swing_trader")


def get_preset_key() -> str:
    """
    Return the preset key for the current profile.
    Maps style + horizon to preset key.
    """
    profile = load_profile()
    style = profile.get("trading_style", "swing_trader")
    if style == "value_investor":
        return "value_investor"
    elif style == "day_trader":
        return "day_trader"
    elif style == "swing_trader":
        horizon = profile.get("swing_horizon", "multi_week")
        if horizon == "single_week":
            return "swing_single_week"
        else:
            return "swing_multi_week"
    return "swing_multi_week"


def get_analysis_timeframes() -> list:
    return load_profile().get("analysis_timeframes", ["1h", "1d"])


def get_primary_chart_timeframe() -> str:
    return load_profile().get("primary_chart_timeframe", "1d")


def get_technical_indicator_set() -> list:
    return load_profile().get("technical_indicator_set", ["RSI", "MACD", "SMA"])


def get_fundamental_weight() -> float:
    return load_profile().get("fundamental_weight", 0.4)


def get_technical_weight() -> float:
    return load_profile().get("technical_weight", 0.6)


def get_news_relevance_window_hours() -> int:
    return load_profile().get("news_relevance_window_hours", 168)


def get_signal_time_horizons() -> list:
    return load_profile().get("signal_time_horizon_days", [3, 5, 7, 10])


def get_preferred_holding_period() -> int:
    return load_profile().get("preferred_holding_period_days", 10)


def get_cost_warning() -> dict:
    """Return cost warning for the current profile."""
    key = get_preset_key()
    return COST_WARNINGS.get(key, COST_WARNINGS["swing_multi_week"])


def get_freshness_thresholds() -> dict:
    """
    Return freshness thresholds in minutes for the current profile.
    Returns {fresh: int, aging: int, stale: int} in minutes.
    """
    key = get_preset_key()
    return FRESHNESS_THRESHOLDS.get(key, FRESHNESS_THRESHOLDS["swing_multi_week"])


def get_swarm_archetype_weight_adjustments() -> dict:
    """
    Return archetype weight multipliers based on trading style.
    Day traders get more momentum/technical archetypes.
    Value investors get more fundamental/income archetypes.
    """
    style = get_trading_style()

    if style == "day_trader":
        return {
            "momentum_trader": 1.5,
            "technical_purist": 1.5,
            "sentiment_trader": 1.3,
            "quantitative": 1.2,
            "value_investor": 0.5,
            "income_focused": 0.3,
            "macro_strategist": 0.7,
        }
    elif style == "value_investor":
        return {
            "value_investor": 1.5,
            "income_focused": 1.5,
            "macro_strategist": 1.3,
            "risk_arbitrageur": 1.2,
            "momentum_trader": 0.5,
            "technical_purist": 0.5,
            "sentiment_trader": 0.7,
        }
    else:
        # Swing trader — balanced, slight technical lean
        return {
            "momentum_trader": 1.1,
            "technical_purist": 1.1,
            "event_driven": 1.2,
            "contrarian": 1.1,
        }
