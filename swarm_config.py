"""
FlowTrace Swarm Configuration Manager

Manages the Trading Agent Swarm configuration: swarm size, archetype
distribution, simulation parameters, anti-convergence settings, and
evolutionary tracking.

Configuration persisted to swarm_config.json.
"""

import os
import json
import math
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "swarm_config.json")

_config_cache = None


# ---------------------------------------------------------------------------
# Configuration Loading / Saving
# ---------------------------------------------------------------------------

def _defaults() -> dict:
    """Return the default swarm configuration."""
    return {
        "version": 1,
        "updated_at": None,
        "enabled": False,
        "swarm_size": 20,
        "simulation_speed": "normal",
        "round_interval_seconds": 30,
        "max_rounds_per_cycle": 50,
        "llm_calls_per_round": "auto",  # "auto" = tiered by swarm size, or an int (e.g. 50, 500, "all")
        "archetypes": {
            "value_investor":   {"weight": 0.15, "description": "Focuses on fundamentals, P/E, book value, margin of safety", "bias": "conservative", "time_horizon_preference": [3, 4, 5], "indicators": ["P/E", "P/B", "FCF", "debt_ratio"]},
            "momentum_trader":  {"weight": 0.15, "description": "Follows price trends, breakouts, volume surges", "bias": "trend_following", "time_horizon_preference": [1, 2, 3], "indicators": ["RSI", "MACD", "volume", "moving_averages"]},
            "contrarian":       {"weight": 0.10, "description": "Bets against crowd consensus, looks for overreactions", "bias": "contrarian", "time_horizon_preference": [2, 3, 4], "indicators": ["short_interest", "put_call_ratio", "sentiment_extremes"]},
            "event_driven":     {"weight": 0.15, "description": "Trades around earnings, M&A, regulatory events", "bias": "catalyst_focused", "time_horizon_preference": [1, 2], "indicators": ["earnings_dates", "M&A_news", "regulatory_filings"]},
            "macro_strategist": {"weight": 0.10, "description": "Top-down analysis, rates, currencies, sector rotation", "bias": "macro_first", "time_horizon_preference": [3, 4, 5], "indicators": ["fed_rates", "yield_curve", "sector_etf_flows"]},
            "quantitative":     {"weight": 0.10, "description": "Statistical patterns, mean reversion, factor models", "bias": "data_driven", "time_horizon_preference": [1, 2, 3], "indicators": ["z_score", "correlation", "beta", "volatility"]},
            "sentiment_trader": {"weight": 0.10, "description": "Social media buzz, retail flow, news sentiment", "bias": "crowd_reading", "time_horizon_preference": [1, 2], "indicators": ["social_volume", "news_sentiment", "retail_flow"]},
            "risk_arbitrageur": {"weight": 0.05, "description": "Relative value, pairs trades, merger arb", "bias": "market_neutral", "time_horizon_preference": [2, 3, 4], "indicators": ["spread", "correlation", "implied_probability"]},
            "technical_purist": {"weight": 0.05, "description": "Chart patterns only, support/resistance, candlesticks", "bias": "chart_driven", "time_horizon_preference": [1, 2, 3], "indicators": ["chart_patterns", "support_resistance", "fibonacci"]},
            "income_focused":   {"weight": 0.05, "description": "Dividend yield, covered calls, stable cash flow", "bias": "yield_seeking", "time_horizon_preference": [4, 5], "indicators": ["dividend_yield", "payout_ratio", "FCF_yield"]},
        },
        "anti_convergence": {
            "min_archetype_diversity": 0.6,
            "forced_contrarian_pct": 0.10,
            "opinion_mutation_rate": 0.05,
            "max_agreement_ratio": 0.85,
            "persona_variance": 0.3,
        },
        "performance_tracking": {
            "track_per_agent": True,
            "track_per_archetype": True,
            "prune_worst_pct": 0.05,
            "promote_best_pct": 0.05,
            "min_rounds_before_pruning": 100,
        },
    }


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *defaults*."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_swarm_config() -> dict:
    """Load and cache the swarm configuration."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    defaults = _defaults()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            disk = json.load(f)
        _config_cache = _deep_merge(defaults, disk)
    else:
        _config_cache = defaults

    return _config_cache


def save_swarm_config(config: dict) -> None:
    """Persist swarm configuration to disk."""
    global _config_cache
    config["updated_at"] = datetime.now(timezone.utc).isoformat()
    _config_cache = config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def reload_swarm_config() -> dict:
    """Force reload from disk (clears cache)."""
    global _config_cache
    _config_cache = None
    return load_swarm_config()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_swarm_size() -> int:
    config = load_swarm_config()
    return max(5, min(10000, config.get("swarm_size", 20)))


def is_swarm_enabled() -> bool:
    return load_swarm_config().get("enabled", False)


def get_archetype_distribution(swarm_size: int = None) -> dict:
    """
    Return a mapping of archetype name -> agent count for the given swarm size.
    Distributes agents proportionally by weight, ensuring at least 1 per type
    when swarm_size >= number of archetypes.
    """
    if swarm_size is None:
        swarm_size = get_swarm_size()

    config = load_swarm_config()
    archetypes = config.get("archetypes", {})

    if swarm_size < len(archetypes):
        # Fewer agents than archetypes: pick top-N by weight
        sorted_types = sorted(archetypes.items(), key=lambda x: x[1]["weight"], reverse=True)
        dist = {}
        for name, _ in sorted_types[:swarm_size]:
            dist[name] = 1
        return dist

    # Proportional distribution
    total_weight = sum(a["weight"] for a in archetypes.values())
    dist = {}
    allocated = 0

    for name, arch in archetypes.items():
        count = max(1, round((arch["weight"] / total_weight) * swarm_size))
        dist[name] = count
        allocated += count

    # Adjust rounding errors
    diff = swarm_size - allocated
    if diff != 0:
        # Add/remove from the largest archetype
        largest = max(dist, key=dist.get)
        dist[largest] = max(1, dist[largest] + diff)

    return dist


def get_anti_convergence_params() -> dict:
    config = load_swarm_config()
    return config.get("anti_convergence", _defaults()["anti_convergence"])


def get_simulation_params() -> dict:
    """Return simulation-related parameters, respecting participation intensity."""
    config = load_swarm_config()

    # Check if intensity overrides exist from learning_config
    try:
        from learning_config_manager import get_intensity_thresholds
        thresholds = get_intensity_thresholds()
        return {
            "round_interval_seconds": thresholds.get(
                "swarm_round_interval",
                config.get("round_interval_seconds", 30)
            ),
            "max_rounds_per_cycle": thresholds.get(
                "swarm_max_rounds_per_cycle",
                config.get("max_rounds_per_cycle", 50)
            ),
            "simulation_speed": config.get("simulation_speed", "normal"),
        }
    except ImportError:
        pass

    return {
        "round_interval_seconds": config.get("round_interval_seconds", 30),
        "max_rounds_per_cycle": config.get("max_rounds_per_cycle", 50),
        "simulation_speed": config.get("simulation_speed", "normal"),
    }


def get_llm_calls_per_round(swarm_size: int = None) -> int:
    """
    Return the number of LLM-driven agents per simulation round.

    Configurable via llm_calls_per_round in swarm_config.json:
      - "auto" (default): Tiered by swarm size (efficient defaults)
      - "all": Every agent gets an LLM call (maximum quality, maximum cost)
      - integer (e.g. 50): Exact number of LLM-driven agents per round

    A larger budget trader can increase this to improve swarm intelligence
    at the cost of higher token expenditure.
    """
    if swarm_size is None:
        swarm_size = get_swarm_size()

    config = load_swarm_config()
    setting = config.get("llm_calls_per_round", "auto")

    if setting == "all":
        return swarm_size

    if isinstance(setting, int) and setting > 0:
        return min(setting, swarm_size)

    if isinstance(setting, str) and setting.isdigit():
        return min(int(setting), swarm_size)

    # "auto" — tiered strategy
    if swarm_size <= 20:
        return swarm_size
    elif swarm_size <= 100:
        return max(10, swarm_size // 5)
    else:
        # 1 per archetype + a few extra high-reputation agents
        n_archetypes = len(config.get("archetypes", {}))
        return max(n_archetypes, 15)


def get_performance_tracking_params() -> dict:
    config = load_swarm_config()
    return config.get("performance_tracking", _defaults()["performance_tracking"])
