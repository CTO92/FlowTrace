"""
FlowTrace Learning Configuration Manager

Manages the learning_config.json file that stores adaptive weights,
thresholds, and preferences learned from trade outcome history.
The LearningAgent writes to this config; ConsensusAgent reads from it.
"""

import os
import json
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "learning_config.json")

# Default configuration — used on first run or if config is missing
DEFAULT_CONFIG = {
    "version": 1,
    "updated_at": None,
    "participation_intensity": "medium",  # low | medium | high

    # Weight multipliers for each agent type's contribution to consensus scoring.
    # 1.0 = neutral. >1.0 = trust more. <1.0 = trust less.
    "agent_weights": {
        "ResearchAgent": 1.0,
        "MacroAgent": 1.0,
        "SentimentAgent": 1.0,
        "RiskManagerAgent": 1.0,
        "StrategyAgent": 1.0,
        "ScoutAgent": 1.0,
        "TechnicalAgent": 1.0,
        "ValidationAgent": 1.0,
        "NewsSentimentAgent": 1.0,
        "PortfolioOptimizerAgent": 1.0,
        "SectorRotationAgent": 1.0,
        "VolatilityAgent": 1.0,
        "CorrelationMatrixAgent": 1.0,
        "SeasonalityAgent": 1.0,
        "FundamentalAgent": 1.0,
        "EarningsAgent": 1.0,
        "ShortInterestAgent": 1.0,
        "NewsAggregatorAgent": 1.0,
        "SupplyChainVisualizerAgent": 1.0,
        "SECFilingsAgent": 1.0,
        "PeerComparisonAgent": 1.0,
    },

    # Trust scores for remote nodes on the AgentForum (node_alias -> score)
    "network_node_trust": {},

    # Per-sector confidence adjustments (GICS sector -> multiplier)
    "sector_confidence_adjustments": {
        "Technology": 1.0,
        "Healthcare": 1.0,
        "Financials": 1.0,
        "Consumer Discretionary": 1.0,
        "Consumer Staples": 1.0,
        "Energy": 1.0,
        "Industrials": 1.0,
        "Materials": 1.0,
        "Utilities": 1.0,
        "Real Estate": 1.0,
        "Communication Services": 1.0,
    },

    # Preferred time horizons (1-5 trading days) based on historical win rate
    "preferred_time_horizons": [1, 2, 3, 4, 5],

    # Minimum confidence to publish a thesis to AgentForum
    "min_publish_confidence": 0.75,

    # Minimum consensus score to emit a signal to the trader
    "min_signal_consensus": 0.65,

    # Per event-type weight multipliers
    "event_type_weights": {
        "Earnings": 1.0,
        "Contract": 1.0,
        "Merger": 1.0,
        "Partnership": 1.0,
        "Regulatory": 1.0,
        "Macro": 1.0,
        "Product Launch": 1.0,
        "Scandal": 1.0,
    },

    # Market regime adjustments
    "regime_adjustments": {
        "low_vix": {"confidence_multiplier": 1.0},
        "normal_vix": {"confidence_multiplier": 1.0},
        "high_vix": {"confidence_multiplier": 1.0},
    },

    # Performance tracking (updated by LearningAgent)
    "performance_history": {
        "total_signals": 0,
        "wins": 0,
        "losses": 0,
        "neutral": 0,
        "avg_return": 0.0,
        "win_rate": 0.0,
        "last_review": None,
    },
}


def load_config() -> dict:
    """Load learning config from disk, or create defaults if missing."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        # Merge with defaults to pick up any new keys added in updates
        merged = _deep_merge(DEFAULT_CONFIG, config)
        return merged
    else:
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)


def save_config(config: dict) -> None:
    """Persist learning config to disk."""
    config["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_agent_weight(agent_type: str) -> float:
    """Get the trust weight for a specific agent type."""
    config = load_config()
    return config["agent_weights"].get(agent_type, 1.0)


def get_participation_intensity() -> str:
    """Get the current participation intensity setting."""
    config = load_config()
    return config.get("participation_intensity", "medium")


def get_intensity_thresholds() -> dict:
    """Return thresholds based on participation intensity level."""
    intensity = get_participation_intensity()

    thresholds = {
        "low": {
            "min_publish_confidence": 0.90,
            "forum_scan_interval": 300,   # seconds
            "debate_engagement": "minimal",  # only respond to direct challenges
            "max_theses_per_day": 3,
        },
        "medium": {
            "min_publish_confidence": 0.75,
            "forum_scan_interval": 60,
            "debate_engagement": "moderate",  # engage on watchlist tickers
            "max_theses_per_day": 10,
        },
        "high": {
            "min_publish_confidence": 0.65,
            "forum_scan_interval": 15,
            "debate_engagement": "aggressive",  # engage broadly
            "max_theses_per_day": 50,
        },
    }

    return thresholds.get(intensity, thresholds["medium"])


def update_agent_weight(agent_type: str, new_weight: float) -> None:
    """Update the trust weight for a specific agent type."""
    config = load_config()
    new_weight = max(0.1, min(2.0, new_weight))  # clamp to [0.1, 2.0]
    config["agent_weights"][agent_type] = round(new_weight, 3)
    save_config(config)


def update_performance(wins: int, losses: int, neutral: int, avg_return: float) -> None:
    """Update the performance history counters."""
    config = load_config()
    perf = config["performance_history"]
    perf["total_signals"] = wins + losses + neutral
    perf["wins"] = wins
    perf["losses"] = losses
    perf["neutral"] = neutral
    perf["avg_return"] = round(avg_return, 4)
    perf["win_rate"] = round(wins / max(1, wins + losses), 4)
    perf["last_review"] = datetime.now(timezone.utc).isoformat()
    save_config(config)


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults, keeping new default keys."""
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
