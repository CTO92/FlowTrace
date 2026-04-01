"""
FlowTrace Trading Agent Persona Generator

Generates unique trading agent personas from archetype templates.
Each agent gets a distinct personality, risk tolerance, preferred indicators,
and behavioral quirks — derived from the archetype with controlled randomness.

For efficiency at scale (up to 10,000 agents):
- 1 LLM call per archetype generates a detailed philosophy (10 calls max)
- Individual agents are created via parametric variation (no LLM)
"""

import os
import json
import uuid
import random
import hashlib
import logging
from datetime import datetime, timezone

from swarm_config import load_swarm_config, get_archetype_distribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAS_FILE = os.path.join(BASE_DIR, "swarm_personas.json")

# Sector lists for random assignment
SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials", "Materials",
    "Real Estate", "Utilities", "Communication Services",
]


# ---------------------------------------------------------------------------
# LLM-Generated Archetype Philosophies
# ---------------------------------------------------------------------------

async def _generate_archetype_philosophy(archetype_name: str, archetype_config: dict) -> str:
    """
    Use the configured LLM to generate a unique trading philosophy for an archetype.
    Called once per archetype — the philosophy is then shared (with variation) across
    all agents of that archetype.
    """
    from llm_config import async_chat_completion, is_llm_configured

    if not is_llm_configured("SwarmPersonaGenerator"):
        return _fallback_philosophy(archetype_name, archetype_config)

    prompt = f"""Generate a brief (2-3 sentence) trading philosophy for an AI trading agent
with the following profile:

Archetype: {archetype_name.replace('_', ' ').title()}
Description: {archetype_config['description']}
Trading Bias: {archetype_config['bias']}
Preferred Time Horizon: {archetype_config['time_horizon_preference']} trading days
Key Indicators: {', '.join(archetype_config['indicators'])}

The philosophy should be first-person, opinionated, and specific about what market conditions
this agent thrives in and what it avoids. Make it sound like a distinct personality.

Output ONLY the philosophy text, no quotes or labels."""

    try:
        content = await async_chat_completion(
            messages=[
                {"role": "system", "content": "You are creating personas for autonomous trading agents in a simulation."},
                {"role": "user", "content": prompt},
            ],
            agent_type="SwarmPersonaGenerator",
            temperature=0.9,
        )
        return content.strip()
    except Exception as e:
        logger.warning(f"LLM persona generation failed for {archetype_name}: {e}")
        return _fallback_philosophy(archetype_name, archetype_config)


def _fallback_philosophy(archetype_name: str, archetype_config: dict) -> str:
    """Template-based philosophy when LLM is unavailable."""
    templates = {
        "value_investor": "I look for companies trading below intrinsic value with strong fundamentals. I am patient and willing to hold through volatility if the thesis remains intact. Margin of safety is everything.",
        "momentum_trader": "I follow the trend — price and volume tell me everything I need to know. I enter on breakouts with confirmation and ride momentum until exhaustion signals appear.",
        "contrarian": "I profit from crowd overreaction. When everyone is bullish, I look for cracks. When panic sets in, I look for opportunity. The crowd is usually wrong at extremes.",
        "event_driven": "I trade catalysts — earnings surprises, M&A announcements, regulatory changes. The event creates a temporary mispricing that I exploit before the market fully adjusts.",
        "macro_strategist": "I start from the top down — rates, inflation, geopolitics determine sector allocation before I touch individual names. The macro tide lifts or sinks all boats.",
        "quantitative": "I trust data over narrative. Statistical patterns, mean reversion, and factor models drive my decisions. If it can't be measured, it doesn't enter my process.",
        "sentiment_trader": "I read the crowd — social media buzz, retail positioning, news sentiment. When retail conviction is extreme, there's usually a trade, either with or against them.",
        "risk_arbitrageur": "I look for relative value — pairs trades, merger spreads, convergence plays. I don't bet on direction, I bet on the relationship between two assets.",
        "technical_purist": "The chart tells me everything. Support, resistance, patterns, candlesticks — price action reflects all known information. I trade what I see, not what I think.",
        "income_focused": "I seek stable cash flows — high dividend yields, covered call candidates, companies with predictable earnings. Capital preservation with income generation is my edge.",
    }
    return templates.get(archetype_name, f"I am a {archetype_name.replace('_', ' ')} focused on {archetype_config['description'].lower()}.")


# ---------------------------------------------------------------------------
# Parametric Persona Generation
# ---------------------------------------------------------------------------

def _random_in_range(base: float, variance: float) -> float:
    """Generate a value around base with given variance, clamped to [0, 1]."""
    return max(0.0, min(1.0, base + random.uniform(-variance, variance)))


def _generate_single_persona(
    archetype_name: str,
    archetype_config: dict,
    philosophy: str,
    index: int,
    variance: float,
    node_id: str,
) -> dict:
    """
    Generate a single agent persona with parametric variation from the archetype.
    No LLM call — this is pure computation.
    """
    agent_uuid = str(uuid.uuid4())[:8]
    prefix = archetype_name[:2].upper()
    persona_name = f"Agent-{prefix}-{index:04d}"
    agent_id = f"swarm:{node_id}:{archetype_name}:{agent_uuid}"

    # Base values per archetype bias
    bias_profiles = {
        "conservative":     {"risk": 0.3, "confidence_bias": -0.05, "contrarian": 0.10, "stubbornness": 0.7, "evidence_threshold": 0.6},
        "trend_following":  {"risk": 0.6, "confidence_bias": 0.05,  "contrarian": 0.10, "stubbornness": 0.4, "evidence_threshold": 0.3},
        "contrarian":       {"risk": 0.5, "confidence_bias": 0.00,  "contrarian": 0.80, "stubbornness": 0.6, "evidence_threshold": 0.5},
        "catalyst_focused": {"risk": 0.7, "confidence_bias": 0.10,  "contrarian": 0.15, "stubbornness": 0.3, "evidence_threshold": 0.4},
        "macro_first":      {"risk": 0.4, "confidence_bias": 0.00,  "contrarian": 0.20, "stubbornness": 0.5, "evidence_threshold": 0.5},
        "data_driven":      {"risk": 0.5, "confidence_bias": -0.02, "contrarian": 0.15, "stubbornness": 0.5, "evidence_threshold": 0.7},
        "crowd_reading":    {"risk": 0.6, "confidence_bias": 0.08,  "contrarian": 0.20, "stubbornness": 0.3, "evidence_threshold": 0.3},
        "market_neutral":   {"risk": 0.3, "confidence_bias": -0.05, "contrarian": 0.25, "stubbornness": 0.6, "evidence_threshold": 0.6},
        "chart_driven":     {"risk": 0.5, "confidence_bias": 0.03,  "contrarian": 0.10, "stubbornness": 0.5, "evidence_threshold": 0.5},
        "yield_seeking":    {"risk": 0.2, "confidence_bias": -0.08, "contrarian": 0.10, "stubbornness": 0.7, "evidence_threshold": 0.6},
    }

    bias = archetype_config.get("bias", "data_driven")
    profile = bias_profiles.get(bias, bias_profiles["data_driven"])

    # Assign 2-4 random sector preferences
    num_sectors = random.randint(2, 4)
    sector_prefs = random.sample(SECTORS, num_sectors)

    return {
        "agent_id": agent_id,
        "archetype": archetype_name,
        "persona_name": persona_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "philosophy": philosophy,
        "risk_tolerance": _random_in_range(profile["risk"], variance),
        "confidence_bias": round(profile["confidence_bias"] + random.uniform(-0.05, 0.05), 3),
        "contrarian_tendency": _random_in_range(profile["contrarian"], variance),
        "stubbornness": _random_in_range(profile["stubbornness"], variance),
        "evidence_threshold": _random_in_range(profile["evidence_threshold"], variance),
        "time_horizon_preference": archetype_config.get("time_horizon_preference", [1, 2, 3]),
        "preferred_indicators": archetype_config.get("indicators", []),
        "sector_preferences": sector_prefs,
        "activity_rate": _random_in_range(0.6, variance),
        "debate_engagement": _random_in_range(0.7, variance),
        "data_sharing_willingness": _random_in_range(0.8, variance),
        # Performance state (initialized)
        "lifetime_trades": 0,
        "lifetime_wins": 0,
        "lifetime_losses": 0,
        "win_rate": 0.0,
        "reputation_score": 0.5,
        "current_positions": {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_personas(swarm_size: int, archetypes: dict = None) -> list:
    """
    Generate a full set of trading agent personas.

    Strategy for efficiency:
    - 1 LLM call per archetype (max 10 calls) to generate philosophy
    - All individual agents created via parametric variation (no LLM)
    - Works for any swarm size from 5 to 10,000
    """
    if archetypes is None:
        config = load_swarm_config()
        archetypes = config.get("archetypes", {})

    variance = load_swarm_config().get("anti_convergence", {}).get("persona_variance", 0.3)

    # Get node_id for agent_id generation
    try:
        from node_identity import get_node_id
        node_id = get_node_id()
    except Exception:
        node_id = str(uuid.uuid4())[:8]

    # Step 1: Generate one philosophy per archetype via LLM (max ~10 calls)
    logger.info(f"Generating philosophies for {len(archetypes)} archetypes...")
    philosophies = {}
    for name, arch_config in archetypes.items():
        philosophies[name] = await _generate_archetype_philosophy(name, arch_config)
        logger.debug(f"  {name}: {philosophies[name][:60]}...")

    # Step 1.5: Adjust archetype weights based on trader profile
    try:
        from trader_profile import get_swarm_archetype_weight_adjustments
        adjustments = get_swarm_archetype_weight_adjustments()
        for arch_name, multiplier in adjustments.items():
            if arch_name in archetypes:
                archetypes[arch_name] = dict(archetypes[arch_name])
                archetypes[arch_name]["weight"] = archetypes[arch_name].get("weight", 0.1) * multiplier
        # Renormalize weights
        total = sum(a["weight"] for a in archetypes.values())
        for a in archetypes.values():
            a["weight"] = a["weight"] / total
    except ImportError:
        pass

    # Step 2: Distribute agents across archetypes
    distribution = get_archetype_distribution(swarm_size)
    logger.info(f"Archetype distribution for {swarm_size} agents: {distribution}")

    # Step 3: Generate personas via parametric variation
    personas = []
    global_index = 0

    for archetype_name, count in distribution.items():
        arch_config = archetypes.get(archetype_name, {})
        philosophy = philosophies.get(archetype_name, "")

        for i in range(count):
            persona = _generate_single_persona(
                archetype_name=archetype_name,
                archetype_config=arch_config,
                philosophy=philosophy,
                index=global_index,
                variance=variance,
                node_id=node_id,
            )
            personas.append(persona)
            global_index += 1

    logger.info(f"Generated {len(personas)} trading agent personas")
    return personas


def mutate_persona(persona: dict, mutation_rate: float = 0.1) -> dict:
    """
    Create a mutated clone of a persona for evolutionary promotion.
    Philosophy stays the same; numerical traits drift slightly.
    Returns a NEW persona dict (original unchanged).
    """
    try:
        from node_identity import get_node_id
        node_id = get_node_id()
    except Exception:
        node_id = str(uuid.uuid4())[:8]

    agent_uuid = str(uuid.uuid4())[:8]
    clone = dict(persona)
    clone["agent_id"] = f"swarm:{node_id}:{persona['archetype']}:{agent_uuid}"
    clone["persona_name"] = f"Agent-{persona['archetype'][:2].upper()}-M{random.randint(1000, 9999)}"
    clone["created_at"] = datetime.now(timezone.utc).isoformat()

    # Mutate numerical traits
    for key in ["risk_tolerance", "contrarian_tendency", "stubbornness",
                "evidence_threshold", "activity_rate", "debate_engagement",
                "data_sharing_willingness"]:
        if key in clone:
            old = clone[key]
            drift = random.uniform(-mutation_rate, mutation_rate)
            clone[key] = max(0.0, min(1.0, old + drift))

    clone["confidence_bias"] = round(
        clone.get("confidence_bias", 0) + random.uniform(-0.03, 0.03), 3
    )

    # Reset performance stats
    clone["lifetime_trades"] = 0
    clone["lifetime_wins"] = 0
    clone["lifetime_losses"] = 0
    clone["win_rate"] = 0.0
    clone["reputation_score"] = 0.5
    clone["current_positions"] = {}

    return clone


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_personas(personas: list) -> None:
    """Persist personas to disk."""
    with open(PERSONAS_FILE, "w") as f:
        json.dump(personas, f, indent=2, default=str)
    logger.info(f"Saved {len(personas)} personas to {PERSONAS_FILE}")


def load_personas() -> list:
    """Load personas from disk. Raises FileNotFoundError if not found."""
    if not os.path.exists(PERSONAS_FILE):
        raise FileNotFoundError(f"No personas file at {PERSONAS_FILE}")
    with open(PERSONAS_FILE, "r") as f:
        return json.load(f)


def personas_exist() -> bool:
    """Check if a personas file exists on disk."""
    return os.path.exists(PERSONAS_FILE)
