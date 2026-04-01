"""
FlowTrace Swarm Evolutionary Learning

Implements agent pruning and promotion:
- Worst-performing agents are deactivated
- Best-performing agents are cloned with slight mutation
- Archetype diversity is enforced after each cycle
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone

from swarm_config import (
    load_swarm_config,
    get_performance_tracking_params,
    get_anti_convergence_params,
    get_archetype_distribution,
    get_swarm_size,
)
from swarm_persona_generator import mutate_persona, save_personas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")


def _get_conn(db_path: str = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or KG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def evaluate_agents(db_path: str = None) -> list:
    """
    Rank all active swarm agents by composite score.

    Score = (win_rate * 0.6) + (reputation_score * 0.2) + (participation_quality * 0.2)

    participation_quality = normalized post count relative to peers.

    Returns sorted list of (agent_id, composite_score, persona_dict).
    """
    conn = _get_conn(db_path)

    agents = conn.execute("""
        SELECT agent_id, archetype, persona_name, persona_json,
               reputation_score, lifetime_wins, lifetime_losses,
               lifetime_trades, win_rate, is_active
        FROM swarm_agents
        WHERE is_active = 1
    """).fetchall()

    if not agents:
        conn.close()
        return []

    # Get post counts per agent (participation)
    post_counts = {}
    rows = conn.execute("""
        SELECT agent_id, COUNT(*) as cnt
        FROM swarm_posts
        GROUP BY agent_id
    """).fetchall()
    for r in rows:
        post_counts[r["agent_id"]] = r["cnt"]

    conn.close()

    max_posts = max(post_counts.values()) if post_counts else 1

    scored = []
    for agent in agents:
        agent_id = agent["agent_id"]
        win_rate = agent["win_rate"] or 0
        reputation = agent["reputation_score"] or 0.5
        posts = post_counts.get(agent_id, 0)
        participation = posts / max(1, max_posts)  # normalize to 0-1

        composite = (win_rate * 0.6) + (reputation * 0.2) + (participation * 0.2)

        try:
            persona = json.loads(agent["persona_json"])
        except (json.JSONDecodeError, TypeError):
            persona = {"agent_id": agent_id, "archetype": agent["archetype"],
                       "persona_name": agent["persona_name"]}

        # Sync DB stats into persona
        persona["lifetime_wins"] = agent["lifetime_wins"] or 0
        persona["lifetime_losses"] = agent["lifetime_losses"] or 0
        persona["lifetime_trades"] = agent["lifetime_trades"] or 0
        persona["win_rate"] = win_rate
        persona["reputation_score"] = reputation

        scored.append((agent_id, composite, persona))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def calculate_diversity_score(personas: list) -> float:
    """
    Measure archetype representation balance.
    0.0 = monoculture (all one type), 1.0 = perfectly balanced.
    Uses normalized Shannon entropy.
    """
    import math

    if not personas:
        return 0.0

    archetype_counts = {}
    for p in personas:
        arch = p.get("archetype", "unknown")
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

    total = sum(archetype_counts.values())
    n_types = len(archetype_counts)

    if n_types <= 1:
        return 0.0

    # Shannon entropy
    entropy = 0.0
    for count in archetype_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(n_types)
    return round(entropy / max_entropy, 3) if max_entropy > 0 else 0.0


def run_evolution_cycle(db_path: str = None) -> dict:
    """
    Execute one evolutionary cycle:
    1. Evaluate all agents
    2. Prune bottom performers
    3. Promote top performers (clone with mutation)
    4. Check and enforce diversity
    5. Persist updated personas

    Returns summary dict: {pruned: [...], promoted: [...], diversity_score: float}
    """
    params = get_performance_tracking_params()
    min_rounds = params.get("min_rounds_before_pruning", 100)
    prune_pct = params.get("prune_worst_pct", 0.05)
    promote_pct = params.get("promote_best_pct", 0.05)

    scored = evaluate_agents(db_path)

    if not scored:
        return {"pruned": [], "promoted": [], "diversity_score": 0}

    # Filter to agents with enough history for pruning
    eligible = [(aid, score, persona) for aid, score, persona in scored
                if persona.get("lifetime_trades", 0) >= min_rounds]

    # If not enough agents are eligible, skip evolution
    if len(eligible) < 10:
        all_personas = [p for _, _, p in scored]
        return {
            "pruned": [],
            "promoted": [],
            "diversity_score": calculate_diversity_score(all_personas),
        }

    # Determine prune and promote counts
    n_prune = max(1, int(len(eligible) * prune_pct))
    n_promote = max(1, int(len(eligible) * promote_pct))

    # Prune: deactivate bottom agents
    worst = eligible[-n_prune:]
    pruned_ids = []
    conn = _get_conn(db_path)

    for agent_id, score, persona in worst:
        conn.execute(
            "UPDATE swarm_agents SET is_active = 0 WHERE agent_id = ?",
            (agent_id,)
        )
        pruned_ids.append(agent_id)
        logger.debug(f"[Evolution] Pruned {persona.get('persona_name')} (score: {score:.3f})")

    # Promote: clone top agents with mutation
    best = eligible[:n_promote]
    promoted = []

    for agent_id, score, persona in best:
        clone = mutate_persona(persona, mutation_rate=0.1)

        # Insert clone into DB
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT OR IGNORE INTO swarm_agents
            (agent_id, archetype, persona_name, persona_json, reputation_score,
             lifetime_wins, lifetime_losses, lifetime_trades, win_rate,
             created_at, is_active)
            VALUES (?, ?, ?, ?, 0.5, 0, 0, 0, 0.0, ?, 1)
        """, (
            clone["agent_id"], clone["archetype"], clone["persona_name"],
            json.dumps(clone, default=str), now,
        ))
        promoted.append(clone)
        logger.debug(f"[Evolution] Promoted clone of {persona.get('persona_name')} -> {clone['persona_name']}")

    conn.commit()

    # Diversity check and repair
    all_active = conn.execute("""
        SELECT agent_id, archetype, persona_json
        FROM swarm_agents WHERE is_active = 1
    """).fetchall()
    conn.close()

    active_personas = []
    for r in all_active:
        try:
            p = json.loads(r["persona_json"])
        except (json.JSONDecodeError, TypeError):
            p = {"agent_id": r["agent_id"], "archetype": r["archetype"]}
        active_personas.append(p)

    diversity = calculate_diversity_score(active_personas)

    # If diversity is too low, enforce archetype balance
    anti_conv = get_anti_convergence_params()
    min_diversity = anti_conv.get("min_archetype_diversity", 0.6)

    if diversity < min_diversity:
        _repair_diversity(active_personas, db_path)
        # Recalculate after repair
        conn = _get_conn(db_path)
        all_active = conn.execute("""
            SELECT persona_json FROM swarm_agents WHERE is_active = 1
        """).fetchall()
        conn.close()
        active_personas = []
        for r in all_active:
            try:
                active_personas.append(json.loads(r["persona_json"]))
            except Exception:
                pass
        diversity = calculate_diversity_score(active_personas)

    # Save updated personas to disk
    save_personas(active_personas)

    return {
        "pruned": pruned_ids,
        "promoted": [p["agent_id"] for p in promoted],
        "diversity_score": diversity,
    }


def _repair_diversity(personas: list, db_path: str = None):
    """
    If certain archetypes are underrepresented, spawn new agents
    to rebalance.
    """
    config = load_swarm_config()
    target_dist = get_archetype_distribution(len(personas))

    # Count current archetypes
    current_counts = {}
    for p in personas:
        arch = p.get("archetype", "unknown")
        current_counts[arch] = current_counts.get(arch, 0) + 1

    conn = _get_conn(db_path)
    now = datetime.now(timezone.utc).isoformat()
    spawned = 0

    for arch_name, target_count in target_dist.items():
        current = current_counts.get(arch_name, 0)
        deficit = target_count - current

        if deficit <= 0:
            continue

        arch_config = config.get("archetypes", {}).get(arch_name, {})

        # Find a persona of this archetype to use as a template
        template = None
        for p in personas:
            if p.get("archetype") == arch_name:
                template = p
                break

        if template:
            for _ in range(min(deficit, 5)):  # Cap at 5 spawns per type per cycle
                clone = mutate_persona(template, mutation_rate=0.15)
                conn.execute("""
                    INSERT OR IGNORE INTO swarm_agents
                    (agent_id, archetype, persona_name, persona_json, reputation_score,
                     lifetime_wins, lifetime_losses, lifetime_trades, win_rate,
                     created_at, is_active)
                    VALUES (?, ?, ?, ?, 0.5, 0, 0, 0, 0.0, ?, 1)
                """, (
                    clone["agent_id"], clone["archetype"], clone["persona_name"],
                    json.dumps(clone, default=str), now,
                ))
                spawned += 1

    conn.commit()
    conn.close()

    if spawned > 0:
        logger.info(f"[Evolution] Diversity repair: spawned {spawned} agents for underrepresented archetypes")
