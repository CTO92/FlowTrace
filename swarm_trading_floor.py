"""
FlowTrace Trading Floor — Local Swarm Simulation Platform

A lightweight simulation engine (inspired by OASIS/CAMEL-AI) tailored for
trading agent debates.  Runs entirely in-memory with SQLite persistence.

Channels:
  #theses      — Agents post trade theses (ticker, direction, confidence)
  #challenges  — Counter-arguments against existing theses
  #evidence    — Supporting data findings
  #results     — Outcome reports (what happened, why right/wrong)
  #market-context — Macro observations, sector reads

Scale strategy (LLM calls per round):
  5-20 agents   → full LLM per agent
  21-100        → LLM for top-20% "leaders", rule-based for followers
  101-1,000     → LLM per archetype representative (~10), rule-based propagation
  1,001-10,000  → LLM per archetype (~10), statistical distribution
"""

import os
import json
import uuid
import random
import asyncio
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

from swarm_config import (
    load_swarm_config,
    get_anti_convergence_params,
    get_swarm_size,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DB_PATH = os.path.join(BASE_DIR, "knowledge_graph.db")

CHANNELS = ["theses", "challenges", "evidence", "results", "market-context"]
DIRECTIONS = ["BULLISH", "BEARISH", "NEUTRAL"]
POST_TYPES = {"theses": "THESIS", "challenges": "CHALLENGE", "evidence": "EVIDENCE",
              "results": "RESULT", "market-context": "CONTEXT"}


# ---------------------------------------------------------------------------
# Database Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS swarm_agents (
    agent_id TEXT PRIMARY KEY,
    archetype TEXT NOT NULL,
    persona_name TEXT NOT NULL,
    persona_json TEXT NOT NULL,
    reputation_score REAL DEFAULT 0.5,
    lifetime_wins INTEGER DEFAULT 0,
    lifetime_losses INTEGER DEFAULT 0,
    lifetime_trades INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_active_at TEXT,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS swarm_posts (
    post_id TEXT PRIMARY KEY,
    round_number INTEGER NOT NULL,
    channel TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    ticker TEXT,
    direction TEXT,
    confidence REAL,
    content TEXT NOT NULL,
    data_points TEXT,
    in_reply_to TEXT,
    reactions TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES swarm_agents(agent_id)
);

CREATE TABLE IF NOT EXISTS swarm_positions (
    position_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    entry_round INTEGER NOT NULL,
    entry_price REAL,
    exit_round INTEGER,
    exit_price REAL,
    outcome TEXT,
    actual_return_pct REAL,
    reasoning TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (agent_id) REFERENCES swarm_agents(agent_id)
);

CREATE TABLE IF NOT EXISTS swarm_rounds (
    round_number INTEGER PRIMARY KEY,
    cycle_number INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    active_agents INTEGER,
    posts_created INTEGER DEFAULT 0,
    positions_opened INTEGER DEFAULT 0,
    positions_resolved INTEGER DEFAULT 0,
    consensus_snapshot TEXT
);

CREATE INDEX IF NOT EXISTS idx_swarm_posts_round ON swarm_posts(round_number);
CREATE INDEX IF NOT EXISTS idx_swarm_posts_ticker ON swarm_posts(ticker);
CREATE INDEX IF NOT EXISTS idx_swarm_posts_agent ON swarm_posts(agent_id);
CREATE INDEX IF NOT EXISTS idx_swarm_positions_agent ON swarm_positions(agent_id);
CREATE INDEX IF NOT EXISTS idx_swarm_positions_ticker ON swarm_positions(ticker);
"""


# ---------------------------------------------------------------------------
# Rule-Based Agent Behavior (for non-LLM agents at scale)
# ---------------------------------------------------------------------------

def _rule_based_thesis(persona: dict, market_context: dict) -> Optional[dict]:
    """
    Generate a thesis post using rule-based logic + persona parameters.
    Returns a post dict or None if the agent decides not to post.
    """
    # Check activity rate — does this agent post this round?
    if random.random() > persona.get("activity_rate", 0.6):
        return None

    tickers = market_context.get("active_tickers", [])
    if not tickers:
        return None

    # Pick a ticker matching sector preferences or random
    preferred_sectors = persona.get("sector_preferences", [])
    sector_tickers = [
        t for t in tickers
        if market_context.get("ticker_sectors", {}).get(t) in preferred_sectors
    ]
    ticker = random.choice(sector_tickers) if sector_tickers else random.choice(tickers)

    # Determine direction based on archetype bias
    bias = persona.get("archetype", "")
    ticker_data = market_context.get("ticker_data", {}).get(ticker, {})
    change_pct = ticker_data.get("change_pct", 0)
    sentiment = ticker_data.get("sentiment", 0)

    if persona.get("contrarian_tendency", 0) > random.random():
        # Contrarian: go against recent movement
        direction = "BEARISH" if change_pct > 0 else "BULLISH"
    elif bias in ("momentum_trader", "trend_following"):
        direction = "BULLISH" if change_pct > 0.5 else "BEARISH" if change_pct < -0.5 else "NEUTRAL"
    elif bias in ("value_investor", "conservative"):
        direction = "BULLISH" if change_pct < -2 else "BEARISH" if change_pct > 5 else "NEUTRAL"
    elif bias in ("sentiment_trader", "crowd_reading"):
        direction = "BULLISH" if sentiment > 0.3 else "BEARISH" if sentiment < -0.3 else "NEUTRAL"
    else:
        direction = random.choice(["BULLISH", "BEARISH", "NEUTRAL"])

    # Confidence based on evidence threshold and risk tolerance
    base_confidence = 0.5 + persona.get("risk_tolerance", 0.5) * 0.3
    base_confidence += persona.get("confidence_bias", 0)
    confidence = max(0.1, min(0.99, base_confidence + random.uniform(-0.15, 0.15)))

    # Only post if confidence exceeds threshold
    if confidence < persona.get("evidence_threshold", 0.4):
        return None

    indicators = persona.get("preferred_indicators", [])
    indicator_str = ", ".join(indicators[:3]) if indicators else "general analysis"

    content = (
        f"[{persona['archetype'].replace('_', ' ').title()}] "
        f"{direction} on {ticker} (conf: {confidence:.0%}). "
        f"Based on {indicator_str}. "
        f"Recent move: {change_pct:+.1f}%."
    )

    return {
        "channel": "theses",
        "ticker": ticker,
        "direction": direction,
        "confidence": round(confidence, 3),
        "content": content,
        "data_points": json.dumps({"indicators": indicators, "change_pct": change_pct}),
    }


def _rule_based_challenge(persona: dict, thesis_post: dict, market_context: dict) -> Optional[dict]:
    """
    Generate a challenge to an existing thesis using rule-based logic.
    Returns a post dict or None.
    """
    if random.random() > persona.get("debate_engagement", 0.7):
        return None

    # Only challenge if we disagree
    our_bias = persona.get("archetype", "")
    thesis_direction = thesis_post.get("direction", "NEUTRAL")
    ticker = thesis_post.get("ticker")

    ticker_data = market_context.get("ticker_data", {}).get(ticker, {})
    change_pct = ticker_data.get("change_pct", 0)

    # Decide if we disagree
    disagree = False
    counter_direction = None

    if persona.get("contrarian_tendency", 0) > 0.5:
        # Contrarians challenge most theses
        disagree = random.random() < 0.7
        counter_direction = "BEARISH" if thesis_direction == "BULLISH" else "BULLISH"
    elif our_bias in ("value_investor", "conservative") and thesis_direction == "BULLISH" and change_pct > 3:
        disagree = True
        counter_direction = "BEARISH"
    elif our_bias in ("momentum_trader",) and thesis_direction == "BEARISH" and change_pct > 1:
        disagree = True
        counter_direction = "BULLISH"
    else:
        disagree = random.random() < 0.3  # Low base rate of disagreement
        counter_direction = "BEARISH" if thesis_direction == "BULLISH" else "BULLISH"

    if not disagree:
        return None

    confidence = max(0.1, min(0.99, 0.5 + random.uniform(-0.2, 0.2)))

    content = (
        f"[{persona['archetype'].replace('_', ' ').title()} challenges] "
        f"Disagree with {thesis_direction} thesis on {ticker}. "
        f"My read: {counter_direction} (conf: {confidence:.0%}). "
        f"The {persona.get('preferred_indicators', ['data'])[0]} suggests otherwise."
    )

    return {
        "channel": "challenges",
        "ticker": ticker,
        "direction": counter_direction,
        "confidence": round(confidence, 3),
        "content": content,
        "data_points": json.dumps({"responding_to": thesis_post.get("post_id")}),
        "in_reply_to": thesis_post.get("post_id"),
    }


def _rule_based_result(persona: dict, position: dict) -> Optional[dict]:
    """Generate a result-sharing post for a resolved position."""
    if random.random() > persona.get("data_sharing_willingness", 0.8):
        return None

    outcome = position.get("outcome", "NEUTRAL")
    ticker = position.get("ticker")
    direction = position.get("direction")
    actual = position.get("actual_return_pct", 0)

    content = (
        f"[Result] My {direction} call on {ticker}: {outcome}. "
        f"Actual return: {actual:+.1f}%. "
    )
    if outcome == "WIN":
        content += f"My {persona['archetype'].replace('_', ' ')} approach worked here."
    elif outcome == "LOSS":
        content += "Reviewing what I missed."

    return {
        "channel": "results",
        "ticker": ticker,
        "direction": direction,
        "confidence": position.get("confidence", 0.5),
        "content": content,
        "data_points": json.dumps({"outcome": outcome, "actual_return": actual}),
    }


# ---------------------------------------------------------------------------
# LLM-Driven Agent Behavior (for "leader" agents)
# ---------------------------------------------------------------------------

async def _llm_agent_action(
    persona: dict,
    recent_posts: list,
    market_context: dict,
    round_num: int,
) -> list:
    """
    Use the configured LLM to decide what a leader agent does this round.
    Returns a list of post dicts.
    """
    from llm_config import async_chat_completion, is_llm_configured

    if not is_llm_configured("SwarmAgent"):
        # Fall back to rule-based
        actions = []
        thesis = _rule_based_thesis(persona, market_context)
        if thesis:
            actions.append(thesis)
        return actions

    # Build context for the LLM
    recent_summary = ""
    for p in recent_posts[-10:]:
        recent_summary += f"- [{p.get('channel')}] {p.get('content', '')[:120]}\n"

    tickers_str = ", ".join(market_context.get("active_tickers", [])[:10])
    ticker_data_str = ""
    for t, data in list(market_context.get("ticker_data", {}).items())[:5]:
        ticker_data_str += f"  {t}: change={data.get('change_pct', 0):+.1f}%, sentiment={data.get('sentiment', 0):.2f}\n"

    positions_str = ""
    for ticker, pos in persona.get("current_positions", {}).items():
        positions_str += f"  {ticker}: {pos.get('direction')} (conf: {pos.get('confidence', 0):.0%})\n"

    prompt = f"""You are a trading agent on a simulated trading floor. Round #{round_num}.

YOUR PERSONA:
- Archetype: {persona['archetype'].replace('_', ' ').title()}
- Philosophy: {persona.get('philosophy', 'N/A')}
- Risk Tolerance: {persona.get('risk_tolerance', 0.5):.0%}
- Preferred Indicators: {', '.join(persona.get('preferred_indicators', []))}
- Time Horizon: {persona.get('time_horizon_preference', [1,2,3])} days

MARKET CONTEXT:
Active tickers: {tickers_str}
{ticker_data_str}

YOUR OPEN POSITIONS:
{positions_str if positions_str else "  None"}

RECENT TRADING FLOOR POSTS:
{recent_summary if recent_summary else "  (empty — be the first to post!)"}

Decide your actions. You may:
1. Post a THESIS on a ticker (with direction, confidence, reasoning)
2. CHALLENGE another agent's thesis (cite your data)
3. Share EVIDENCE you've found
4. Do NOTHING if you don't see an opportunity

Output STRICT JSON — an array of action objects:
[
  {{
    "action": "THESIS" | "CHALLENGE" | "EVIDENCE" | "NOTHING",
    "ticker": "AAPL",
    "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence": 0.75,
    "content": "Your reasoning (1-2 sentences)",
    "reply_to_content": "First few words of the post you're challenging (if CHALLENGE)"
  }}
]

If NOTHING, return: [{{"action": "NOTHING"}}]"""

    try:
        content = await async_chat_completion(
            messages=[
                {"role": "system", "content": "You are a trading agent in a multi-agent simulation. Be concise and data-driven."},
                {"role": "user", "content": prompt},
            ],
            agent_type="SwarmAgent",
            temperature=0.7,
            json_mode=True,
        )

        actions_raw = json.loads(content)
        if not isinstance(actions_raw, list):
            actions_raw = [actions_raw]

        posts = []
        for action in actions_raw:
            act_type = action.get("action", "NOTHING")
            if act_type == "NOTHING":
                continue

            channel = {
                "THESIS": "theses",
                "CHALLENGE": "challenges",
                "EVIDENCE": "evidence",
            }.get(act_type, "theses")

            # Try to find the post being challenged
            in_reply_to = None
            if act_type == "CHALLENGE" and action.get("reply_to_content"):
                search = action["reply_to_content"][:30].lower()
                for p in recent_posts:
                    if search in p.get("content", "").lower()[:50]:
                        in_reply_to = p.get("post_id")
                        break

            posts.append({
                "channel": channel,
                "ticker": action.get("ticker"),
                "direction": action.get("direction", "NEUTRAL"),
                "confidence": round(max(0.1, min(0.99, action.get("confidence", 0.5))), 3),
                "content": action.get("content", ""),
                "data_points": json.dumps({"llm_generated": True}),
                "in_reply_to": in_reply_to,
            })

        return posts

    except Exception as e:
        logger.warning(f"LLM agent action failed for {persona['persona_name']}: {e}")
        # Fallback to rule-based
        thesis = _rule_based_thesis(persona, market_context)
        return [thesis] if thesis else []


# ---------------------------------------------------------------------------
# TradingFloor Class
# ---------------------------------------------------------------------------

class TradingFloor:
    """
    In-memory simulation platform with SQLite persistence.
    Manages rounds, agent actions, and the post feed.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or KG_DB_PATH
        self.current_round = 0
        self.current_cycle = 0
        self._personas = []
        self._leader_ids = set()
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()

    async def initialize(self, personas: list):
        """Load or create swarm agents from personas."""
        self._personas = personas
        swarm_size = len(personas)

        # Determine leader agents (get LLM calls)
        self._leader_ids = self._select_leaders(personas, swarm_size)
        logger.info(
            f"[TradingFloor] Initialized with {swarm_size} agents, "
            f"{len(self._leader_ids)} leaders (LLM-driven)"
        )

        # Persist agents to DB
        conn = self._get_conn()
        for p in personas:
            conn.execute("""
                INSERT OR IGNORE INTO swarm_agents
                (agent_id, archetype, persona_name, persona_json, reputation_score,
                 lifetime_wins, lifetime_losses, lifetime_trades, win_rate,
                 created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                p["agent_id"], p["archetype"], p["persona_name"],
                json.dumps(p, default=str), p.get("reputation_score", 0.5),
                p.get("lifetime_wins", 0), p.get("lifetime_losses", 0),
                p.get("lifetime_trades", 0), p.get("win_rate", 0.0),
                p.get("created_at", datetime.now(timezone.utc).isoformat()),
            ))
        conn.commit()
        conn.close()

        # Resume round counter
        conn = self._get_conn()
        row = conn.execute("SELECT MAX(round_number) FROM swarm_rounds").fetchone()
        self.current_round = (row[0] or 0)
        conn.close()

    def _select_leaders(self, personas: list, swarm_size: int) -> set:
        """Select which agents get LLM calls based on swarm size tier."""
        if swarm_size <= 20:
            return {p["agent_id"] for p in personas}
        elif swarm_size <= 100:
            # Top 20% by reputation
            sorted_p = sorted(personas, key=lambda x: x.get("reputation_score", 0.5), reverse=True)
            n_leaders = max(10, swarm_size // 5)
            return {p["agent_id"] for p in sorted_p[:n_leaders]}
        else:
            # One representative per archetype
            leaders = set()
            by_arch = {}
            for p in personas:
                arch = p["archetype"]
                if arch not in by_arch:
                    by_arch[arch] = []
                by_arch[arch].append(p)
            for arch, agents in by_arch.items():
                best = max(agents, key=lambda x: x.get("reputation_score", 0.5))
                leaders.add(best["agent_id"])
            return leaders

    async def run_round(self, market_context: dict) -> dict:
        """
        Execute one simulation round.
        Returns a round summary dict.
        """
        self.current_round += 1
        round_num = self.current_round
        now = datetime.now(timezone.utc).isoformat()

        # Record round start
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO swarm_rounds (round_number, cycle_number, started_at, active_agents)
            VALUES (?, ?, ?, ?)
        """, (round_num, self.current_cycle, now, len(self._personas)))

        # Get recent posts for context (last 3 rounds)
        recent_posts = self._get_recent_posts(conn, round_num - 3, round_num - 1)

        all_posts = []
        positions_opened = 0

        # Phase 1: Leader agents act via LLM (parallelized)
        leader_personas = [p for p in self._personas if p["agent_id"] in self._leader_ids]
        if leader_personas:
            leader_tasks = [
                _llm_agent_action(p, recent_posts, market_context, round_num)
                for p in leader_personas
            ]
            leader_results = await asyncio.gather(*leader_tasks, return_exceptions=True)

            for persona, result in zip(leader_personas, leader_results):
                if isinstance(result, Exception):
                    logger.warning(f"Leader {persona['persona_name']} error: {result}")
                    continue
                for post_data in result:
                    if post_data:
                        post = self._create_post(post_data, persona["agent_id"], round_num)
                        all_posts.append(post)

        # Phase 2: Follower agents act via rule-based logic
        follower_personas = [p for p in self._personas if p["agent_id"] not in self._leader_ids]

        # Combine leader posts with recent for followers to react to
        round_posts_so_far = recent_posts + all_posts

        for persona in follower_personas:
            # Maybe post a thesis
            thesis = _rule_based_thesis(persona, market_context)
            if thesis:
                post = self._create_post(thesis, persona["agent_id"], round_num)
                all_posts.append(post)
                round_posts_so_far.append(post)

            # Maybe challenge a recent thesis
            thesis_posts = [p for p in round_posts_so_far if p.get("channel") == "theses"]
            if thesis_posts:
                target = random.choice(thesis_posts[-10:])  # Recent theses
                challenge = _rule_based_challenge(persona, target, market_context)
                if challenge:
                    post = self._create_post(challenge, persona["agent_id"], round_num)
                    all_posts.append(post)

        # Phase 3: Position management
        positions_opened = await self._manage_positions(all_posts, market_context, round_num)

        # Phase 4: Anti-convergence check
        anti_conv_posts = self._enforce_anti_convergence(all_posts, market_context, round_num)
        all_posts.extend(anti_conv_posts)

        # Persist all posts
        self._batch_insert_posts(conn, all_posts)

        # Build consensus snapshot
        consensus = self._build_consensus_snapshot(conn, round_num)

        # Update round record
        conn.execute("""
            UPDATE swarm_rounds SET completed_at = ?, posts_created = ?,
            positions_opened = ?, consensus_snapshot = ?
            WHERE round_number = ?
        """, (
            datetime.now(timezone.utc).isoformat(), len(all_posts),
            positions_opened, json.dumps(consensus), round_num,
        ))
        conn.commit()
        conn.close()

        # Collect active tickers from this round
        tickers_this_round = list({p.get("ticker") for p in all_posts if p.get("ticker")})

        return {
            "round_number": round_num,
            "posts_created": len(all_posts),
            "positions_opened": positions_opened,
            "active_agents": len(self._personas),
            "leader_count": len(self._leader_ids),
            "consensus": consensus,
            "tickers": tickers_this_round,
        }

    def _create_post(self, post_data: dict, agent_id: str, round_num: int) -> dict:
        """Create a post dict with generated ID and timestamp."""
        post = dict(post_data)
        post["post_id"] = str(uuid.uuid4())
        post["agent_id"] = agent_id
        post["round_number"] = round_num
        post["created_at"] = datetime.now(timezone.utc).isoformat()
        post.setdefault("reactions", "{}")
        post.setdefault("in_reply_to", None)
        return post

    def _batch_insert_posts(self, conn: sqlite3.Connection, posts: list):
        """Batch insert posts in a single transaction."""
        if not posts:
            return
        conn.executemany("""
            INSERT INTO swarm_posts
            (post_id, round_number, channel, agent_id, ticker, direction,
             confidence, content, data_points, in_reply_to, reactions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (p["post_id"], p["round_number"], p.get("channel", "theses"),
             p["agent_id"], p.get("ticker"), p.get("direction"),
             p.get("confidence"), p.get("content", ""),
             p.get("data_points"), p.get("in_reply_to"),
             p.get("reactions", "{}"), p["created_at"])
            for p in posts
        ])

    def _get_recent_posts(self, conn: sqlite3.Connection, from_round: int, to_round: int) -> list:
        """Fetch posts from a range of rounds."""
        rows = conn.execute("""
            SELECT * FROM swarm_posts
            WHERE round_number BETWEEN ? AND ?
            ORDER BY round_number DESC, created_at DESC
            LIMIT 100
        """, (max(0, from_round), to_round)).fetchall()
        return [dict(r) for r in rows]

    async def _manage_positions(self, posts: list, market_context: dict, round_num: int) -> int:
        """
        Open positions for high-confidence theses.
        Returns count of positions opened.
        """
        opened = 0
        conn = self._get_conn()

        for post in posts:
            if post.get("channel") != "theses":
                continue
            confidence = post.get("confidence", 0)
            if confidence < 0.6:
                continue

            ticker = post.get("ticker")
            if not ticker:
                continue

            agent_id = post.get("agent_id")

            # Check if agent already has a position on this ticker
            existing = conn.execute("""
                SELECT 1 FROM swarm_positions
                WHERE agent_id = ? AND ticker = ? AND outcome IS NULL
            """, (agent_id, ticker)).fetchone()

            if existing:
                continue

            # Get current price from market context
            entry_price = market_context.get("ticker_data", {}).get(ticker, {}).get("price")

            # Determine time horizon from persona
            persona = next((p for p in self._personas if p["agent_id"] == agent_id), None)
            horizons = persona.get("time_horizon_preference", [3]) if persona else [3]
            horizon = random.choice(horizons)

            conn.execute("""
                INSERT INTO swarm_positions
                (position_id, agent_id, ticker, direction, confidence,
                 entry_round, entry_price, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), agent_id, ticker,
                post.get("direction", "NEUTRAL"), confidence,
                round_num, entry_price, post.get("content", ""),
                datetime.now(timezone.utc).isoformat(),
            ))
            opened += 1

        conn.commit()
        conn.close()
        return opened

    async def resolve_positions(self, market_data: dict):
        """
        Resolve matured positions against current prices.
        market_data: {ticker: {"price": float}} — current prices.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        # Get open positions
        rows = conn.execute("""
            SELECT * FROM swarm_positions WHERE outcome IS NULL
        """).fetchall()

        resolved_count = 0
        result_posts = []

        for row in rows:
            position = dict(row)
            ticker = position["ticker"]
            entry_round = position["entry_round"]
            agent_id = position["agent_id"]

            # Determine if position has matured
            persona = next((p for p in self._personas if p["agent_id"] == agent_id), None)
            horizons = persona.get("time_horizon_preference", [3]) if persona else [3]
            max_horizon = max(horizons)

            # Each round ≈ some time. Use rounds as proxy for days.
            # We'll resolve after max_horizon * rounds_per_day rounds
            rounds_per_day = 10  # configurable
            maturity_rounds = max_horizon * rounds_per_day

            if (self.current_round - entry_round) < maturity_rounds:
                continue

            # Get current price
            current_price = market_data.get(ticker, {}).get("price")
            entry_price = position.get("entry_price")

            if current_price is None or entry_price is None or entry_price == 0:
                continue

            # Calculate return
            actual_return = ((current_price - entry_price) / entry_price) * 100
            direction = position["direction"]

            # Determine outcome
            if direction == "BULLISH":
                outcome = "WIN" if actual_return > 0 else "LOSS" if actual_return < -1 else "NEUTRAL"
            elif direction == "BEARISH":
                outcome = "WIN" if actual_return < 0 else "LOSS" if actual_return > 1 else "NEUTRAL"
            else:
                outcome = "NEUTRAL"

            # Update position
            conn.execute("""
                UPDATE swarm_positions
                SET exit_round = ?, exit_price = ?, outcome = ?,
                    actual_return_pct = ?, resolved_at = ?
                WHERE position_id = ?
            """, (
                self.current_round, current_price, outcome,
                round(actual_return, 2), now, position["position_id"],
            ))

            # Update agent stats
            if outcome == "WIN":
                conn.execute("""
                    UPDATE swarm_agents SET
                        lifetime_wins = lifetime_wins + 1,
                        lifetime_trades = lifetime_trades + 1,
                        win_rate = CAST(lifetime_wins + 1 AS REAL) / (lifetime_trades + 1),
                        reputation_score = MIN(1.0, reputation_score + 0.02),
                        last_active_at = ?
                    WHERE agent_id = ?
                """, (now, agent_id))
            elif outcome == "LOSS":
                conn.execute("""
                    UPDATE swarm_agents SET
                        lifetime_losses = lifetime_losses + 1,
                        lifetime_trades = lifetime_trades + 1,
                        win_rate = CAST(lifetime_wins AS REAL) / (lifetime_trades + 1),
                        reputation_score = MAX(0.0, reputation_score - 0.03),
                        last_active_at = ?
                    WHERE agent_id = ?
                """, (now, agent_id))
            else:
                conn.execute("""
                    UPDATE swarm_agents SET
                        lifetime_trades = lifetime_trades + 1,
                        win_rate = CAST(lifetime_wins AS REAL) / (lifetime_trades + 1),
                        last_active_at = ?
                    WHERE agent_id = ?
                """, (now, agent_id))

            # Update persona in memory too
            if persona:
                if outcome == "WIN":
                    persona["lifetime_wins"] = persona.get("lifetime_wins", 0) + 1
                    persona["reputation_score"] = min(1.0, persona.get("reputation_score", 0.5) + 0.02)
                elif outcome == "LOSS":
                    persona["lifetime_losses"] = persona.get("lifetime_losses", 0) + 1
                    persona["reputation_score"] = max(0.0, persona.get("reputation_score", 0.5) - 0.03)
                persona["lifetime_trades"] = persona.get("lifetime_trades", 0) + 1
                total = persona["lifetime_trades"]
                persona["win_rate"] = persona["lifetime_wins"] / total if total > 0 else 0

                # Generate result post
                result_post = _rule_based_result(persona, {
                    "outcome": outcome, "ticker": ticker,
                    "direction": direction, "actual_return_pct": actual_return,
                    "confidence": position["confidence"],
                })
                if result_post:
                    post = self._create_post(result_post, agent_id, self.current_round)
                    result_posts.append(post)

            resolved_count += 1

        # Batch insert result posts
        if result_posts:
            self._batch_insert_posts(conn, result_posts)

        conn.commit()
        conn.close()

        if resolved_count > 0:
            logger.info(f"[TradingFloor] Resolved {resolved_count} positions")

    def _enforce_anti_convergence(self, posts: list, market_context: dict, round_num: int) -> list:
        """
        Check for herding and inject dissent if needed.
        Returns additional posts to add.
        """
        params = get_anti_convergence_params()
        max_agree = params.get("max_agreement_ratio", 0.85)
        forced_contrarian_pct = params.get("forced_contrarian_pct", 0.10)
        mutation_rate = params.get("opinion_mutation_rate", 0.05)

        injected = []

        # Check per-ticker agreement
        ticker_votes = {}
        for p in posts:
            if p.get("channel") == "theses" and p.get("ticker"):
                ticker = p["ticker"]
                direction = p.get("direction", "NEUTRAL")
                if ticker not in ticker_votes:
                    ticker_votes[ticker] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
                ticker_votes[ticker][direction] = ticker_votes[ticker].get(direction, 0) + 1

        for ticker, votes in ticker_votes.items():
            total = sum(votes.values())
            if total < 3:
                continue
            max_direction = max(votes, key=votes.get)
            agree_ratio = votes[max_direction] / total

            if agree_ratio > max_agree:
                # Inject contrarian posts
                n_inject = max(1, int(total * forced_contrarian_pct))
                counter_dir = "BEARISH" if max_direction == "BULLISH" else "BULLISH"

                contrarian_personas = [
                    p for p in self._personas
                    if p.get("contrarian_tendency", 0) > 0.4
                ]
                if not contrarian_personas:
                    contrarian_personas = random.sample(
                        self._personas, min(n_inject, len(self._personas))
                    )

                for persona in contrarian_personas[:n_inject]:
                    content = (
                        f"[Anti-convergence dissent] The floor is {agree_ratio:.0%} {max_direction} "
                        f"on {ticker}. As a {persona['archetype'].replace('_', ' ')}, "
                        f"I see risks the crowd is ignoring. Going {counter_dir}."
                    )
                    post = self._create_post({
                        "channel": "challenges",
                        "ticker": ticker,
                        "direction": counter_dir,
                        "confidence": round(0.4 + random.uniform(0, 0.2), 3),
                        "content": content,
                        "data_points": json.dumps({"anti_convergence": True}),
                    }, persona["agent_id"], round_num)
                    injected.append(post)

        # Opinion mutation: random agents flip position
        for persona in self._personas:
            if random.random() < mutation_rate:
                if persona.get("current_positions"):
                    # Shift a random position
                    ticker = random.choice(list(persona["current_positions"].keys()))
                    old_dir = persona["current_positions"][ticker].get("direction", "NEUTRAL")
                    new_dir = random.choice([d for d in DIRECTIONS if d != old_dir])
                    persona["current_positions"][ticker]["direction"] = new_dir

        return injected

    def _build_consensus_snapshot(self, conn: sqlite3.Connection, round_num: int) -> dict:
        """Build per-ticker consensus from recent posts."""
        rows = conn.execute("""
            SELECT ticker, direction, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM swarm_posts
            WHERE round_number = ? AND channel IN ('theses', 'challenges') AND ticker IS NOT NULL
            GROUP BY ticker, direction
        """, (round_num,)).fetchall()

        consensus = {}
        for r in rows:
            ticker = r["ticker"]
            if ticker not in consensus:
                consensus[ticker] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0, "avg_confidence": {}}
            consensus[ticker][r["direction"]] = r["cnt"]
            consensus[ticker]["avg_confidence"][r["direction"]] = round(r["avg_conf"], 3)

        return consensus

    # ---------------------------------------------------------------------------
    # Query methods
    # ---------------------------------------------------------------------------

    def get_round_summary(self, round_number: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM swarm_rounds WHERE round_number = ?", (round_number,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_agent_leaderboard(self, top_n: int = 20) -> list:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT agent_id, archetype, persona_name, reputation_score,
                   lifetime_wins, lifetime_losses, lifetime_trades, win_rate
            FROM swarm_agents
            WHERE is_active = 1 AND lifetime_trades >= 5
            ORDER BY win_rate DESC, reputation_score DESC
            LIMIT ?
        """, (top_n,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_ticker_consensus(self, ticker: str) -> dict:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT direction, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM swarm_posts
            WHERE ticker = ? AND channel IN ('theses', 'challenges')
              AND round_number >= ?
            GROUP BY direction
        """, (ticker, max(0, self.current_round - 10))).fetchall()
        conn.close()

        result = {"ticker": ticker, "BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0,
                  "total": 0, "avg_confidence": {}}
        for r in rows:
            result[r["direction"]] = r["cnt"]
            result["total"] += r["cnt"]
            result["avg_confidence"][r["direction"]] = round(r["avg_conf"], 3)

        if result["total"] > 0:
            result["consensus_strength"] = max(
                result["BULLISH"], result["BEARISH"], result["NEUTRAL"]
            ) / result["total"]
            result["dominant_direction"] = max(
                ["BULLISH", "BEARISH", "NEUTRAL"],
                key=lambda d: result[d]
            )
        else:
            result["consensus_strength"] = 0
            result["dominant_direction"] = "NEUTRAL"

        return result

    def get_active_debates(self, ticker: str = None, limit: int = 20) -> list:
        conn = self._get_conn()
        query = """
            SELECT t.post_id, t.ticker, t.direction, t.confidence, t.content,
                   t.agent_id, t.round_number,
                   COUNT(c.post_id) as challenge_count
            FROM swarm_posts t
            LEFT JOIN swarm_posts c ON c.in_reply_to = t.post_id
            WHERE t.channel = 'theses'
        """
        params = []
        if ticker:
            query += " AND t.ticker = ?"
            params.append(ticker)
        query += """
            GROUP BY t.post_id
            HAVING challenge_count > 0
            ORDER BY challenge_count DESC, t.round_number DESC
            LIMIT ?
        """
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_archetype_performance(self) -> dict:
        """Win rate and trade count per archetype."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT archetype,
                   SUM(lifetime_wins) as wins,
                   SUM(lifetime_losses) as losses,
                   SUM(lifetime_trades) as trades,
                   AVG(reputation_score) as avg_reputation
            FROM swarm_agents
            WHERE is_active = 1
            GROUP BY archetype
        """).fetchall()
        conn.close()

        result = {}
        for r in rows:
            trades = r["trades"] or 0
            wins = r["wins"] or 0
            result[r["archetype"]] = {
                "wins": wins,
                "losses": r["losses"] or 0,
                "trades": trades,
                "win_rate": round(wins / trades, 3) if trades > 0 else 0,
                "avg_reputation": round(r["avg_reputation"], 3) if r["avg_reputation"] else 0.5,
            }
        return result
