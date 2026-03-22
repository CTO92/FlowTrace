"""
Node and agent reputation scoring service for FlowTrace AgentForum.

Provides reputation calculation, decay for inactivity, participation
requirement checks, and a leaderboard query.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from platform.models import Node, AgentScore, Post


# ---------------------------------------------------------------------------
# Reputation calculation
# ---------------------------------------------------------------------------

async def update_node_reputation(db: AsyncSession, node_id: UUID) -> None:
    """Recalculate reputation for a single node.

    Reputation is the weighted average of all agent_scores for the node,
    with more recent theses weighted higher.  The weighting uses an
    exponential decay: weight = 0.95 ** weeks_ago, so a score from this
    week counts fully while one from ~14 weeks ago counts about half.
    """
    now = datetime.utcnow()

    result = await db.execute(
        select(AgentScore).where(AgentScore.node_id == node_id)
    )
    scores = result.scalars().all()

    if not scores:
        return  # Nothing to update

    total_weight = 0.0
    weighted_sum = 0.0

    for score in scores:
        weeks_ago = max((now - score.scored_at).days / 7.0, 0.0)
        weight = 0.95 ** weeks_ago
        weighted_sum += score.score_delta * weight
        total_weight += weight

    reputation = weighted_sum / total_weight if total_weight > 0 else 0.0

    node_result = await db.execute(
        select(Node).where(Node.id == node_id)
    )
    node = node_result.scalar_one_or_none()
    if node is not None:
        node.reputation = reputation

    await db.flush()


# ---------------------------------------------------------------------------
# Inactivity decay
# ---------------------------------------------------------------------------

async def decay_inactive_reputations(db: AsyncSession) -> None:
    """Apply reputation decay to nodes that have not posted in 14+ days.

    Decay rate: multiply by 0.99 for each day of inactivity beyond the
    14-day grace period, with a floor of 0.1.
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(days=14)

    # Subquery: latest post date per node
    latest_post_sq = (
        select(
            Post.node_id,
            func.max(Post.created_at).label("last_post"),
        )
        .group_by(Post.node_id)
        .subquery()
    )

    result = await db.execute(
        select(Node, latest_post_sq.c.last_post)
        .outerjoin(latest_post_sq, Node.id == latest_post_sq.c.node_id)
        .where(
            (latest_post_sq.c.last_post < cutoff)
            | (latest_post_sq.c.last_post.is_(None))
        )
    )

    rows = result.all()

    for node, last_post in rows:
        if last_post is None:
            inactive_days = (now - node.created_at).days - 14
        else:
            inactive_days = (now - last_post).days - 14

        if inactive_days <= 0:
            continue

        decay_factor = 0.99 ** inactive_days
        new_reputation = max(node.reputation * decay_factor, 0.1)
        node.reputation = new_reputation

    await db.flush()


# ---------------------------------------------------------------------------
# Participation requirements
# ---------------------------------------------------------------------------

async def check_participation_requirements(db: AsyncSession) -> list[dict]:
    """Return active nodes that have not posted in 30+ days, flagged for
    potential suspension."""
    now = datetime.utcnow()
    cutoff = now - timedelta(days=30)

    latest_post_sq = (
        select(
            Post.node_id,
            func.max(Post.created_at).label("last_post"),
        )
        .group_by(Post.node_id)
        .subquery()
    )

    result = await db.execute(
        select(Node, latest_post_sq.c.last_post)
        .outerjoin(latest_post_sq, Node.id == latest_post_sq.c.node_id)
        .where(
            Node.status == "active",
            (latest_post_sq.c.last_post < cutoff)
            | (latest_post_sq.c.last_post.is_(None)),
        )
    )

    flagged: list[dict] = []
    for node, last_post in result.all():
        days_inactive = (
            (now - last_post).days if last_post else (now - node.created_at).days
        )
        flagged.append(
            {
                "node_id": str(node.id),
                "node_name": node.name,
                "reputation": node.reputation,
                "last_post": last_post.isoformat() if last_post else None,
                "days_inactive": days_inactive,
            }
        )

    return flagged


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

async def get_leaderboard(db: AsyncSession, limit: int = 50) -> list[dict]:
    """Return the top *limit* nodes ordered by reputation, along with
    summary statistics."""
    # Count of scored threads per node
    score_count_sq = (
        select(
            AgentScore.node_id,
            func.count(AgentScore.id).label("total_scores"),
            func.sum(
                func.case(
                    (AgentScore.outcome == "WIN", 1),
                    else_=0,
                )
            ).label("wins"),
            func.sum(
                func.case(
                    (AgentScore.outcome == "LOSS", 1),
                    else_=0,
                )
            ).label("losses"),
            func.sum(
                func.case(
                    (AgentScore.outcome == "NEUTRAL", 1),
                    else_=0,
                )
            ).label("neutrals"),
        )
        .group_by(AgentScore.node_id)
        .subquery()
    )

    result = await db.execute(
        select(
            Node,
            score_count_sq.c.total_scores,
            score_count_sq.c.wins,
            score_count_sq.c.losses,
            score_count_sq.c.neutrals,
        )
        .outerjoin(score_count_sq, Node.id == score_count_sq.c.node_id)
        .order_by(desc(Node.reputation))
        .limit(limit)
    )

    leaderboard: list[dict] = []
    for node, total, wins, losses, neutrals in result.all():
        total = total or 0
        wins = wins or 0
        losses = losses or 0
        neutrals = neutrals or 0
        win_rate = (wins / total * 100) if total > 0 else 0.0

        leaderboard.append(
            {
                "rank": len(leaderboard) + 1,
                "node_id": str(node.id),
                "node_name": node.name,
                "reputation": node.reputation,
                "total_theses": total,
                "wins": wins,
                "losses": losses,
                "neutrals": neutrals,
                "win_rate": round(win_rate, 2),
            }
        )

    return leaderboard
