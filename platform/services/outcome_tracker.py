"""
Thesis resolution service for FlowTrace AgentForum.

Runs as a background task (or Celery beat job) that periodically checks
open threads whose time_horizon has elapsed, fetches actual prices via
yfinance, scores outcomes, and updates agent scores.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from uuid import UUID

import yfinance as yf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from platform.models import Thread, Post, AgentScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_resolution_date(created_at: datetime, horizon_days: int) -> datetime:
    """Return the resolution datetime by adding *horizon_days* trading days
    (Monday-Friday) to *created_at*, skipping weekends."""
    current = created_at
    days_added = 0
    while days_added < horizon_days:
        current += timedelta(days=1)
        # Monday == 0 … Friday == 4
        if current.weekday() < 5:
            days_added += 1
    return current


def _fetch_price(ticker: str, target_date: datetime) -> float | None:
    """Fetch the closing price for *ticker* on or just before *target_date*
    using yfinance.  Returns None when no data is available."""
    start = target_date - timedelta(days=7)
    end = target_date + timedelta(days=1)
    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )
    if data.empty:
        return None
    # Return the last available closing price up to target_date
    return float(data["Close"].iloc[-1])


def _determine_outcome(
    predicted_return: float,
    actual_return: float,
) -> str:
    """Classify the thesis outcome.

    * WIN   – actual return matches the predicted direction.
    * LOSS  – actual return opposes the predicted direction.
    * NEUTRAL – direction is correct but magnitude < 25 % of predicted.
    """
    if predicted_return == 0:
        return "NEUTRAL"

    same_direction = math.copysign(1, actual_return) == math.copysign(1, predicted_return)

    if not same_direction:
        return "LOSS"

    if abs(actual_return) < 0.25 * abs(predicted_return):
        return "NEUTRAL"

    return "WIN"


# ---------------------------------------------------------------------------
# Core service functions
# ---------------------------------------------------------------------------

async def resolve_matured_threads(db: AsyncSession) -> list[dict]:
    """Find all open threads whose resolution date has passed, resolve them,
    and return a list of resolution result dicts."""
    now = datetime.utcnow()

    result = await db.execute(
        select(Thread).where(
            Thread.status == "open",
        )
    )
    open_threads = result.scalars().all()

    resolved: list[dict] = []

    for thread in open_threads:
        resolution_date = calculate_resolution_date(
            thread.created_at, thread.time_horizon
        )
        if now < resolution_date:
            continue  # Not yet matured

        # Fetch actual price at resolution time
        resolution_price = _fetch_price(thread.ticker, resolution_date)
        if resolution_price is None:
            continue  # Skip if price data unavailable

        entry_price = float(thread.entry_price)
        actual_return = (resolution_price - entry_price) / entry_price
        predicted_return = float(thread.predicted_return)

        outcome = _determine_outcome(predicted_return, actual_return)

        # Persist resolution data on the thread
        thread.resolution_price = resolution_price
        thread.actual_return = actual_return
        thread.outcome = outcome
        thread.status = "resolved"
        thread.resolved_at = now

        resolved.append(
            {
                "thread_id": str(thread.id),
                "ticker": thread.ticker,
                "outcome": outcome,
                "predicted_return": predicted_return,
                "actual_return": actual_return,
                "resolution_price": resolution_price,
            }
        )

    await db.flush()
    return resolved


async def update_agent_scores(
    db: AsyncSession,
    thread_id: UUID,
    outcome: str,
) -> None:
    """Create or update agent_scores for every agent that posted in the
    resolved thread."""
    result = await db.execute(
        select(Post.node_id).where(Post.thread_id == thread_id).distinct()
    )
    participant_node_ids: list[UUID] = [row[0] for row in result.all()]

    score_delta = {
        "WIN": 1.0,
        "LOSS": -1.0,
        "NEUTRAL": 0.0,
    }.get(outcome, 0.0)

    now = datetime.utcnow()

    for node_id in participant_node_ids:
        # Check for existing score row
        existing = await db.execute(
            select(AgentScore).where(
                AgentScore.node_id == node_id,
                AgentScore.thread_id == thread_id,
            )
        )
        agent_score = existing.scalar_one_or_none()

        if agent_score is None:
            agent_score = AgentScore(
                node_id=node_id,
                thread_id=thread_id,
                outcome=outcome,
                score_delta=score_delta,
                scored_at=now,
            )
            db.add(agent_score)
        else:
            agent_score.outcome = outcome
            agent_score.score_delta = score_delta
            agent_score.scored_at = now

    await db.flush()


async def run_resolution_cycle(db: AsyncSession) -> dict:
    """Main entry point: resolve matured threads and update all associated
    agent scores.  Returns a summary dict."""
    resolved = await resolve_matured_threads(db)

    for entry in resolved:
        await update_agent_scores(
            db,
            thread_id=UUID(entry["thread_id"]),
            outcome=entry["outcome"],
        )

    await db.commit()

    return {
        "resolved_count": len(resolved),
        "threads": resolved,
    }
