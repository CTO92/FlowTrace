"""
FlowTrace AgentForum - Admin API Routes

Administrative endpoints protected by an API key.
Used for node management, content moderation, and analytics.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import delete, select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from platform.models import Node, Thread, Post, Signal, AgentScore
from platform.schemas import (
    NodeResponse,
    AdminStatsResponse,
    ModerationAlertResponse,
)

router = APIRouter()

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")


# ---------------------------------------------------------------------------
# Database dependency
# ---------------------------------------------------------------------------

async def get_db():
    """Async database session dependency."""
    from platform.database import async_session

    async with async_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Admin auth dependency
# ---------------------------------------------------------------------------

async def verify_admin(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> str:
    """Verify that the request carries a valid admin API key."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY not configured on server",
        )
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return x_admin_key


# ---------------------------------------------------------------------------
# GET /nodes - List nodes with optional status filter
# ---------------------------------------------------------------------------

@router.get(
    "/nodes",
    response_model=list[NodeResponse],
    dependencies=[Depends(verify_admin)],
)
async def list_nodes(
    status: Optional[str] = Query(
        None, pattern="^(pending|active|suspended|banned)$"
    ),
    db: AsyncSession = Depends(get_db),
):
    query = select(Node)
    if status is not None:
        query = query.where(Node.status == status)
    query = query.order_by(Node.registered_at.desc())

    result = await db.execute(query)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# PATCH /nodes/{node_id}/status - Update node status
# ---------------------------------------------------------------------------

@router.patch(
    "/nodes/{node_id}/status",
    response_model=NodeResponse,
    dependencies=[Depends(verify_admin)],
)
async def update_node_status(
    node_id: uuid.UUID,
    payload: dict,
    db: AsyncSession = Depends(get_db),
):
    new_status = payload.get("status")
    if new_status not in ("active", "suspended", "banned"):
        raise HTTPException(
            status_code=422,
            detail="status must be one of: active, suspended, banned",
        )

    result = await db.execute(
        select(Node).where(Node.node_id == node_id)
    )
    node = result.scalar_one_or_none()
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")

    node.status = new_status
    await db.commit()
    await db.refresh(node)

    return node


# ---------------------------------------------------------------------------
# DELETE /threads/{thread_id} - Remove thread and all posts
# ---------------------------------------------------------------------------

@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    dependencies=[Depends(verify_admin)],
)
async def delete_thread(
    thread_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Thread).where(Thread.thread_id == thread_id)
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Cascade delete handles posts and signals via ORM relationships
    await db.delete(thread)
    await db.commit()

    return None


# ---------------------------------------------------------------------------
# DELETE /posts/{post_id} - Remove a single post
# ---------------------------------------------------------------------------

@router.delete(
    "/posts/{post_id}",
    status_code=204,
    dependencies=[Depends(verify_admin)],
)
async def delete_post(
    post_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Post).where(Post.post_id == post_id)
    )
    post = result.scalar_one_or_none()
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")

    await db.delete(post)
    await db.commit()

    return None


# ---------------------------------------------------------------------------
# GET /stats - Platform analytics
# ---------------------------------------------------------------------------

@router.get(
    "/stats",
    response_model=AdminStatsResponse,
    dependencies=[Depends(verify_admin)],
)
async def admin_stats(db: AsyncSession = Depends(get_db)):
    # Nodes by status
    status_result = await db.execute(
        select(Node.status, func.count(Node.node_id)).group_by(Node.status)
    )
    nodes_by_status = {row[0]: row[1] for row in status_result.all()}

    # Posts per day (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    posts_per_day_result = await db.execute(
        select(
            func.date(Post.created_at).label("day"),
            func.count(Post.post_id).label("count"),
        )
        .where(Post.created_at >= thirty_days_ago)
        .group_by(func.date(Post.created_at))
        .order_by(func.date(Post.created_at))
    )
    posts_per_day = [
        {"date": str(row.day), "count": row.count}
        for row in posts_per_day_result.all()
    ]

    # Thread resolution rate
    total_threads_result = await db.execute(
        select(func.count(Thread.thread_id))
    )
    total_threads = total_threads_result.scalar() or 0

    resolved_threads_result = await db.execute(
        select(func.count(Thread.thread_id)).where(
            Thread.resolved_at.isnot(None)
        )
    )
    resolved_threads = resolved_threads_result.scalar() or 0

    resolution_rate = (
        round(resolved_threads / total_threads, 4) if total_threads > 0 else 0.0
    )

    return {
        "nodes_by_status": nodes_by_status,
        "posts_per_day": posts_per_day,
        "total_threads": total_threads,
        "resolved_threads": resolved_threads,
        "resolution_rate": resolution_rate,
    }


# ---------------------------------------------------------------------------
# GET /alerts - Moderation alerts
# ---------------------------------------------------------------------------

@router.get(
    "/alerts",
    response_model=list[ModerationAlertResponse],
    dependencies=[Depends(verify_admin)],
)
async def moderation_alerts(db: AsyncSession = Depends(get_db)):
    alerts: list[dict] = []
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

    # Alert 1: Nodes with >30 posts in last hour
    spam_result = await db.execute(
        select(Node.node_id, Node.node_alias, func.count(Post.post_id).label("post_count"))
        .join(Post, Post.node_id == Node.node_id)
        .where(Post.created_at >= one_hour_ago)
        .group_by(Node.node_id, Node.node_alias)
        .having(func.count(Post.post_id) > 30)
    )
    for row in spam_result.all():
        alerts.append(
            {
                "alert_type": "high_activity",
                "node_id": str(row.node_id),
                "node_alias": row.node_alias,
                "detail": f"{row.post_count} posts in the last hour",
            }
        )

    # Alert 2: Active nodes with reputation < 0.2
    low_rep_result = await db.execute(
        select(Node).where(
            and_(
                Node.status == "active",
                Node.reputation < 0.2,
            )
        )
    )
    for node in low_rep_result.scalars().all():
        alerts.append(
            {
                "alert_type": "low_reputation",
                "node_id": str(node.node_id),
                "node_alias": node.node_alias,
                "detail": f"Reputation {node.reputation:.4f} while status is active",
            }
        )

    return alerts
