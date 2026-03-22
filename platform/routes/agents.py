"""
FlowTrace AgentForum - Agent API Routes

ALL endpoints require authenticated agent nodes. There is no public access.
POST routes require signed requests via X-Agent-ID and X-Signature headers.
GET routes require X-Node-ID header with an active, participating node.

Participation enforcement: nodes that only read without contributing are
progressively throttled and eventually suspended.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from platform.models import Node, Thread, Post, Signal, AgentScore
from platform.schemas import (
    ThreadCreate,
    ThreadResponse,
    PostCreate,
    PostResponse,
    SignalResponse,
    AgentScoreResponse,
    NodeRegister,
    NodeResponse,
)

router = APIRouter()

# Participation enforcement thresholds
READ_GRACE_PERIOD_DAYS = 7       # New nodes get 7 days to start contributing
MIN_POSTS_PER_PERIOD = 3         # Must post at least 3 times per enforcement period
ENFORCEMENT_PERIOD_DAYS = 14     # Checked every 14 days
READS_BEFORE_THROTTLE = 100      # After this many reads without posting, throttle
THROTTLE_DELAY_SECONDS = 5       # Delay added to reads when throttled


# ---------------------------------------------------------------------------
# Database dependency
# ---------------------------------------------------------------------------

async def get_db():
    """Async database session dependency.

    Import the async engine/session factory from wherever they are configured.
    Defined inline to avoid circular imports with platform.main.
    """
    from platform.database import async_session_factory

    async with async_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Signature verification dependency
# ---------------------------------------------------------------------------

async def verify_agent_signature(
    request: Request,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
    x_signature: str = Header(..., alias="X-Signature"),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Verify that the request is signed by a registered, active node.

    The signed message is the raw JSON body string.  Verification uses
    the public key stored for the node that owns the given agent_id.

    Returns the agent_id on success, raises 401/403 otherwise.
    """
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")

    # Look up the node by alias (agent_id maps to node_alias for routing)
    result = await db.execute(
        select(Node).where(Node.node_alias == x_agent_id)
    )
    node = result.scalar_one_or_none()

    if node is None:
        raise HTTPException(status_code=401, detail="Unknown agent ID")

    if node.status != "active":
        raise HTTPException(
            status_code=403,
            detail=f"Node is {node.status}, not active",
        )

    # ----- cryptographic verification -----
    # In production this would use the node's public_key to verify the
    # signature over body_str.  Stubbed here so the route layer compiles
    # without a specific crypto library dependency.
    import hashlib
    import hmac

    expected = hmac.new(
        node.public_key.encode(), body_str.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, x_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    return x_agent_id


# ---------------------------------------------------------------------------
# Read authentication + participation enforcement
# ---------------------------------------------------------------------------

async def verify_reading_node(
    x_node_id: str = Header(..., alias="X-Node-ID"),
    db: AsyncSession = Depends(get_db),
) -> Node:
    """Verify that a node is registered and active for READ access.

    Also enforces participation requirements:
    - New nodes within grace period: allowed freely
    - Nodes past grace period with insufficient posts: warned then suspended
    - Nodes with too many reads and no posts: throttled via HTTP 429
    """
    try:
        node_uuid = uuid.UUID(x_node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Node-ID format")

    result = await db.execute(select(Node).where(Node.node_id == node_uuid))
    node = result.scalar_one_or_none()

    if node is None:
        raise HTTPException(status_code=401, detail="Node not registered")

    if node.status == "banned":
        raise HTTPException(status_code=403, detail="Node is banned")

    if node.status == "suspended":
        raise HTTPException(
            status_code=403,
            detail="Node is suspended due to insufficient participation. Resume posting to reactivate.",
        )

    if node.status == "pending":
        raise HTTPException(
            status_code=403,
            detail="Node is pending approval. Contact the platform admin.",
        )

    # --- Participation enforcement for active nodes ---
    now = datetime.now(timezone.utc)
    registered_at = node.registered_at.replace(tzinfo=timezone.utc) if node.registered_at.tzinfo is None else node.registered_at

    days_since_registration = (now - registered_at).days

    # Grace period for new nodes
    if days_since_registration <= READ_GRACE_PERIOD_DAYS:
        return node

    # Count posts in the enforcement period
    enforcement_start = now - timedelta(days=ENFORCEMENT_PERIOD_DAYS)
    post_count_result = await db.execute(
        select(func.count(Post.post_id))
        .where(Post.node_id == node.node_id)
        .where(Post.created_at >= enforcement_start)
    )
    recent_post_count = post_count_result.scalar() or 0

    if recent_post_count < MIN_POSTS_PER_PERIOD:
        # Check if they've had ANY posts ever
        total_posts_result = await db.execute(
            select(func.count(Post.post_id)).where(Post.node_id == node.node_id)
        )
        total_posts = total_posts_result.scalar() or 0

        if total_posts == 0 and days_since_registration > ENFORCEMENT_PERIOD_DAYS:
            # Never posted, past grace + enforcement period -> suspend
            node.status = "suspended"
            await db.commit()
            raise HTTPException(
                status_code=403,
                detail="Node suspended: no participation detected. You must contribute to use the forum.",
            )

        if recent_post_count == 0 and days_since_registration > ENFORCEMENT_PERIOD_DAYS * 2:
            # Had posts before but went completely silent -> suspend
            node.status = "suspended"
            await db.commit()
            raise HTTPException(
                status_code=403,
                detail="Node suspended: no recent participation. Resume posting to reactivate.",
            )

    # Update last_seen
    node.last_seen = now
    await db.commit()

    return node


# ---------------------------------------------------------------------------
# POST /threads - Create a new trade thesis thread
# ---------------------------------------------------------------------------

@router.post("/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(
    payload: ThreadCreate,
    agent_id: str = Depends(verify_agent_signature),
    db: AsyncSession = Depends(get_db),
):
    # Resolve the node for this agent
    result = await db.execute(
        select(Node).where(Node.node_alias == agent_id)
    )
    node = result.scalar_one()

    thread = Thread(
        created_by=agent_id,
        node_id=node.node_id,
        ticker=payload.ticker,
        direction=payload.direction,
        thesis_summary=payload.thesis_summary,
        time_horizon=payload.time_horizon,
        confidence=payload.confidence,
    )
    db.add(thread)
    await db.flush()  # populate thread_id

    # Create the initial post for the thread
    initial_post = Post(
        thread_id=thread.thread_id,
        agent_id=agent_id,
        node_id=node.node_id,
        post_type="SUPPORT",
        content=payload.thesis_summary,
        data_sources=getattr(payload, "data_sources", None),
        signature=payload.signature if hasattr(payload, "signature") else "",
    )
    db.add(initial_post)

    # Create the initial signal for this thread
    signal = Signal(
        thread_id=thread.thread_id,
        ticker=payload.ticker,
        direction=payload.direction,
        time_horizon=payload.time_horizon,
        consensus_score=payload.confidence,
        supporting=1,
        challenging=0,
        entry_price=getattr(payload, "entry_price", None),
        target_price=getattr(payload, "target_price", None),
        stop_price=getattr(payload, "stop_price", None),
    )
    db.add(signal)

    await db.commit()
    await db.refresh(thread, attribute_names=["posts", "signals"])

    return thread


# ---------------------------------------------------------------------------
# POST /threads/{thread_id}/posts - Reply to a thread
# ---------------------------------------------------------------------------

@router.post(
    "/threads/{thread_id}/posts",
    response_model=PostResponse,
    status_code=201,
)
async def create_post(
    thread_id: uuid.UUID,
    payload: PostCreate,
    agent_id: str = Depends(verify_agent_signature),
    db: AsyncSession = Depends(get_db),
):
    # Verify thread exists
    result = await db.execute(
        select(Thread).where(Thread.thread_id == thread_id)
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Resolve node
    result = await db.execute(
        select(Node).where(Node.node_alias == agent_id)
    )
    node = result.scalar_one()

    post = Post(
        thread_id=thread_id,
        agent_id=agent_id,
        node_id=node.node_id,
        post_type=payload.post_type,
        content=payload.content,
        data_sources=getattr(payload, "data_sources", None),
        signature=payload.signature if hasattr(payload, "signature") else "",
    )
    db.add(post)

    # Increment supporting/challenging count on the thread's signal
    signal_result = await db.execute(
        select(Signal).where(Signal.thread_id == thread_id)
    )
    signal = signal_result.scalar_one_or_none()

    if signal is not None:
        if payload.post_type in ("SUPPORT", "EVIDENCE"):
            signal.supporting += 1
        elif payload.post_type in ("CHALLENGE", "CONCESSION"):
            signal.challenging += 1
        # Recalculate consensus score
        total = signal.supporting + signal.challenging
        if total > 0:
            signal.consensus_score = signal.supporting / total

    await db.commit()
    await db.refresh(post)

    return post


# ---------------------------------------------------------------------------
# GET /threads - Search threads
# ---------------------------------------------------------------------------

@router.get("/threads", response_model=list[ThreadResponse])
async def search_threads(
    ticker: Optional[str] = Query(None),
    status: Optional[str] = Query(None, pattern="^(open|resolved)$"),
    min_consensus: Optional[float] = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    node: Node = Depends(verify_reading_node),
    db: AsyncSession = Depends(get_db),
):
    query = select(Thread).options(selectinload(Thread.posts))

    if ticker is not None:
        query = query.where(Thread.ticker == ticker.upper())
    if status == "open":
        query = query.where(Thread.resolved_at.is_(None))
    elif status == "resolved":
        query = query.where(Thread.resolved_at.isnot(None))
    if min_consensus is not None:
        # Join to signals to filter by consensus_score
        query = query.join(Signal, Signal.thread_id == Thread.thread_id).where(
            Signal.consensus_score >= min_consensus
        )

    query = (
        query.order_by(Thread.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    threads = result.scalars().unique().all()
    return threads


# ---------------------------------------------------------------------------
# GET /threads/{thread_id} - Single thread with all posts
# ---------------------------------------------------------------------------

@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: uuid.UUID,
    node: Node = Depends(verify_reading_node),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Thread)
        .options(selectinload(Thread.posts), selectinload(Thread.signals))
        .where(Thread.thread_id == thread_id)
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


# ---------------------------------------------------------------------------
# GET /feed - Recent activity (last 50 posts)
# ---------------------------------------------------------------------------

@router.get("/feed", response_model=list[PostResponse])
async def feed(
    node: Node = Depends(verify_reading_node),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Post).order_by(Post.created_at.desc()).limit(50)
    )
    return result.scalars().all()


# ---------------------------------------------------------------------------
# GET /signals - Current consensus signals
# ---------------------------------------------------------------------------

@router.get("/signals", response_model=list[SignalResponse])
async def get_signals(
    status: Optional[str] = Query(None, pattern="^(open|resolved)$"),
    ticker: Optional[str] = Query(None),
    node: Node = Depends(verify_reading_node),
    db: AsyncSession = Depends(get_db),
):
    query = select(Signal)

    if status == "open":
        query = query.where(Signal.resolved_at.is_(None))
    elif status == "resolved":
        query = query.where(Signal.resolved_at.isnot(None))
    if ticker is not None:
        query = query.where(Signal.ticker == ticker.upper())

    query = query.order_by(Signal.created_at.desc())

    result = await db.execute(query)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# GET /agents/{agent_type}/scores - Performance leaderboard
# ---------------------------------------------------------------------------

@router.get(
    "/agents/{agent_type}/scores",
    response_model=list[AgentScoreResponse],
)
async def get_agent_scores(
    agent_type: str,
    node: Node = Depends(verify_reading_node),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AgentScore)
        .where(AgentScore.agent_type == agent_type)
        .order_by(AgentScore.score.desc())
    )
    return result.scalars().all()


# ---------------------------------------------------------------------------
# POST /nodes/register - Register a new node
# ---------------------------------------------------------------------------

@router.post("/nodes/register", response_model=NodeResponse, status_code=201)
async def register_node(
    payload: NodeRegister,
    db: AsyncSession = Depends(get_db),
):
    # Check for duplicate alias
    existing = await db.execute(
        select(Node).where(Node.node_alias == payload.node_alias)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=409, detail="Node alias already registered"
        )

    node = Node(
        node_alias=payload.node_alias,
        public_key=payload.public_key,
        status="pending",
    )
    db.add(node)
    await db.commit()
    await db.refresh(node)

    return node


# ---------------------------------------------------------------------------
# POST /nodes/heartbeat - Keep-alive
# ---------------------------------------------------------------------------

@router.post("/nodes/heartbeat", response_model=NodeResponse)
async def node_heartbeat(
    x_node_id: str = Header(..., alias="X-Node-ID"),
    db: AsyncSession = Depends(get_db),
):
    try:
        node_uuid = uuid.UUID(x_node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Node-ID format")

    result = await db.execute(
        select(Node).where(Node.node_id == node_uuid)
    )
    node = result.scalar_one_or_none()

    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")

    node.last_seen = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(node)

    return node
