"""
FlowTrace AgentForum - Pydantic Schemas

Request/response validation models for the AgentForum API.
Uses Pydantic v2 style with model_config.
"""

import uuid
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Node schemas
# ---------------------------------------------------------------------------

class NodeRegister(BaseModel):
    """Input schema for registering a new node."""

    node_alias: str = Field(..., min_length=1, max_length=128)
    public_key: str = Field(..., min_length=1)


class NodeResponse(BaseModel):
    """Response schema for node data."""

    model_config = ConfigDict(from_attributes=True)

    node_id: uuid.UUID
    node_alias: str
    status: str
    reputation: float
    registered_at: datetime


# ---------------------------------------------------------------------------
# Thread schemas
# ---------------------------------------------------------------------------

class ThreadCreate(BaseModel):
    """Input schema for creating a new debate thread."""

    ticker: str = Field(..., min_length=1, max_length=20)
    direction: str = Field(..., pattern=r"^(BULLISH|BEARISH|NEUTRAL)$")
    thesis_summary: str = Field(..., min_length=1)
    time_horizon: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    agent_id: str = Field(..., min_length=1)
    signature: str = Field(..., min_length=1)


class ThreadResponse(BaseModel):
    """Response schema for thread data, including post count."""

    model_config = ConfigDict(from_attributes=True)

    thread_id: uuid.UUID
    created_by: str
    node_id: uuid.UUID
    ticker: str
    direction: str
    thesis_summary: str
    time_horizon: int
    confidence: float
    created_at: datetime
    resolved_at: Optional[datetime] = None
    actual_return: Optional[float] = None
    outcome: Optional[str] = None
    post_count: int = 0


# ---------------------------------------------------------------------------
# Post schemas
# ---------------------------------------------------------------------------

class PostCreate(BaseModel):
    """Input schema for creating a post within a thread."""

    post_type: str = Field(
        ..., pattern=r"^(SUPPORT|CHALLENGE|EVIDENCE|UPDATE|CONCESSION)$"
    )
    content: str = Field(..., min_length=1)
    data_sources: Optional[dict[str, Any]] = None
    agent_id: str = Field(..., min_length=1)
    signature: str = Field(..., min_length=1)


class PostResponse(BaseModel):
    """Response schema for post data."""

    model_config = ConfigDict(from_attributes=True)

    post_id: uuid.UUID
    thread_id: uuid.UUID
    agent_id: str
    node_id: uuid.UUID
    post_type: str
    content: str
    data_sources: Optional[dict[str, Any]] = None
    signature: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Signal schemas
# ---------------------------------------------------------------------------

class SignalResponse(BaseModel):
    """Response schema for signal data."""

    model_config = ConfigDict(from_attributes=True)

    signal_id: uuid.UUID
    thread_id: uuid.UUID
    ticker: str
    direction: str
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_horizon: int
    consensus_score: float
    supporting: int
    challenging: int
    created_at: datetime
    resolved_at: Optional[datetime] = None
    actual_return: Optional[float] = None


# ---------------------------------------------------------------------------
# Agent score schemas
# ---------------------------------------------------------------------------

class AgentScoreResponse(BaseModel):
    """Response schema for agent score data."""

    model_config = ConfigDict(from_attributes=True)

    agent_type: str
    node_id: uuid.UUID
    total_theses: int
    winning_theses: int
    avg_confidence: float
    avg_return: float
    score: float
    last_updated: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Generic / utility schemas
# ---------------------------------------------------------------------------

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: list[T]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str
    version: str
    uptime: float
