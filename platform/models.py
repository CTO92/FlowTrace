"""
FlowTrace AgentForum - SQLAlchemy ORM Models

PostgreSQL database models for the AgentForum platform.
Uses SQLAlchemy 2.0 mapped_column style with DeclarativeBase.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://localhost:5432/flowtrace"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class Node(Base):
    __tablename__ = "nodes"

    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    node_alias: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )
    public_key: Mapped[str] = mapped_column(Text, nullable=False)
    registered_at: Mapped[datetime] = mapped_column(
        default=func.now(), server_default=func.now()
    )
    last_seen: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    reputation: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    status: Mapped[str] = mapped_column(
        String(20), default="pending", server_default="pending"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'active', 'suspended', 'banned')",
            name="ck_nodes_status",
        ),
    )

    # Relationships
    threads: Mapped[list["Thread"]] = relationship(
        back_populates="node", cascade="all, delete-orphan"
    )
    posts: Mapped[list["Post"]] = relationship(
        back_populates="node", cascade="all, delete-orphan"
    )
    agent_scores: Mapped[list["AgentScore"]] = relationship(
        back_populates="node", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Node(node_id={self.node_id}, alias={self.node_alias}, status={self.status})>"


class Thread(Base):
    __tablename__ = "threads"

    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_by: Mapped[str] = mapped_column(Text, nullable=False, comment="agent_id")
    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("nodes.node_id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    thesis_summary: Mapped[str] = mapped_column(Text, nullable=False)
    time_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(), server_default=func.now()
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    actual_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "direction IN ('BULLISH', 'BEARISH', 'NEUTRAL')",
            name="ck_threads_direction",
        ),
        CheckConstraint(
            "time_horizon >= 1 AND time_horizon <= 5",
            name="ck_threads_time_horizon",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_threads_confidence",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('WIN', 'LOSS', 'NEUTRAL')",
            name="ck_threads_outcome",
        ),
    )

    # Relationships
    node: Mapped["Node"] = relationship(back_populates="threads")
    posts: Mapped[list["Post"]] = relationship(
        back_populates="thread", cascade="all, delete-orphan"
    )
    signals: Mapped[list["Signal"]] = relationship(
        back_populates="thread", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Thread(thread_id={self.thread_id}, ticker={self.ticker}, "
            f"direction={self.direction})>"
        )


class Post(Base):
    __tablename__ = "posts"

    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("threads.thread_id"), nullable=False
    )
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("nodes.node_id"), nullable=False
    )
    post_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    data_sources: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )
    signature: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "post_type IN ('SUPPORT', 'CHALLENGE', 'EVIDENCE', 'UPDATE', 'CONCESSION')",
            name="ck_posts_post_type",
        ),
    )

    # Relationships
    thread: Mapped["Thread"] = relationship(back_populates="posts")
    node: Mapped["Node"] = relationship(back_populates="posts")

    def __repr__(self) -> str:
        return (
            f"<Post(post_id={self.post_id}, type={self.post_type}, "
            f"agent={self.agent_id})>"
        )


class Signal(Base):
    __tablename__ = "signals"

    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("threads.thread_id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    consensus_score: Mapped[float] = mapped_column(Float, nullable=False)
    supporting: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    challenging: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(), server_default=func.now()
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    actual_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    thread: Mapped["Thread"] = relationship(back_populates="signals")

    def __repr__(self) -> str:
        return (
            f"<Signal(signal_id={self.signal_id}, ticker={self.ticker}, "
            f"direction={self.direction})>"
        )


class AgentScore(Base):
    __tablename__ = "agent_scores"

    agent_type: Mapped[str] = mapped_column(Text, primary_key=True)
    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("nodes.node_id"),
        primary_key=True,
    )
    total_theses: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    winning_theses: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    avg_return: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    score: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    last_updated: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    node: Mapped["Node"] = relationship(back_populates="agent_scores")

    def __repr__(self) -> str:
        return (
            f"<AgentScore(agent_type={self.agent_type}, node_id={self.node_id}, "
            f"score={self.score})>"
        )


def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
