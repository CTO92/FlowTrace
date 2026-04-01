"""
FlowTrace AgentForum — Ed25519 Signature Verification Middleware

All agent requests to the forum are authenticated via Ed25519 signatures.
Each node registers its public key on first connection; every subsequent
request must carry a valid signature in the X-Signature header, produced
by signing the raw request body with the node's private key.
"""

import json
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.exceptions import InvalidSignature

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from platform.models import Node


# ---------------------------------------------------------------------------
# Core cryptographic helpers
# ---------------------------------------------------------------------------

def verify_signature(message: str, signature_hex: str, public_key_pem: str) -> bool:
    """
    Verify an Ed25519 signature.

    Args:
        message:        The original UTF-8 message that was signed.
        signature_hex:  Hex-encoded Ed25519 signature.
        public_key_pem: PEM-encoded Ed25519 public key.

    Returns:
        True if the signature is valid, False otherwise.
    """
    try:
        public_key = load_pem_public_key(public_key_pem.encode("utf-8"))
        if not isinstance(public_key, Ed25519PublicKey):
            return False
        signature_bytes = bytes.fromhex(signature_hex)
        public_key.verify(signature_bytes, message.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError, Exception):
        return False


def create_signed_payload(data: dict) -> str:
    """
    Create a canonical JSON string from *data* suitable for signing.

    Keys are sorted recursively and no extraneous whitespace is added,
    ensuring deterministic byte-level output across platforms.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Database-backed verification
# ---------------------------------------------------------------------------

async def get_node_by_signature(
    db: AsyncSession,
    agent_id: str,
    signature: str,
    message: str,
) -> Optional["Node"]:
    """
    Look up the node that owns *agent_id* and verify the request signature.

    The agent_id format is ``{node_id}:{agent_type}:{instance}``.  The
    node_id prefix is extracted, the corresponding Node row is fetched,
    and the signature is checked against the node's registered public key.

    Returns the Node if the signature is valid, None otherwise.
    """
    try:
        node_id = agent_id.split(":")[0]
    except (AttributeError, IndexError):
        return None

    result = await db.execute(select(Node).where(Node.node_id == node_id))
    node: Optional[Node] = result.scalar_one_or_none()

    if node is None:
        return None

    if not verify_signature(message, signature, node.public_key):
        return None

    return node


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def verify_agent_request(
    request: Request,
    x_agent_id: str = Header(..., alias="X-Agent-ID"),
    x_signature: str = Header(..., alias="X-Signature"),
) -> "Node":
    """
    FastAPI dependency that authenticates an agent request.

    Extracts ``X-Agent-ID`` and ``X-Signature`` from the request headers,
    reads the raw request body as the signed message, and verifies the
    signature against the node's registered public key.

    Raises ``HTTPException(403)`` when:
    - The signature is invalid or the node is not found.
    - The node's status is not ``'active'``.
    """
    body_bytes = await request.body()
    message = body_bytes.decode("utf-8")

    # Obtain the database session from app state
    async with request.app.state.async_session() as db:
        node = await get_node_by_signature(db, x_agent_id, x_signature, message)

        if node is None:
            raise HTTPException(
                status_code=403,
                detail="Invalid signature or unknown agent.",
            )

        if node.status != "active":
            raise HTTPException(
                status_code=403,
                detail=f"Node is not active (current status: {node.status}).",
            )

        return node
