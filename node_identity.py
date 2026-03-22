"""
FlowTrace Node & Agent Identity Management

Generates and persists a unique identity for this trader's node,
including an Ed25519 keypair for cryptographic signing of all
messages posted to the AgentForum.
"""

import os
import json
import uuid
import hashlib
import random
import time
from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDENTITY_FILE = os.path.join(BASE_DIR, "node_identity.json")

# Word lists for human-readable node aliases
_ADJECTIVES = [
    "iron", "steel", "swift", "bright", "silent", "bold", "sharp", "dark",
    "calm", "cold", "deep", "fast", "keen", "red", "blue", "green", "gold",
    "wild", "true", "lone", "clear", "prime", "strong", "steady", "quiet",
]
_NOUNS = [
    "falcon", "bear", "wolf", "hawk", "tiger", "eagle", "fox", "raven",
    "lion", "shark", "viper", "cobra", "puma", "lynx", "orca", "stag",
    "bull", "ram", "pike", "crane", "heron", "condor", "drake", "otter",
]


def _generate_alias() -> str:
    """Generate a human-readable alias like 'iron-falcon-42'."""
    adj = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    num = random.randint(10, 99)
    return f"{adj}-{noun}-{num}"


def _generate_keypair() -> tuple[str, str]:
    """Generate an Ed25519 keypair, returning (private_pem, public_pem) as strings."""
    private_key = Ed25519PrivateKey.generate()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")

    return private_pem, public_pem


def generate_node_identity() -> dict:
    """Create a brand-new node identity (UUID, alias, keypair)."""
    private_pem, public_pem = _generate_keypair()

    identity = {
        "node_id": str(uuid.uuid4()),
        "node_alias": _generate_alias(),
        "public_key": public_pem,
        "private_key": private_pem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "forum_registered": False,
        "forum_status": "unregistered",  # unregistered | pending | active | suspended
    }
    return identity


def save_identity(identity: dict) -> None:
    """Persist identity to the local JSON file."""
    with open(IDENTITY_FILE, "w") as f:
        json.dump(identity, f, indent=2)


def load_identity() -> dict:
    """Load existing identity or generate and save a new one."""
    if os.path.exists(IDENTITY_FILE):
        with open(IDENTITY_FILE, "r") as f:
            return json.load(f)

    identity = generate_node_identity()
    save_identity(identity)
    return identity


def get_node_id() -> str:
    """Return this node's UUID."""
    return load_identity()["node_id"]


def get_node_alias() -> str:
    """Return this node's human-readable alias."""
    return load_identity()["node_alias"]


def get_public_key() -> str:
    """Return this node's public key PEM."""
    return load_identity()["public_key"]


def get_private_key_bytes() -> bytes:
    """Load the private key as bytes for signing operations."""
    identity = load_identity()
    return identity["private_key"].encode("utf-8")


def load_private_key() -> Ed25519PrivateKey:
    """Load the Ed25519 private key object for signing."""
    pem_bytes = get_private_key_bytes()
    return serialization.load_pem_private_key(pem_bytes, password=None)


def sign_message(message: str) -> str:
    """Sign a message string with this node's private key. Returns hex-encoded signature."""
    private_key = load_private_key()
    signature = private_key.sign(message.encode("utf-8"))
    return signature.hex()


def generate_agent_id(agent_type: str) -> str:
    """
    Generate a globally unique agent ID scoped to this node.
    Format: {node_id}:{agent_type}:{instance_uuid}
    """
    node_id = get_node_id()
    instance_id = str(uuid.uuid4())[:8]
    return f"{node_id}:{agent_type}:{instance_id}"


def compute_persona_hash(system_prompt: str, config: dict = None) -> str:
    """
    Hash an agent's system prompt + config to produce a persona fingerprint.
    This lets other agents know if an agent has been reconfigured.
    """
    payload = system_prompt + json.dumps(config or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def update_forum_status(status: str) -> None:
    """Update the local identity file with the current forum registration status."""
    identity = load_identity()
    identity["forum_status"] = status
    if status != "unregistered":
        identity["forum_registered"] = True
    save_identity(identity)


def get_forum_status() -> str:
    """Return the current forum registration status."""
    return load_identity().get("forum_status", "unregistered")
