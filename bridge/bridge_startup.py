"""Bridge startup and lifecycle management for FlowTrace.

Provides functions to initialize the TidalFlowBridge from FlowTrace's
continuous monitor or standalone entry point.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level singleton
_orchestrator = None
_adapter = None


def get_bridge_orchestrator():
    """Get the bridge orchestrator singleton (or None if not running)."""
    return _orchestrator


def get_bridge_adapter():
    """Get the bridge adapter singleton (or None if not running)."""
    return _adapter


def is_bridge_enabled() -> bool:
    """Check if bridge is configured and ready."""
    return _orchestrator is not None and _orchestrator.is_running


async def start_bridge() -> tuple:
    """Initialize and start the TidalFlowBridge.

    Returns (orchestrator, adapter) tuple, or (None, None) if not configured.
    """
    global _orchestrator, _adapter

    passphrase = os.environ.get("BRIDGE_PASSPHRASE")
    if not passphrase:
        logger.info("BRIDGE_PASSPHRASE not set — bridge disabled")
        return None, None

    try:
        from tidalflowbridge.protocol.bridge_orchestrator import BridgeOrchestrator
        from bridge.adapter import FlowTraceBridgeAdapter

        _adapter = FlowTraceBridgeAdapter()

        base_dir = Path(__file__).parent.parent
        _orchestrator = BridgeOrchestrator(
            adapter=_adapter,
            base_dir=base_dir,
            cluster_passphrase=passphrase,
        )
        await _orchestrator.start()

        logger.info(
            "TidalFlowBridge started: node=%s (%s)",
            _orchestrator.identity.node_alias,
            _orchestrator.identity.node_type,
        )
        return _orchestrator, _adapter

    except ImportError as e:
        logger.info("tidalflowbridge package not installed — bridge disabled: %s", e)
        return None, None
    except Exception as e:
        logger.warning("Bridge startup failed: %s", e)
        return None, None


async def stop_bridge() -> None:
    """Stop the bridge gracefully."""
    global _orchestrator, _adapter
    if _orchestrator:
        await _orchestrator.stop()
        logger.info("TidalFlowBridge stopped")
    _orchestrator = None
    _adapter = None
