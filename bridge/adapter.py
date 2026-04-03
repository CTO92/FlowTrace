"""FlowTrace bridge adapter — translates between FlowTrace internals and the bridge protocol.

Responsibilities:
- Collects trade signals, market confirmations, and outcome resolutions for publishing
- Handles incoming TidalShift pathway alerts, cascade warnings, and leading indicators
- Answers queries from TidalShift (market_validation, sentiment_check, ground_truth)
- Formulates auto-queries to TidalShift when unusual sector-wide moves are detected
"""

from __future__ import annotations

import logging
import os
import sqlite3
import json
from datetime import datetime, timezone
from typing import Any

from tidalflowbridge.adapter.base_adapter import BaseBridgeAdapter
from tidalflowbridge.schema.envelope import BridgeEnvelope
from tidalflowbridge.schema.signals import BridgeSignal
from tidalflowbridge.schema.queries import BridgeQuery, BridgeQueryResponse

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_graph.db")


class FlowTraceBridgeAdapter(BaseBridgeAdapter):
    """FlowTrace-specific bridge adapter implementation."""

    def __init__(self) -> None:
        self._outbound_queue: list[BridgeSignal] = []
        self._received_signals: list[BridgeEnvelope] = []
        self._cascade_cache: list[dict] = []  # Recent TidalShift pathway alerts
        self._cascade_warnings: list[dict] = []  # Active cascade warnings
        self._leading_indicators: list[dict] = []  # Indicators to watch
        self._db_path = DB_PATH

    @property
    def node_type(self) -> str:
        return "flowtrace"

    # ── Outbound: Collecting signals to publish ──

    async def collect_outbound_signals(self) -> list[BridgeSignal]:
        """Collect pending signals from FlowTrace for bridge publication."""
        signals = list(self._outbound_queue)
        self._outbound_queue.clear()
        return signals

    def queue_trade_signal(self, consensus_signal: dict) -> None:
        """Queue a consensus trade signal for publication to TidalShift."""
        evidence = {}
        try:
            evidence = json.loads(consensus_signal.get("evidence_summary", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass

        signal = BridgeSignal(
            signal_type="trade_signal",
            domain="market",
            summary=f"{consensus_signal.get('direction', '?')} on {consensus_signal.get('ticker', '?')} "
                    f"(score: {consensus_signal.get('consensus_score', 0):.0%})",
            confidence=float(consensus_signal.get("consensus_score", 0)),
            direction=consensus_signal.get("direction"),
            ticker=consensus_signal.get("ticker"),
            sectors=[consensus_signal.get("sector", "")] if consensus_signal.get("sector") else [],
            key_data={
                "ticker": consensus_signal.get("ticker", ""),
                "direction": consensus_signal.get("direction", ""),
                "consensus_score": float(consensus_signal.get("consensus_score", 0)),
                "expected_move_pct": float(consensus_signal.get("expected_move_pct", 0)),
                "time_horizon_days": int(consensus_signal.get("time_horizon_days", 5)),
                "event_type": consensus_signal.get("event_type", ""),
                "contributing_agents": consensus_signal.get("contributing_agents", []),
                "market_regime": consensus_signal.get("market_regime", "normal"),
                "sector": consensus_signal.get("sector", ""),
            },
        )
        self._outbound_queue.append(signal)

    def queue_outcome_resolution(self, signal: dict) -> None:
        """Queue a resolved signal for TidalShift pathway calibration."""
        bridge_signal = BridgeSignal(
            signal_type="outcome_resolution",
            domain="market",
            summary=f"Resolved: {signal.get('ticker')} {signal.get('outcome', '?')} "
                    f"({signal.get('actual_return', 0):.1f}%)",
            confidence=0.95,
            ticker=signal.get("ticker"),
            key_data={
                "original_bridge_signal_id": str(signal.get("id", "")),
                "ticker": signal.get("ticker", ""),
                "predicted_direction": signal.get("direction", ""),
                "actual_return_pct": float(signal.get("actual_return", 0)),
                "time_horizon_days": int(signal.get("time_horizon_days", 5)),
                "outcome": signal.get("outcome", "neutral"),
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._outbound_queue.append(bridge_signal)

    def queue_market_confirmation(self, ticker: str, data: dict) -> None:
        """Queue a market data snapshot for TidalShift pathway validation."""
        bridge_signal = BridgeSignal(
            signal_type="market_confirmation",
            domain="market",
            summary=f"Market snapshot: {ticker} {data.get('price_change_pct', 0):+.1f}%",
            confidence=0.9,
            ticker=ticker,
            key_data={
                "ticker": ticker,
                "current_price": data.get("current_price", 0),
                "price_change_pct": data.get("price_change_pct", 0),
                "volume_ratio": data.get("volume_ratio", 1.0),
                "sector_performance": data.get("sector_performance", {}),
                "vix_level": data.get("vix_level", 0),
                "relevant_technicals": data.get("technicals", {}),
            },
        )
        self._outbound_queue.append(bridge_signal)

    # ── Inbound: Handling TidalShift signals ──

    async def handle_incoming_signal(self, envelope: BridgeEnvelope) -> None:
        """Process an incoming signal from TidalShift."""
        self._received_signals.append(envelope)
        # Keep bounded
        if len(self._received_signals) > 500:
            self._received_signals = self._received_signals[-250:]

        signal_type = envelope.payload.get("signal_type", "")

        if signal_type == "pathway_alert":
            await self._handle_pathway_alert(envelope)
        elif signal_type == "cascade_warning":
            await self._handle_cascade_warning(envelope)
        elif signal_type == "leading_indicator":
            await self._handle_leading_indicator(envelope)
        else:
            logger.debug("Unhandled TidalShift signal type: %s", signal_type)

    async def _handle_pathway_alert(self, envelope: BridgeEnvelope) -> None:
        """Store a TidalShift pathway alert for enriching FlowTrace analysis."""
        payload = envelope.payload
        key_data = payload.get("key_data", payload)

        pathway_info = {
            "pathway_id": key_data.get("pathway_id"),
            "label": key_data.get("pathway_label", key_data.get("cascade_summary", "")),
            "plausibility": key_data.get("plausibility_score", payload.get("confidence", 0)),
            "composite": key_data.get("composite_score", 0),
            "domains_involved": key_data.get("domains_involved", []),
            "affected_sectors": key_data.get("affected_sectors", payload.get("sectors", [])),
            "affected_entities": key_data.get("affected_entities", payload.get("entities", [])),
            "leading_indicators": key_data.get("leading_indicators", []),
            "feedback_loops": key_data.get("feedback_loops", []),
            "key_links": key_data.get("key_links", []),
            "time_horizon": key_data.get("time_horizon", "weeks"),
            "summary": key_data.get("cascade_summary", payload.get("summary", "")),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cascade_cache.append(pathway_info)
        # Keep cache bounded
        if len(self._cascade_cache) > 100:
            self._cascade_cache = self._cascade_cache[-50:]

        logger.info(
            "Received TidalShift pathway: %s (plausibility=%.2f, sectors=%s)",
            pathway_info["label"][:60],
            pathway_info["plausibility"],
            pathway_info["affected_sectors"],
        )

    async def _handle_cascade_warning(self, envelope: BridgeEnvelope) -> None:
        """Store a cascade warning for risk context."""
        payload = envelope.payload
        key_data = payload.get("key_data", payload)

        warning = {
            "warning_type": key_data.get("warning_type", "unknown"),
            "severity": key_data.get("severity", payload.get("confidence", 0.5)),
            "summary": key_data.get("summary", payload.get("summary", "")),
            "affected_sectors": key_data.get("affected_sectors", []),
            "affected_entities": key_data.get("affected_entities", []),
            "recommended_action": key_data.get("recommended_action", ""),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cascade_warnings.append(warning)
        if len(self._cascade_warnings) > 50:
            self._cascade_warnings = self._cascade_warnings[-25:]

        logger.warning(
            "TidalShift cascade warning: %s (severity=%s)",
            warning["summary"][:80],
            warning["severity"],
        )

    async def _handle_leading_indicator(self, envelope: BridgeEnvelope) -> None:
        """Store a leading indicator to monitor."""
        payload = envelope.payload
        key_data = payload.get("key_data", payload)

        indicator = {
            "name": key_data.get("indicator_name", ""),
            "description": key_data.get("description", ""),
            "direction": key_data.get("direction", "change"),
            "threshold": key_data.get("threshold"),
            "pathway_id": key_data.get("pathway_id"),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        self._leading_indicators.append(indicator)
        if len(self._leading_indicators) > 50:
            self._leading_indicators = self._leading_indicators[-25:]

    # ── Query handling: Answering TidalShift questions ──

    async def handle_query(self, query: BridgeQuery) -> BridgeQueryResponse:
        """Handle an incoming query from TidalShift."""
        try:
            match query.query_type:
                case "market_validation":
                    return await self._query_market_validation(query)
                case "sentiment_check":
                    return await self._query_sentiment_check(query)
                case "ground_truth":
                    return await self._query_ground_truth(query)
                case "trade_consensus":
                    return await self._query_trade_consensus(query)
                case "custom":
                    return await self._query_custom(query)
                case _:
                    return BridgeQueryResponse(
                        query_id=query.query_id,
                        status="error",
                        error_message=f"Unknown query type: {query.query_type}",
                    )
        except Exception as e:
            logger.error("Error handling query %s: %s", query.query_id[:8], e)
            return BridgeQueryResponse(
                query_id=query.query_id,
                status="error",
                error_message=str(e),
            )

    async def _query_market_validation(self, query: BridgeQuery) -> BridgeQueryResponse:
        """TidalShift asks: Is the market pricing in this pathway?"""
        context = query.context
        tickers = context.get("tickers", [])
        sectors = context.get("sectors", [])

        results = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row

            for ticker in tickers[:10]:
                cursor = conn.execute(
                    """SELECT direction, consensus_score, expected_move_pct,
                              market_regime, timestamp
                       FROM consensus_signals
                       WHERE ticker = ? AND resolved_at IS NULL
                       ORDER BY timestamp DESC LIMIT 3""",
                    (ticker,),
                )
                rows = cursor.fetchall()
                if rows:
                    results.append({
                        "ticker": ticker,
                        "signals": [
                            {
                                "direction": r["direction"],
                                "consensus_score": r["consensus_score"],
                                "expected_move_pct": r["expected_move_pct"],
                                "regime": r["market_regime"],
                                "timestamp": r["timestamp"],
                            }
                            for r in rows
                        ],
                        "latest_direction": rows[0]["direction"],
                        "latest_score": rows[0]["consensus_score"],
                    })

            conn.close()
        except Exception as e:
            logger.warning("Error querying market validation: %s", e)

        return BridgeQueryResponse(
            query_id=query.query_id,
            status="success" if results else "partial",
            response={
                "validations": results,
                "count": len(results),
                "tickers_checked": tickers[:10],
            },
            confidence=0.7 if results else 0.2,
            cost_usd=0.0,
            depth_achieved=query.depth,
        )

    async def _query_sentiment_check(self, query: BridgeQuery) -> BridgeQueryResponse:
        """TidalShift asks: What's current sentiment on these tickers?"""
        context = query.context
        tickers = context.get("tickers", [])

        results = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row

            for ticker in tickers[:10]:
                cursor = conn.execute(
                    """SELECT direction, confidence, reasoning, event_type, timestamp
                       FROM signals
                       WHERE source_ticker = ? OR target_ticker = ?
                       ORDER BY timestamp DESC LIMIT 5""",
                    (ticker, ticker),
                )
                rows = cursor.fetchall()
                if rows:
                    bullish = sum(1 for r in rows if r["direction"] and "bull" in str(r["direction"]).lower())
                    bearish = sum(1 for r in rows if r["direction"] and "bear" in str(r["direction"]).lower())
                    results.append({
                        "ticker": ticker,
                        "signal_count": len(rows),
                        "bullish_count": bullish,
                        "bearish_count": bearish,
                        "dominant_sentiment": "BULLISH" if bullish > bearish else ("BEARISH" if bearish > bullish else "NEUTRAL"),
                        "avg_confidence": sum(r["confidence"] or 0 for r in rows) / len(rows),
                        "latest_event": rows[0]["event_type"],
                    })

            conn.close()
        except Exception as e:
            logger.warning("Error querying sentiment: %s", e)

        return BridgeQueryResponse(
            query_id=query.query_id,
            status="success" if results else "partial",
            response={"sentiments": results, "count": len(results)},
            confidence=0.65 if results else 0.2,
            cost_usd=0.0,
            depth_achieved=query.depth,
        )

    async def _query_ground_truth(self, query: BridgeQuery) -> BridgeQueryResponse:
        """TidalShift asks: What actually happened with asset X since date Y?"""
        context = query.context
        ticker = context.get("ticker", "")
        since = context.get("since", "")

        result = {}
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if not hist.empty:
                latest = hist.iloc[-1]
                first = hist.iloc[0]
                result = {
                    "ticker": ticker,
                    "current_price": float(latest["Close"]),
                    "period_start_price": float(first["Close"]),
                    "period_return_pct": float((latest["Close"] - first["Close"]) / first["Close"] * 100),
                    "period_high": float(hist["High"].max()),
                    "period_low": float(hist["Low"].min()),
                    "avg_volume": float(hist["Volume"].mean()),
                    "data_points": len(hist),
                }
        except Exception as e:
            logger.warning("Error fetching ground truth for %s: %s", ticker, e)
            result = {"ticker": ticker, "error": str(e)}

        return BridgeQueryResponse(
            query_id=query.query_id,
            status="success" if "current_price" in result else "error",
            response=result,
            confidence=0.9 if "current_price" in result else 0.0,
            cost_usd=0.0,
            depth_achieved=query.depth,
        )

    async def _query_trade_consensus(self, query: BridgeQuery) -> BridgeQueryResponse:
        """TidalShift asks: What's FlowTrace consensus on these tickers?"""
        context = query.context
        tickers = context.get("tickers", [])

        results = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row

            for ticker in tickers[:10]:
                cursor = conn.execute(
                    """SELECT direction, consensus_score, expected_move_pct,
                              time_horizon_days, event_type, reasoning, timestamp
                       FROM consensus_signals
                       WHERE ticker = ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    (ticker,),
                )
                row = cursor.fetchone()
                if row:
                    results.append({
                        "ticker": ticker,
                        "direction": row["direction"],
                        "consensus_score": row["consensus_score"],
                        "expected_move_pct": row["expected_move_pct"],
                        "time_horizon_days": row["time_horizon_days"],
                        "event_type": row["event_type"],
                        "reasoning": (row["reasoning"] or "")[:300],
                        "timestamp": row["timestamp"],
                    })

            conn.close()
        except Exception as e:
            logger.warning("Error querying trade consensus: %s", e)

        return BridgeQueryResponse(
            query_id=query.query_id,
            status="success" if results else "partial",
            response={"consensus_signals": results, "count": len(results)},
            confidence=0.75 if results else 0.2,
            cost_usd=0.0,
            depth_achieved=query.depth,
        )

    async def _query_custom(self, query: BridgeQuery) -> BridgeQueryResponse:
        """Handle free-text custom queries."""
        return BridgeQueryResponse(
            query_id=query.query_id,
            status="partial",
            response={
                "note": "Custom queries require LLM analysis",
                "question": query.question,
                "cached_pathways": len(self._cascade_cache),
                "active_warnings": len(self._cascade_warnings),
            },
            confidence=0.3,
            cost_usd=0.0,
            depth_achieved="cached",
        )

    # ── Auto-query formulation ──

    async def formulate_auto_query(self, trigger: dict[str, Any]) -> BridgeQuery | None:
        """Formulate an automatic query to TidalShift based on a local trigger."""
        trigger_type = trigger.get("type", "")

        if trigger_type == "sector_cluster":
            sector = trigger.get("sector", "")
            direction = trigger.get("direction", "")
            if sector:
                return BridgeQuery(
                    query_type="explain_move",
                    question=f"What causal pathways explain the {direction} move in {sector}?",
                    context={"sector": sector, "direction": direction},
                    depth="cached",
                )

        return None

    # ── Context accessors for FlowTrace integration ──

    def get_cascade_context_for_ticker(self, ticker: str, sector: str = "") -> dict:
        """Get TidalShift pathway context relevant to a specific ticker/sector.

        Used by consensus calculation and signal renderer.
        """
        relevant_pathways = []
        for pw in reversed(self._cascade_cache):
            entities = pw.get("affected_entities", [])
            sectors = pw.get("affected_sectors", [])
            is_match = (
                ticker.upper() in [e.upper() for e in entities]
                or (sector and sector in sectors)
            )
            if is_match:
                relevant_pathways.append(pw)
            if len(relevant_pathways) >= 5:
                break

        relevant_warnings = [
            w for w in self._cascade_warnings
            if ticker.upper() in [e.upper() for e in w.get("affected_entities", [])]
            or (sector and sector in w.get("affected_sectors", []))
        ]

        if not relevant_pathways and not relevant_warnings:
            return {"available": False}

        return {
            "available": True,
            "relevant_pathways": relevant_pathways,
            "cascade_warnings": relevant_warnings[:5],
            "leading_indicators": [
                ind for ind in self._leading_indicators
                if ind.get("pathway_id") in [pw.get("pathway_id") for pw in relevant_pathways]
            ],
            "source": "TidalShift",
            "last_updated": relevant_pathways[0]["received_at"] if relevant_pathways else None,
        }

    def calculate_bridge_multiplier(self, ticker: str, direction: str, sector: str = "") -> float:
        """Calculate a bridge multiplier for the consensus calculation.

        Range: 0.90 (pathways contradict) to 1.10 (pathways confirm).
        Returns 1.0 when no bridge data is available.
        """
        context = self.get_cascade_context_for_ticker(ticker, sector)
        if not context.get("available"):
            return 1.0

        pathways = context.get("relevant_pathways", [])
        if not pathways:
            return 1.0

        # Check if pathway direction aligns with trade direction
        alignment_scores = []
        for pw in pathways:
            links = pw.get("key_links", [])
            if not links:
                continue
            # Positive magnitude = bullish implication, negative = bearish
            avg_mag = sum(l.get("magnitude", 0) for l in links) / max(len(links), 1)
            pathway_direction = "BULLISH" if avg_mag > 0 else "BEARISH"

            plausibility = pw.get("plausibility", 0.5)
            if pathway_direction == direction:
                alignment_scores.append(plausibility)
            else:
                alignment_scores.append(-plausibility)

        if not alignment_scores:
            return 1.0

        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        # Scale: -1.0 (full contradiction) to +1.0 (full confirmation)
        # Map to multiplier: 0.90 to 1.10
        multiplier = 1.0 + (avg_alignment * 0.10)
        return max(0.90, min(1.10, multiplier))

    def get_recent_received(self, limit: int = 20) -> list[dict]:
        """Get recent received signals formatted for UI."""
        results = []
        for env in reversed(self._received_signals[-limit:]):
            results.append({
                "message_id": env.message_id,
                "timestamp": env.timestamp,
                "source": env.source_node_type,
                "signal_type": env.payload.get("signal_type", env.message_type),
                "summary": env.payload.get("summary", ""),
                "confidence": env.payload.get("confidence", 0),
            })
        return results

    def get_cascade_cache(self) -> list[dict]:
        """Get all cached pathway alerts."""
        return list(self._cascade_cache)

    def get_active_warnings(self) -> list[dict]:
        """Get active cascade warnings."""
        return list(self._cascade_warnings)
