"""
FlowTrace ContinuousMonitorAgent

Orchestrates the continuous autonomous loop:
  ingest news -> analyze -> check forum -> debate -> synthesize -> signal

Keeps the system running without trader intervention. The trader starts
the application and walks away.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone, timedelta

from node_identity import load_identity, get_node_id, get_node_alias, get_forum_status
from forum_config import is_forum_configured
from learning_config_manager import load_config, get_participation_intensity, get_intensity_thresholds
from agent_learning import run_learning_review, resolve_open_signals
from agent_consensus import process_raw_signals, get_open_consensus_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ContinuousMonitorAgent:
    """
    Orchestrator that manages all background processes:
    - News ingestion (via ingestion_listener)
    - Forum monitoring (via ForumScoutAgent - Phase 3)
    - Thesis publication (via ThesisAgent - Phase 3)
    - Debate participation (via DebateAgent - Phase 3)
    - Learning reviews (via LearningAgent)
    - Signal consensus (via ConsensusAgent)
    - Health checks
    """

    def __init__(self):
        self.identity = load_identity()
        self.config = load_config()
        self.running = False
        self._tasks = []
        self._last_learning_review = None
        self._last_health_check = None
        self._last_forum_scan = None
        self._last_signal_process = None
        self._last_swarm_cycle = None
        self._error_counts = {}
        self._start_time = None
        self._latest_swarm_brief = None

    async def start(self):
        """Start all background monitoring loops."""
        self.running = True
        self._start_time = datetime.now(timezone.utc)

        logger.info("=" * 60)
        logger.info("FlowTrace ContinuousMonitorAgent Starting")
        logger.info(f"  Node ID:    {self.identity['node_id']}")
        logger.info(f"  Node Alias: {self.identity['node_alias']}")
        logger.info(f"  Forum:      {'configured' if is_forum_configured() else 'not configured (local-only mode)'}")
        logger.info(f"  Intensity:  {get_participation_intensity()}")
        logger.info("=" * 60)

        # Launch concurrent loops
        self._tasks = [
            asyncio.create_task(self._ingestion_loop()),
            asyncio.create_task(self._signal_processing_loop()),
            asyncio.create_task(self._learning_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

        # Forum loops only if configured (Phase 3)
        if is_forum_configured():
            self._tasks.append(asyncio.create_task(self._forum_loop()))

        # Trading Agent Swarm loop
        self._tasks.append(asyncio.create_task(self._swarm_loop()))

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("ContinuousMonitorAgent shutting down...")

    async def stop(self):
        """Gracefully stop all background loops."""
        self.running = False
        for task in self._tasks:
            task.cancel()
        logger.info("ContinuousMonitorAgent stopped.")

    # --- Core Loops ---

    async def _ingestion_loop(self):
        """
        Run the news ingestion listener.
        This wraps the existing ingestion_listener.py as a module.
        """
        while self.running:
            try:
                from ingestion_listener import start_listener
                logger.info("[Ingestion] Starting news listener...")
                await start_listener()
            except ImportError:
                logger.warning("[Ingestion] ingestion_listener.start_listener not available yet.")
                await asyncio.sleep(60)
            except Exception as e:
                self._record_error("ingestion", str(e))
                logger.error(f"[Ingestion] Error: {e}. Restarting in 30s...")
                await asyncio.sleep(30)

    async def _signal_processing_loop(self):
        """
        Periodically process raw signals through the ConsensusAgent
        to produce weighted consensus signals.
        """
        import sqlite3

        kg_db = os.path.join(BASE_DIR, "knowledge_graph.db")

        while self.running:
            try:
                await asyncio.sleep(120)  # every 2 minutes

                if not os.path.exists(kg_db):
                    continue

                conn = sqlite3.connect(kg_db)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Fetch recent unprocessed signals (last hour, high confidence)
                one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                cursor.execute("""
                    SELECT * FROM signals
                    WHERE timestamp >= ? AND confidence >= 60
                    ORDER BY timestamp DESC
                    LIMIT 20
                """, (one_hour_ago,))

                raw_signals = [dict(row) for row in cursor.fetchall()]
                conn.close()

                if raw_signals:
                    emitted = process_raw_signals(raw_signals)
                    if emitted:
                        logger.info(f"[Consensus] Emitted {len(emitted)} consensus signals")
                        self._last_signal_process = datetime.now(timezone.utc)

                        # Send notifications for high-score signals
                        for sig in emitted:
                            if sig["consensus_score"] >= 0.85:
                                await self._notify_trader(sig)

            except Exception as e:
                self._record_error("signal_processing", str(e))
                logger.error(f"[Consensus] Error: {e}")
                await asyncio.sleep(60)

    async def _learning_loop(self):
        """
        Run learning reviews on schedule:
        - Resolve open signals continuously
        - Full weight adjustment daily after market close
        """
        while self.running:
            try:
                # Resolve matured signals every 30 minutes
                await asyncio.sleep(1800)

                resolve_result = resolve_open_signals()
                if resolve_result["resolved"] > 0:
                    logger.info(f"[Learning] Resolved {resolve_result['resolved']} signals")

                # Full learning review once daily (check if 24h since last review)
                now = datetime.now(timezone.utc)
                if (self._last_learning_review is None or
                        (now - self._last_learning_review) > timedelta(hours=24)):

                    logger.info("[Learning] Running daily learning review...")
                    summary = await run_learning_review()
                    self._last_learning_review = now

                    adjustments = summary.get("latest_adjustments", {})
                    if adjustments:
                        logger.info(f"[Learning] Made {len(adjustments)} weight adjustments")
                    else:
                        logger.info("[Learning] No adjustments needed")

            except Exception as e:
                self._record_error("learning", str(e))
                logger.error(f"[Learning] Error: {e}")
                await asyncio.sleep(300)

    async def _forum_loop(self):
        """
        Forum participation loop.
        Scans forum, publishes theses, participates in debates.
        """
        from agent_forum_scout import scan_relevant_threads
        from agent_thesis import publish_high_confidence_signals, monitor_thesis_responses
        from agent_debate import engage_with_threads, respond_to_challenges
        from forum_client import get_forum_client

        # Ensure node is registered
        client = get_forum_client()
        if get_forum_status() == "unregistered":
            await client.register_node()

        thresholds = get_intensity_thresholds()
        scan_interval = thresholds["forum_scan_interval"]

        published_thread_ids = []

        while self.running:
            try:
                await asyncio.sleep(scan_interval)

                forum_status = get_forum_status()
                if forum_status != "active":
                    # Send heartbeat to check if we've been approved
                    await client.heartbeat()
                    logger.debug(f"[Forum] Node status is '{forum_status}', waiting for approval")
                    continue

                # 1. Publish local high-confidence signals as theses
                new_threads = await publish_high_confidence_signals()
                if new_threads:
                    published_thread_ids.extend(new_threads)
                    logger.info(f"[Forum] Published {len(new_threads)} new theses")

                # 2. Scan for relevant threads
                relevant = await scan_relevant_threads()
                if relevant:
                    logger.info(f"[Forum] Found {len(relevant)} relevant threads")

                    # 3. Evaluate and debate relevant threads
                    responses = await engage_with_threads(relevant)
                    if responses:
                        logger.info(f"[Forum] Posted {len(responses)} debate responses")

                # 4. Check for challenges to our published theses
                if published_thread_ids:
                    challenges = await monitor_thesis_responses(published_thread_ids[-50:])
                    if challenges:
                        counter_responses = await respond_to_challenges(challenges)
                        logger.info(f"[Forum] Responded to {len(counter_responses)} challenges")

                self._last_forum_scan = datetime.now(timezone.utc)

            except Exception as e:
                self._record_error("forum", str(e))
                logger.error(f"[Forum] Error: {e}")
                await asyncio.sleep(60)

    async def _swarm_loop(self):
        """
        Continuous Trading Agent Swarm simulation loop.
        Runs alongside all other loops. Generates personas on first run,
        then continuously runs simulation rounds, synthesizes output,
        and feeds results into the consensus pipeline.
        """
        from swarm_config import load_swarm_config, is_swarm_enabled, get_simulation_params
        from swarm_persona_generator import generate_personas, load_personas, save_personas, personas_exist
        from swarm_trading_floor import TradingFloor
        from swarm_synthesizer import synthesize_cycle, format_swarm_signals_for_consensus
        from swarm_evolutionary import run_evolution_cycle

        if not is_swarm_enabled():
            logger.info("[Swarm] Trading Agent Swarm is disabled in config. Skipping.")
            return

        config = load_swarm_config()

        # Initialize personas (generate on first run, load thereafter)
        try:
            if personas_exist():
                personas = load_personas()
                logger.info(f"[Swarm] Loaded {len(personas)} existing personas")
            else:
                logger.info(f"[Swarm] Generating {config['swarm_size']} initial personas...")
                personas = await generate_personas(config["swarm_size"], config.get("archetypes"))
                save_personas(personas)
                logger.info(f"[Swarm] Generated and saved {len(personas)} personas")
        except Exception as e:
            self._record_error("swarm", str(e))
            logger.error(f"[Swarm] Persona initialization failed: {e}")
            return

        # Initialize trading floor
        floor = TradingFloor()
        await floor.initialize(personas)

        cycle_number = 0

        while self.running:
            try:
                sim_params = get_simulation_params()
                round_interval = sim_params["round_interval_seconds"]
                max_rounds = sim_params["max_rounds_per_cycle"]

                cycle_number += 1
                floor.current_cycle = cycle_number
                logger.info(f"[Swarm] Starting cycle {cycle_number} ({max_rounds} rounds, {round_interval}s interval)")

                for round_num in range(max_rounds):
                    if not self.running:
                        break

                    # Gather current market context from existing signals
                    market_context = self._get_swarm_market_context()

                    # Run one simulation round
                    round_summary = await floor.run_round(market_context)

                    # Resolve any matured positions
                    tickers = round_summary.get("tickers", [])
                    if tickers:
                        market_data = self._get_current_prices(tickers)
                        await floor.resolve_positions(market_data)

                    await asyncio.sleep(round_interval)

                # Cycle complete — synthesize output
                swarm_brief, swarm_signals = synthesize_cycle(floor.db_path, cycle_number)

                # Store swarm brief for Supervisor access
                self._latest_swarm_brief = swarm_brief

                # Feed swarm signals into consensus pipeline
                if swarm_signals:
                    formatted = format_swarm_signals_for_consensus(swarm_signals)
                    emitted = process_raw_signals(formatted)
                    if emitted:
                        logger.info(f"[Swarm] Cycle {cycle_number}: {len(emitted)} signals fed to consensus")

                # Run evolutionary cycle (prune/promote)
                try:
                    evolution_result = run_evolution_cycle(floor.db_path)
                    pruned = len(evolution_result.get("pruned", []))
                    promoted = len(evolution_result.get("promoted", []))
                    if pruned > 0 or promoted > 0:
                        logger.info(f"[Swarm] Evolution: pruned {pruned}, promoted {promoted}")
                        # Reload personas after evolution
                        if personas_exist():
                            personas = load_personas()
                            floor._personas = personas
                            floor._leader_ids = floor._select_leaders(personas, len(personas))
                except Exception as e:
                    logger.warning(f"[Swarm] Evolution cycle error: {e}")

                self._last_swarm_cycle = datetime.now(timezone.utc)

            except Exception as e:
                self._record_error("swarm", str(e))
                logger.error(f"[Swarm] Error in cycle {cycle_number}: {e}")
                await asyncio.sleep(60)

    def _get_swarm_market_context(self) -> dict:
        """
        Build market context dict for the swarm from existing signals
        and knowledge graph data.
        """
        import sqlite3

        kg_db = os.path.join(BASE_DIR, "knowledge_graph.db")
        context = {
            "active_tickers": [],
            "ticker_data": {},
            "ticker_sectors": {},
        }

        if not os.path.exists(kg_db):
            return context

        try:
            conn = sqlite3.connect(kg_db)
            conn.row_factory = sqlite3.Row

            # Get tickers from recent signals
            one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
            rows = conn.execute("""
                SELECT DISTINCT target_ticker FROM signals
                WHERE timestamp >= ?
                LIMIT 30
            """, (one_day_ago,)).fetchall()

            tickers = [r["target_ticker"] for r in rows if r["target_ticker"]]
            context["active_tickers"] = tickers

            # Get basic data per ticker from recent signals
            for ticker in tickers:
                row = conn.execute("""
                    SELECT expected_move_pct, confidence
                    FROM signals
                    WHERE target_ticker = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (ticker,)).fetchone()

                if row:
                    context["ticker_data"][ticker] = {
                        "change_pct": row["expected_move_pct"] or 0,
                        "sentiment": (row["confidence"] or 50) / 100 - 0.5,
                        "price": None,  # Filled by yfinance if available
                    }

            # Try to get current prices
            try:
                import yfinance as yf
                if tickers:
                    data = yf.download(tickers[:10], period="1d", progress=False)
                    if not data.empty and "Close" in data.columns:
                        for ticker in tickers[:10]:
                            try:
                                price = float(data["Close"][ticker].iloc[-1])
                                if ticker in context["ticker_data"]:
                                    context["ticker_data"][ticker]["price"] = price
                                else:
                                    context["ticker_data"][ticker] = {"price": price, "change_pct": 0, "sentiment": 0}
                            except Exception:
                                pass
            except ImportError:
                pass

            conn.close()
        except Exception as e:
            logger.debug(f"[Swarm] Market context build error: {e}")

        return context

    def _get_current_prices(self, tickers: list) -> dict:
        """Fetch current prices for a list of tickers via yfinance."""
        prices = {}
        try:
            import yfinance as yf
            if tickers:
                data = yf.download(tickers[:20], period="1d", progress=False)
                if not data.empty and "Close" in data.columns:
                    for ticker in tickers:
                        try:
                            price = float(data["Close"][ticker].iloc[-1])
                            prices[ticker] = {"price": price}
                        except Exception:
                            pass
        except ImportError:
            pass
        return prices

    async def _health_check_loop(self):
        """Monitor system health every 5 minutes."""
        while self.running:
            try:
                await asyncio.sleep(300)

                health = self.get_health_status()
                self._last_health_check = datetime.now(timezone.utc)

                # Log warnings for any issues
                for component, status in health["components"].items():
                    if status == "error":
                        logger.warning(f"[Health] {component} has errors")

                # Alert if too many errors
                total_errors = sum(self._error_counts.values())
                if total_errors > 50:
                    logger.error(f"[Health] High error count: {total_errors} total errors")
                    try:
                        from plyer import notification
                        notification.notify(
                            title="FlowTrace Health Alert",
                            message=f"System experiencing errors ({total_errors} total). Check logs.",
                            app_name="FlowTrace",
                            timeout=10,
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"[Health] Health check error: {e}")
                await asyncio.sleep(300)

    # --- Helpers ---

    async def _notify_trader(self, signal: dict):
        """Send a desktop notification for a high-priority signal."""
        try:
            from plyer import notification
            notification.notify(
                title=f"FlowTrace: {signal['direction']} {signal['ticker']}",
                message=(
                    f"Score: {signal['consensus_score']:.0%} | "
                    f"Expected: {signal['expected_move_pct']:+.1f}% | "
                    f"Horizon: {signal['time_horizon_days']}d"
                ),
                app_name="FlowTrace",
                timeout=15,
            )
        except Exception as e:
            logger.debug(f"Notification failed (headless?): {e}")

    def _record_error(self, component: str, error: str):
        """Track error counts per component."""
        self._error_counts[component] = self._error_counts.get(component, 0) + 1

    def get_health_status(self) -> dict:
        """Return current system health for the dashboard."""
        config = load_config()

        def _component_status(component):
            errors = self._error_counts.get(component, 0)
            if errors == 0:
                return "healthy"
            elif errors < 5:
                return "warning"
            else:
                return "error"

        uptime = None
        if self._start_time:
            uptime = str(datetime.now(timezone.utc) - self._start_time)

        return {
            "running": self.running,
            "uptime": uptime,
            "node_id": self.identity["node_id"],
            "node_alias": self.identity["node_alias"],
            "forum_configured": is_forum_configured(),
            "forum_status": get_forum_status(),
            "participation_intensity": get_participation_intensity(),
            "components": {
                "ingestion": _component_status("ingestion"),
                "signal_processing": _component_status("signal_processing"),
                "learning": _component_status("learning"),
                "forum": _component_status("forum") if is_forum_configured() else "not_configured",
                "swarm": _component_status("swarm"),
                "health": "healthy",
            },
            "last_learning_review": self._last_learning_review.isoformat() if self._last_learning_review else None,
            "last_forum_scan": self._last_forum_scan.isoformat() if self._last_forum_scan else None,
            "last_signal_process": self._last_signal_process.isoformat() if self._last_signal_process else None,
            "last_swarm_cycle": self._last_swarm_cycle.isoformat() if self._last_swarm_cycle else None,
            "swarm_brief": self._latest_swarm_brief,
            "error_counts": dict(self._error_counts),
            "performance": config.get("performance_history", {}),
        }


# --- Singleton instance for use by app.py and other modules ---
_monitor = None


def get_monitor() -> ContinuousMonitorAgent:
    """Get or create the singleton monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ContinuousMonitorAgent()
    return _monitor


async def start_monitor():
    """Start the continuous monitor (entry point)."""
    monitor = get_monitor()
    await monitor.start()


if __name__ == "__main__":
    try:
        asyncio.run(start_monitor())
    except KeyboardInterrupt:
        print("\n[!] ContinuousMonitorAgent stopped.")
