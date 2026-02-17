import os
import asyncio
import sqlite3
from datetime import datetime
from polygon import WebSocketClient, RESTClient
from dotenv import load_dotenv
from grok_analysis import analyze_impact
from agent_workflow import run_research_task
from plyer import notification

# Load environment variables
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")

# Initialize REST Client for snapshots
rest_client = RESTClient(api_key=POLYGON_API_KEY)

# Event Keywords for Pre-filtering (Simple heuristic)
EVENT_KEYWORDS = [
    "contract", "agreement", "partnership", "deal", "supply", "supplier",
    "order", "award", "selected", "expansion", "earnings", "guidance",
    "beat", "miss", "revenue", "strategic", "acquisition", "merger"
]

def get_connected_small_caps(ticker):
    """
    Query the knowledge graph for small-cap partners associated with the given F500 ticker.
    """
    if not os.path.exists(DB_PATH):
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find targets where source is the ticker (Hub -> Partner)
        query = """
            SELECT r.target_ticker, r.relationship_type, c.name
            FROM relationships r
            LEFT JOIN companies c ON r.target_ticker = c.ticker
            WHERE r.source_ticker = ?
        """
        
        cursor.execute(query, (ticker,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "ticker": row["target_ticker"],
                "name": row["name"],
                "relationship": row["relationship_type"]
            })
        return results
    except Exception as e:
        print(f"[-] DB Error: {e}")
        return []

def save_signals_to_db(source_ticker, analysis_result, agent_data):
    """
    Save the analysis results to the SQLite database.
    """
    if not os.path.exists(DB_PATH): return
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        summary = analysis_result.get('analysis_summary', '')
        timestamp = datetime.now()
        
        for target in analysis_result.get('targets', []):
            cursor.execute('''
                INSERT INTO signals (
                    timestamp, source_ticker, target_ticker, event_type, 
                    expected_move_pct, confidence, unified_score, 
                    reasoning, summary, agent_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, source_ticker, target.get('ticker'), target.get('event_type'),
                target.get('expected_move_pct'), target.get('confidence'), 
                target.get('unified_correlation_score'), target.get('reasoning'),
                summary, str(agent_data)
            ))
        conn.commit()
        conn.close()
        print("    [+] Signals saved to database.")
    except Exception as e:
        print(f"[-] DB Save Error: {e}")

def fetch_market_snapshot(tickers):
    """
    Fetch real-time price, volume, and change % for a list of tickers.
    """
    data = {}
    try:
        for t in tickers:
            # Fetch snapshot (synchronous call, acceptable for prototype)
            snap = rest_client.get_snapshot_ticker(None, t)
            if snap:
                data[t] = {
                    "price": getattr(snap.day, 'close', 0),
                    "change_pct": getattr(snap, 'todays_change_percent', 0),
                    "volume": getattr(snap.day, 'volume', 0)
                }
    except Exception as e:
        print(f"[-] Snapshot Error: {e}")
    return data

async def process_news_item(item):
    """
    Process a single news item from the WebSocket.
    """
    tickers = getattr(item, "tickers", [])
    title = getattr(item, "title", "")
    description = getattr(item, "description", "")
    
    if not tickers or not title:
        return

    # 1. Keyword Filter
    full_text = (f"{title} {description}").lower()
    
    if not any(kw in full_text for kw in EVENT_KEYWORDS):
        return

    print(f"\n[!] NEWS ALERT: {title}")
    print(f"    Tickers: {tickers}")

    # 2. Graph Lookup
    found_impact = False
    for ticker in tickers:
        partners = get_connected_small_caps(ticker)
        if partners:
            found_impact = True
            print(f"    [+] IMPACT DETECTED for {ticker}. Connected Small-Caps:")
            for p in partners:
                print(f"        -> {p['ticker']} ({p['relationship']}) - {p['name']}")
            
            # Trigger Agentic Research
            print(f"    [*] Launching Agentic Research for context...")
            research_query = f"Investigate the relationship and recent news between {ticker} and {', '.join([p['ticker'] for p in partners])}."
            agent_findings = await run_research_task(research_query)

            # Fetch Market Context (Price/Volume)
            all_tickers = [ticker] + [p['ticker'] for p in partners]
            market_data = fetch_market_snapshot(all_tickers)
            print(f"    [*] Fetched market data for {len(market_data)} tickers.")

            print(f"    [*] Triggering Grok Analysis for {len(partners)} targets...")
            analysis_result = await analyze_impact(ticker, partners, item, agent_findings, market_data)
            
            if analysis_result:
                print("\n    [Grok Analysis Result]")
                print(f"    Summary: {analysis_result.get('analysis_summary')}")
                for target in analysis_result.get('targets', []):
                    print(f"    -> Target: {target['ticker']}")
                    print(f"       Exp Move: {target['expected_move_pct']}% (Conf: {target['confidence']}%)")
                    print(f"       Corr Score: {target['unified_correlation_score']}")
                    print(f"       Reasoning: {target['reasoning']}")
                
                # Trigger Desktop Notification for High Confidence Signals
                conf = target.get('confidence', 0)
                if conf >= 80:
                    try:
                        notification.notify(
                            title=f"⚡ High Conviction: {target['ticker']}",
                            message=f"Source: {ticker} | Exp Move: {target['expected_move_pct']}% | Conf: {conf}%",
                            app_name="FlowTrace",
                            timeout=10
                        )
                    except Exception as e:
                        # This may fail in Docker/Headless environments
                        print(f"    [-] Notification failed (headless?): {e}")

                # Save to DB for Dashboard
                save_signals_to_db(ticker, analysis_result, agent_findings)
    
    if not found_impact:
        print("    [-] No direct supply chain connections found in graph.")

async def main():
    if not POLYGON_API_KEY:
        print("[-] Error: POLYGON_API_KEY not set in .env")
        return

    if not os.path.exists(DB_PATH):
        print(f"[-] Warning: Database not found at {DB_PATH}. Run Phase 2 first.")

    print("[*] Starting Polygon.io News Listener...")
    print("[*] Filtering for keywords: " + ", ".join(EVENT_KEYWORDS[:5]) + "...")

    # Subscribe to all news ('N.*')
    client = WebSocketClient(api_key=POLYGON_API_KEY, subscriptions=["N.*"])

    # Async iterator
    async for msgs in client:
        for msg in msgs:
            await process_news_item(msg)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Listener stopped.")