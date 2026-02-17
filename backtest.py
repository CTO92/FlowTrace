import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from polygon import RESTClient
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")

# Configuration
LOOKBACK_DAYS = 90  # How far back to test
MAX_EVENTS = 50     # Limit to avoid API rate limits/costs during testing
EVENT_KEYWORDS = [
    "contract", "agreement", "partnership", "deal", "supply", "supplier",
    "order", "award", "selected", "expansion", "earnings"
]

def get_db_connection():
    if not os.path.exists(DB_PATH):
        print(f"[-] Database not found at {DB_PATH}")
        return None
    return sqlite3.connect(DB_PATH)

def get_hub_companies():
    """Get all F500 companies currently in the graph."""
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT source_ticker FROM relationships")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

def get_connected_targets(source_ticker):
    """Get small-cap partners for a hub."""
    conn = get_db_connection()
    if not conn: return []
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT target_ticker, relationship_type FROM relationships WHERE source_ticker = ?", (source_ticker,))
    rows = cursor.fetchall()
    conn.close()
    return [{"ticker": r["target_ticker"], "type": r["relationship_type"]} for r in rows]

def fetch_historical_news(client, ticker, start_date, end_date):
    """Fetch news for a ticker within a date range."""
    news = []
    try:
        # Polygon list_ticker_news is an iterator
        # We limit to 10 per ticker for the backtest sample
        resp = client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=start_date,
            published_utc_lte=end_date,
            limit=10
        )
        for n in resp:
            news.append(n)
    except Exception as e:
        print(f"[-] Error fetching news for {ticker}: {e}")
    return news

def get_price_data(client, ticker, event_date_str):
    """
    Fetch OHLCV data for T-1 to T+5 days around the event.
    """
    event_dt = datetime.strptime(event_date_str[:10], "%Y-%m-%d")
    start_dt = event_dt - timedelta(days=5) # Get some context
    end_dt = event_dt + timedelta(days=10)  # Look forward
    
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_dt.strftime("%Y-%m-%d"),
            to=end_dt.strftime("%Y-%m-%d")
        )
        
        data = []
        for agg in aggs:
            # Polygon timestamp is ms
            date = datetime.fromtimestamp(agg.timestamp / 1000).strftime("%Y-%m-%d")
            data.append({
                "date": date,
                "close": agg.close,
                "volume": agg.volume
            })
        return pd.DataFrame(data)
    except Exception as e:
        # print(f"[-] Error fetching price for {ticker}: {e}")
        return pd.DataFrame()

def calculate_event_impact(df, event_date_str):
    """
    Calculate returns:
    - 1-Day Return (Close T+1 / Close T-1) - 1
    - 5-Day Return (Close T+5 / Close T-1) - 1
    """
    if df.empty:
        return None
    
    event_date = event_date_str[:10]
    
    # Find index of event date
    try:
        # Ensure date column is string for comparison
        df['date'] = df['date'].astype(str)
        
        # Locate event row or closest previous
        mask = df['date'] >= event_date
        if not mask.any(): return None
        
        event_idx = df[mask].index[0]
        
        # We need T-1 (baseline)
        if event_idx == 0: return None # No prior data
        
        baseline_price = df.iloc[event_idx - 1]['close']
        
        # T+1
        ret_1d = None
        if event_idx + 1 < len(df):
            ret_1d = (df.iloc[event_idx + 1]['close'] - baseline_price) / baseline_price
            
        # T+5
        ret_5d = None
        if event_idx + 5 < len(df):
            ret_5d = (df.iloc[event_idx + 5]['close'] - baseline_price) / baseline_price
            
        return {
            "baseline": baseline_price,
            "1d_return": ret_1d,
            "5d_return": ret_5d
        }
        
    except Exception as e:
        print(f"Error calc impact: {e}")
        return None

def run_backtest():
    print("--- FlowTrace: Phase 6 Backtest ---")
    
    if not POLYGON_API_KEY:
        print("[-] POLYGON_API_KEY missing.")
        return

    client = RESTClient(api_key=POLYGON_API_KEY)
    
    hubs = get_hub_companies()
    if not hubs:
        print("[-] No hub companies found in DB. Run Phase 2 first.")
        return
        
    print(f"[*] Found {len(hubs)} Hub companies. Fetching historical data (Last {LOOKBACK_DAYS} days)...")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    
    results = []
    events_processed = 0
    
    for hub in hubs:
        if events_processed >= MAX_EVENTS: break
        
        print(f"[*] Checking news for {hub}...")
        news_items = fetch_historical_news(client, hub, start_date, end_date)
        
        for item in news_items:
            if events_processed >= MAX_EVENTS: break
            
            # Filter keywords
            title = getattr(item, "title", "") or ""
            desc = getattr(item, "description", "") or ""
            text = (title + " " + desc).lower()
            
            if not any(kw in text for kw in EVENT_KEYWORDS):
                continue
                
            # Check for partners
            targets = get_connected_targets(hub)
            if not targets:
                continue
                
            pub_date = getattr(item, "published_utc", "")
            if not pub_date: continue
            
            print(f"    > Event Found: {pub_date[:10]} - {title[:50]}...")
            
            for target in targets:
                ticker = target['ticker']
                # Fetch price data
                df_price = get_price_data(client, ticker, pub_date)
                
                impact = calculate_event_impact(df_price, pub_date)
                
                if impact:
                    res = {
                        "event_date": pub_date[:10],
                        "source": hub,
                        "target": ticker,
                        "relationship": target['type'],
                        "1d_return": impact['1d_return'],
                        "5d_return": impact['5d_return']
                    }
                    results.append(res)
                    print(f"      -> {ticker}: 1D={res['1d_return']:.2%} | 5D={res['5d_return']:.2%}")
            
            events_processed += 1
            
    # Summary
    if not results:
        print("[-] No valid events or price data found in sample.")
        return

    df_res = pd.DataFrame(results)
    print("\n--- Backtest Results ---")
    print(f"Total Events Analyzed: {len(df_res)}")
    
    # Clean None values
    df_res = df_res.dropna(subset=['1d_return'])
    
    if df_res.empty:
        print("[-] No price data available for calculated events.")
        return

    avg_1d = df_res['1d_return'].mean()
    avg_5d = df_res['5d_return'].mean()
    win_rate = (df_res['1d_return'] > 0).mean()
    
    print(f"Average 1-Day Return: {avg_1d:.2%}")
    print(f"Average 5-Day Return: {avg_5d:.2%}")
    print(f"Win Rate (Positive 1D): {win_rate:.2%}")
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"[+] Detailed results saved to {csv_path}")

if __name__ == "__main__":
    run_backtest()