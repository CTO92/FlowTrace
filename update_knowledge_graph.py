import os
import sqlite3
import glob
import json
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")
XAI_API_KEY = os.getenv("XAI_API_KEY")
SEC_EMAIL = os.getenv("SEC_EMAIL", "user@example.com") # Required by SEC API
FILINGS_PATH = os.path.join(os.path.dirname(__file__), "sec_filings")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1"
)

def get_hub_companies():
    """Get list of companies to update."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM companies") # In a real scenario, filter for Hubs
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

def download_latest_filing(ticker):
    """Download latest 10-Q or 10-K."""
    dl = Downloader("FlowTrace_Research", SEC_EMAIL, FILINGS_PATH)
    try:
        # Try 10-Q first (Quarterly)
        print(f"[*] Downloading latest 10-Q for {ticker}...")
        dl.get("10-Q", ticker, limit=1)
        
        # Find the file
        path = os.path.join(FILINGS_PATH, "sec-edgar-filings", ticker, "10-Q", "*", "*.html")
        files = glob.glob(path)
        if files:
            return files[0]
        return None
    except Exception as e:
        print(f"[-] Error downloading {ticker}: {e}")
        return None

def extract_text(file_path):
    """Parse HTML filing to text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            # Remove tables to save tokens (optional, but tables often break LLM context)
            # for table in soup(["table"]): table.decompose()
            text = soup.get_text(separator="\n")
            return text[:100000] # Truncate to fit context window if needed
    except Exception as e:
        print(f"[-] Error parsing {file_path}: {e}")
        return ""

def extract_relationships_with_grok(ticker, text):
    """Ask Grok to find suppliers in the text."""
    prompt = f"""
    Analyze the following SEC filing text for {ticker}.
    Identify any mentioned SUPPLIERS, VENDORS, or STRATEGIC PARTNERS.
    Focus on smaller companies that rely on {ticker}.
    
    Return JSON only:
    {{
        "relationships": [
            {{"target_ticker": "XYZ", "name": "XYZ Corp", "type": "Supplier", "confidence": 0.9}}
        ]
    }}
    
    If none found, return empty list.
    
    TEXT EXCERPT:
    {text[:50000]}...
    """
    
    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are a financial data extractor."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[-] Grok Error: {e}")
        return {"relationships": []}

def update_db(source_ticker, relationships):
    """Update the database with new edges."""
    if not relationships: return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    for rel in relationships:
        target = rel.get('target_ticker')
        if target and target != source_ticker:
            cursor.execute('''
                INSERT OR REPLACE INTO relationships 
                (source_ticker, target_ticker, relationship_type, confidence, source_origin, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (source_ticker, target, rel.get('type', 'Partner'), rel.get('confidence', 0.5), "EDGAR_10Q", datetime.now()))
            count += 1
            
    conn.commit()
    conn.close()
    print(f"    [+] Updated {count} relationships for {source_ticker}")

def main():
    print("--- Quarterly Knowledge Graph Update (EDGAR) ---")
    hubs = get_hub_companies()
    
    for ticker in hubs:
        file_path = download_latest_filing(ticker)
        if file_path:
            text = extract_text(file_path)
            if text:
                print(f"    [*] Analyzing text ({len(text)} chars)...")
                data = extract_relationships_with_grok(ticker, text)
                rels = data.get("relationships", [])
                update_db(ticker, rels)

if __name__ == "__main__":
    main()