import os
import sqlite3
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# List of Fortune 500 "Hub" companies to seed the graph
# In production, this list would be dynamic or larger.
SEED_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "WMT", "TSLA", "META",
    "PG", "JNJ", "V", "JPM", "UNH", "HD", "MA", "LLY", "PEP", "KO"
]

def init_db():
    """Initialize the SQLite database schema."""
    print(f"[*] Initializing database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table: Companies (Nodes)
    # Stores metadata about both Fortune 500 hubs and small-cap partners
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            industry TEXT,
            market_cap REAL,
            last_updated DATETIME
        )
    ''')

    # Table: Relationships (Edges)
    # Stores the connection between a Hub (source) and a Partner (target)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_ticker TEXT,
            target_ticker TEXT,
            relationship_type TEXT, -- e.g., 'Supplier', 'Customer', 'Partner'
            revenue_dependency REAL, -- Estimated % of revenue target gets from source
            confidence REAL, -- 0.0 to 1.0
            source_origin TEXT, -- e.g., 'Finnhub', 'EDGAR', 'Inferred'
            last_updated DATETIME,
            UNIQUE(source_ticker, target_ticker, relationship_type)
        )
    ''')

    # Table: Signals (Analysis Results)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_ticker TEXT,
            target_ticker TEXT,
            event_type TEXT,
            expected_move_pct REAL,
            confidence REAL,
            unified_score REAL,
            reasoning TEXT,
            summary TEXT,
            agent_data TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("[+] Database schema ready.")

def fetch_finnhub_supply_chain(symbol):
    """
    Fetch supply chain data from Finnhub API.
    Note: This endpoint requires a Finnhub Premium subscription.
    """
    if not FINNHUB_API_KEY:
        print("[-] Error: FINNHUB_API_KEY not found in .env")
        return None

    url = "https://finnhub.io/api/v1/stock/supply-chain"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            print(f"[-] Access Denied for {symbol} (403). Finnhub Premium required for supply-chain endpoint.")
            return None
        else:
            print(f"[-] Error fetching {symbol}: {response.status_code}")
            return None
    except Exception as e:
        print(f"[-] Exception fetching {symbol}: {e}")
        return None

def fetch_finnhub_peers(symbol):
    """
    Fetch company peers (competitors) from Finnhub API.
    """
    if not FINNHUB_API_KEY:
        return []

    url = "https://finnhub.io/api/v1/stock/peers"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Returns a list of strings: ["AAPL", "DELL", "HPQ", ...]
            return response.json()
    except Exception as e:
        print(f"[-] Exception fetching peers for {symbol}: {e}")
    return []

def setup_vector_db():
    """Initialize ChromaDB and Sentence Transformer model."""
    print(f"[*] Initializing Vector Database at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="supply_chain_embeddings")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, model

def seed_database():
    """Fetch data and populate the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    vector_collection, embedding_model = setup_vector_db()

    print(f"[*] Seeding graph for {len(SEED_TICKERS)} hub companies...")
    
    count_hubs = 0
    count_rels = 0

    for hub_ticker in SEED_TICKERS:
        print(f"[*] Processing {hub_ticker}...")
        
        # 1. Ensure Hub exists in Companies table
        cursor.execute('''
            INSERT OR IGNORE INTO companies (ticker, last_updated) 
            VALUES (?, ?)
        ''', (hub_ticker, datetime.now()))
        count_hubs += 1

        # 2. Fetch Supply Chain
        data = fetch_finnhub_supply_chain(hub_ticker)
        
        if not data or 'data' not in data:
            continue

        suppliers = data.get('data', [])
        print(f"    > Found {len(suppliers)} connections for {hub_ticker}")

        for item in suppliers:
            # Finnhub structure: {'symbol': '...', 'name': '...', 'attributes': {...}}
            supp_ticker = item.get('symbol')
            supp_name = item.get('name')
            
            if not supp_ticker:
                continue

            # Insert Supplier into Companies
            cursor.execute('''
                INSERT OR IGNORE INTO companies (ticker, name, last_updated) 
                VALUES (?, ?, ?)
            ''', (supp_ticker, supp_name, datetime.now()))

            # Insert Relationship (Hub <- Supplier)
            # We assume these are suppliers to the Hub (Hub is the Customer)
            cursor.execute('''
                INSERT OR REPLACE INTO relationships 
                (source_ticker, target_ticker, relationship_type, confidence, source_origin, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hub_ticker, supp_ticker, "Supplier", 0.9, "Finnhub", datetime.now()))
            count_rels += 1
            
            # Vector Embedding for Semantic Search
            # Description: "AAPL is a customer of SWKS (Skyworks Solutions). Relationship: Supplier."
            desc = f"{hub_ticker} is a customer of {supp_ticker} ({supp_name}). Relationship: Supplier."
            embedding = embedding_model.encode(desc).tolist()
            vector_collection.add(
                documents=[desc],
                embeddings=[embedding],
                metadatas=[{"source": hub_ticker, "target": supp_ticker, "type": "Supplier"}],
                ids=[f"{hub_ticker}_{supp_ticker}"]
            )

        # 3. Fetch Competitors/Peers
        peers = fetch_finnhub_peers(hub_ticker)
        if peers:
            print(f"    > Found {len(peers)} peers for {hub_ticker}")
            for peer_ticker in peers:
                if peer_ticker == hub_ticker: continue
                
                # Insert Peer into Companies
                cursor.execute('''
                    INSERT OR IGNORE INTO companies (ticker, last_updated) 
                    VALUES (?, ?)
                ''', (peer_ticker, datetime.now()))

                # Insert Relationship (Hub <-> Peer) as "Competitor"
                cursor.execute('''
                    INSERT OR REPLACE INTO relationships 
                    (source_ticker, target_ticker, relationship_type, confidence, source_origin, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (hub_ticker, peer_ticker, "Competitor", 0.85, "Finnhub", datetime.now()))
                count_rels += 1

                # Vector Embedding
                desc = f"{hub_ticker} is a competitor of {peer_ticker}. Relationship: Competitor."
                embedding = embedding_model.encode(desc).tolist()
                vector_collection.add(
                    documents=[desc],
                    embeddings=[embedding],
                    metadatas=[{"source": hub_ticker, "target": peer_ticker, "type": "Competitor"}],
                    ids=[f"{hub_ticker}_{peer_ticker}_comp"]
                )

        conn.commit()
        # Respect API rate limits (Finnhub free is 60/min, premium is higher but good practice)
        time.sleep(0.5)

    conn.close()
    print(f"[+] Seeding complete. Processed {count_hubs} hubs and created {count_rels} relationships.")

if __name__ == "__main__":
    init_db()
    seed_database()