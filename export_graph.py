import os
import sqlite3
import pandas as pd
import webbrowser

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exports")

def export_for_gephi():
    """
    Exports the knowledge graph from SQLite into Gephi-compatible CSV files (nodes and edges).
    """
    if not os.path.exists(DB_PATH):
        print(f"[-] Database not found at {DB_PATH}. Please run build_knowledge_graph.py first.")
        return

    print(f"[*] Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # 1. Export Nodes
    print("[*] Exporting nodes (companies)...")
    nodes_query = "SELECT ticker, name, sector, industry, market_cap FROM companies"
    nodes_df = pd.read_sql_query(nodes_query, conn)
    
    # Rename columns for Gephi import
    nodes_df.rename(columns={'ticker': 'Id', 'name': 'Label'}, inplace=True)
    
    # Ensure no duplicate nodes
    nodes_df.drop_duplicates(subset=['Id'], inplace=True)

    # 2. Export Edges
    print("[*] Exporting edges (relationships)...")
    edges_query = "SELECT source_ticker, target_ticker, relationship_type, confidence FROM relationships"
    edges_df = pd.read_sql_query(edges_query, conn)
    
    # Rename columns for Gephi import
    edges_df.rename(columns={
        'source_ticker': 'Source',
        'target_ticker': 'Target',
        'relationship_type': 'Label',
        'confidence': 'Weight'
    }, inplace=True)
    
    # Add a 'Type' column for directed graph
    edges_df['Type'] = 'Directed'

    conn.close()

    # 3. Save to CSV
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        print(f"[+] Created export directory: {EXPORT_DIR}")

    nodes_path = os.path.join(EXPORT_DIR, "gephi_nodes.csv")
    edges_path = os.path.join(EXPORT_DIR, "gephi_edges.csv")

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    print("\n--- Success! ---")
    print(f"[+] Nodes file saved to: {nodes_path}")
    print(f"[+] Edges file saved to: {edges_path}")
    print("\nTo visualize in Gephi, open the 'Data Laboratory', click 'Import Spreadsheet', and import the nodes file first, then the edges file.")

def visualize_with_pyvis():
    """
    Generates an interactive HTML visualization of the graph using Pyvis.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("\n[-] Pyvis not installed. Skipping interactive visualization.")
        print("    Run: pip install pyvis")
        return

    if not os.path.exists(DB_PATH):
        return

    print(f"\n[*] Generating Pyvis visualization...")
    conn = sqlite3.connect(DB_PATH)
    
    # Fetch data
    nodes_df = pd.read_sql_query("SELECT * FROM companies", conn)
    edges_df = pd.read_sql_query("SELECT * FROM relationships", conn)
    conn.close()

    # Initialize Network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote')
    
    # Add Nodes
    for _, row in nodes_df.iterrows():
        label = row['ticker']
        title = f"{row['name']}<br>Sector: {row.get('sector', 'N/A')}<br>Cap: {row.get('market_cap', 'N/A')}"
        # Color hubs (Fortune 500) differently if possible, but for now generic green
        net.add_node(row['ticker'], label=label, title=title, color="#00ff00")

    # Add Edges
    for _, row in edges_df.iterrows():
        title = f"{row['relationship_type']} (Conf: {row['confidence']})"
        net.add_edge(row['source_ticker'], row['target_ticker'], title=title, color="#aaaaaa")

    # Physics
    net.barnes_hut()
    
    # Save and Open
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        
    output_path = os.path.join(EXPORT_DIR, "graph_viz.html")
    net.save_graph(output_path)
    
    print(f"[+] Interactive graph saved to: {output_path}")
    print("[*] Opening in browser...")
    webbrowser.open('file://' + os.path.abspath(output_path))

if __name__ == "__main__":
    export_for_gephi()
    visualize_with_pyvis()