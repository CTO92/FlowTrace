import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import time
import asyncio
from agent_workflow import run_research_task

# --- Configuration ---
st.set_page_config(
    page_title="FlowTrace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    .signal-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM signals ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()

# --- Sidebar ---
st.sidebar.title("⚡ FlowTrace")
st.sidebar.markdown("---")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
min_confidence = st.sidebar.slider("Min Confidence %", 0, 100, 70)
min_score = st.sidebar.slider("Min Unified Score", 0.0, 1.0, 0.7)

if st.sidebar.button("Refresh Data"):
    st.rerun()

# --- Manual Agent Trigger ---
st.sidebar.markdown("---")
st.sidebar.subheader("🕵️ Manual Research")
research_ticker = st.sidebar.text_input("Target Ticker", placeholder="e.g. NVDA")
research_query = st.sidebar.text_area("Research Query", placeholder="Find recent supplier contracts...")

if st.sidebar.button("Launch Agent"):
    if research_ticker and research_query:
        with st.sidebar.status(f"Agent researching {research_ticker}...") as status:
            full_query = f"Focusing on {research_ticker}: {research_query}"
            result = asyncio.run(run_research_task(full_query))
            status.update(label="Research Complete", state="complete")
            st.sidebar.markdown(f"**Findings:**\n{result}")

# --- Main Content ---
st.title("Supply Chain Event Monitor")

df = load_data()

if df.empty:
    st.info("No signals generated yet. Waiting for ingestion listener...")
else:
    # Filter Data
    df_filtered = df[
        (df['confidence'] >= min_confidence) & 
        (df['unified_score'] >= min_score)
    ]

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", len(df))
    col2.metric("High Conviction", len(df_filtered))
    
    avg_move = df_filtered['expected_move_pct'].mean() if not df_filtered.empty else 0
    col3.metric("Avg Exp Move", f"{avg_move:.2f}%")
    
    recent_ticker = df.iloc[0]['source_ticker'] if not df.empty else "N/A"
    col4.metric("Latest Event", recent_ticker)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📡 Live Feed", "📊 Analysis & History", "🤖 Agent Logs"])

    with tab1:
        st.subheader("Latest Signals")
        if df_filtered.empty:
            st.warning("No signals match current filters.")
        else:
            for index, row in df_filtered.head(10).iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="signal-card">
                        <h3>{row['source_ticker']} ➔ {row['target_ticker']}</h3>
                        <p><strong>Event:</strong> {row['event_type']} | <strong>Confidence:</strong> {row['confidence']}% | <strong>Score:</strong> {row['unified_score']}</p>
                        <p><em>{row['summary']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"View Analysis Details for {row['target_ticker']}"):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown(f"**Reasoning:**\n{row['reasoning']}")
                            st.markdown(f"**Expected Move:** {row['expected_move_pct']}%")
                        with c2:
                            st.metric("Unified Score", row['unified_score'])
                            st.metric("Confidence", f"{row['confidence']}%")

    with tab2:
        st.subheader("Signal History")
        st.dataframe(
            df_filtered[['timestamp', 'source_ticker', 'target_ticker', 'event_type', 'expected_move_pct', 'confidence', 'unified_score']],
            use_container_width=True
        )
        
        if not df_filtered.empty:
            st.subheader("Correlation Scatter Plot")
            fig = px.scatter(
                df_filtered, 
                x="confidence", 
                y="expected_move_pct", 
                size="unified_score", 
                color="event_type",
                hover_data=["source_ticker", "target_ticker"],
                title="Confidence vs. Expected Move"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Agent Research Logs")
        st.markdown("Raw output from the Agentic Research Layer for recent signals.")
        
        for index, row in df_filtered.head(5).iterrows():
            with st.expander(f"Research for {row['source_ticker']} -> {row['target_ticker']}"):
                if row['agent_data'] and row['agent_data'] != 'None':
                    st.text(row['agent_data'])
                else:
                    st.info("No agent data available for this signal.")

# Auto-refresh logic
time.sleep(refresh_rate)
st.rerun()