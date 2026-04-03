import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import time
import asyncio
from agent_workflow import run_research_task
import portfolio_manager
from grok_analysis import analyze_impact, generate_briefing
import re
from report_generator import generate_signal_report
from dotenv import load_dotenv, set_key
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import io
from polygon import RESTClient
from agent_workflow import identify_agents_for_prompt_improvement
from node_identity import load_identity, get_forum_status
from learning_config_manager import load_config, save_config, get_participation_intensity
from agent_learning import (
    get_learning_summary, calculate_signal_stats, calculate_agent_performance,
    calibrate_confidence, detect_market_regime, generate_prompt_improvement_suggestions,
)
from agent_consensus import get_recent_consensus_signals
from forum_config import is_forum_configured
from llm_config import (
    load_llm_config, save_llm_config, get_available_providers,
    reload_config,
)
from swarm_config import load_swarm_config, save_swarm_config, is_swarm_enabled
from version import VERSION
from trader_profile import (
    load_profile, save_profile, apply_preset, profile_exists,
    get_trading_style, get_cost_warning, PRESETS, COST_WARNINGS,
)

# --- Configuration ---
st.set_page_config(
    page_title=f"FlowTrace V{VERSION}",
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

def generate_execution_graph(agent_log):
    """Generates a Graphviz DOT string from the agent execution log."""
    if not agent_log: return None
    
    dot = "digraph G {\n"
    dot += "  rankdir=TB;\n"
    dot += "  node [shape=box, style=filled, fontname=\"Arial\"];\n"
    
    entries = agent_log.split('\n\n')
    step = 0
    prev_node = "Start"
    has_steps = False
    
    dot += f'  Start [shape=circle, fillcolor="#4CAF50", style=filled, fontcolor=white];\n'
    
    for entry in entries:
        if "CIO Thought" in entry:
            node_id = f"CIO_{step}"
            dot += f'  {node_id} [label="CIO\\n(Reasoning)", fillcolor="#FFC107"];\n'
            dot += f'  {prev_node} -> {node_id};\n'
            prev_node = node_id
            step += 1
            has_steps = True
        elif "🤖 **" in entry:
            match = re.search(r"🤖 \*\*(.*?)\*\*", entry)
            if match:
                agent_name = match.group(1)
                node_id = f"{agent_name}_{step}"
                dot += f'  {node_id} [label="{agent_name}", fillcolor="#2196F3", fontcolor=white];\n'
                dot += f'  {prev_node} -> {node_id};\n'
                prev_node = node_id
                step += 1
                has_steps = True
        elif "Final Report" in entry:
            node_id = "End"
            dot += f'  {node_id} [label="Final\\nReport", shape=circle, fillcolor="#F44336", fontcolor=white];\n'
            dot += f'  {prev_node} -> {node_id};\n'
            prev_node = node_id
            has_steps = True
            
    dot += "}"
    
    if not has_steps:
        return None
        
    return dot

@st.cache_data(ttl=300)
def fetch_watchlist_news(tickers, api_key):
    """Fetches recent news for watchlist tickers."""
    if not tickers or not api_key: return []
    try:
        client = RESTClient(api_key=api_key)
        all_news = []
        for t in tickers:
            try:
                # Fetch last 2 news items per ticker
                resp = client.list_ticker_news(ticker=t, limit=2)
                for n in resp:
                    all_news.append({
                        "ticker": t,
                        "title": n.title,
                        "published_utc": n.published_utc,
                        "article_url": n.article_url,
                        "description": n.description or "No description available."
                    })
            except Exception:
                continue
        
        all_news.sort(key=lambda x: x['published_utc'], reverse=True)
        return all_news
    except Exception:
        return []

def get_ticker_sectors(tickers):
    """Fetches sector info from Knowledge Graph for given tickers."""
    if not os.path.exists(DB_PATH):
        return {t: "Unknown" for t in tickers}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        # Parameterized query for list
        placeholders = ','.join(['?'] * len(tickers))
        query = f"SELECT ticker, sector FROM companies WHERE ticker IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=tickers)
        conn.close()
        
        sector_map = dict(zip(df['ticker'], df['sector']))
        return {t: sector_map.get(t, "Unknown") for t in tickers}
    except Exception:
        return {t: "Unknown" for t in tickers}

# --- Sidebar ---
st.sidebar.title("⚡ FlowTrace")
st.sidebar.caption(f"V{VERSION} | {get_trading_style().replace('_', ' ').title()}")

# Load env for sidebar display
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path, override=True)
mode = os.getenv("TRADING_MODE", "PAPER")
if mode == "LIVE":
    st.sidebar.warning("🔴 LIVE TRADING ACTIVE")
else:
    st.sidebar.success("🟢 PAPER TRADING ACTIVE")

st.sidebar.markdown("---")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
min_confidence = st.sidebar.slider("Min Confidence %", 0, 100, 70)
min_score = st.sidebar.slider("Min Unified Score", 0.0, 1.0, 0.7)

if st.sidebar.button("Refresh Data"):
    st.rerun()

# --- Simulation Mode ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Simulation Mode")
sim_ticker = st.sidebar.text_input("Sim Source Ticker", placeholder="e.g. AAPL")
sim_headline = st.sidebar.text_area("Sim Headline", placeholder="Apple announces massive order increase for...")
sim_partners = st.sidebar.text_input("Sim Partners (comma sep)", placeholder="SWKS, QRVO")

if st.sidebar.button("Run Simulation"):
    if sim_ticker and sim_headline and sim_partners:
        with st.status("Running Simulation...", expanded=True) as status:
            st.write("1. Parsing inputs...")
            partners_list = [{"ticker": p.strip(), "name": "Simulated Partner", "relationship": "Supplier"} for p in sim_partners.split(",")]
            
            # Mock news object
            class MockNews:
                def __init__(self, t, d):
                    self.title = t
                    self.description = d
                    self.published_utc = "2024-01-01T12:00:00Z"
            
            news_item = MockNews(sim_headline, "Simulated description.")
            
            st.write("2. Triggering Grok Analysis...")
            # We pass None for agent_data/market_data for simplicity in sim mode, or could mock them
            result = asyncio.run(analyze_impact(sim_ticker, partners_list, news_item))
            
            status.update(label="Simulation Complete", state="complete")
            
            if result:
                st.success("Analysis Generated!")
                st.json(result)
                
                # Optional: Trigger Strategy Agent on the result
                if 'targets' in result and len(result['targets']) > 0:
                    target = result['targets'][0]
                    st.write(f"3. Requesting Strategy for {target['ticker']}...")
                    strategy_query = f"Given a {target['expected_move_pct']}% move expectation for {target['ticker']} due to {sim_headline}, propose an option strategy."
                    strat_result = asyncio.run(run_research_task(strategy_query))
                    st.info(f"Strategy Agent Proposal:\n{strat_result}")

# --- Gallery Management ---
st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Gallery Management")
if st.sidebar.button("Clear Gallery"):
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    if os.path.exists(screenshot_dir):
        count = 0
        for file in os.listdir(screenshot_dir):
            file_path = os.path.join(screenshot_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    count += 1
            except Exception as e:
                st.sidebar.error(f"Error deleting {file}: {e}")
        st.sidebar.success(f"Cleared {count} screenshots!")
        time.sleep(1)
        st.rerun()
    else:
        st.sidebar.info("Gallery is already empty.")

# --- Report Generation ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Reports")

# We need to load data here to pass to the report generator if the button is clicked
# Since load_data is called in Main Content, we might want to ensure we have the filtered df available.
# However, Streamlit reruns the script, so we can access the data logic.
# To keep it clean, we'll rely on the main data load or reload it for the report.
if st.sidebar.button("Prepare PDF Report"):
    with st.spinner("Generating PDF..."):
        # Logic handled in main flow to access filtered dataframe or reload
        st.session_state['generate_report'] = True

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

    # Handle Report Generation Trigger
    if st.session_state.get('generate_report'):
        pdf_bytes = generate_signal_report(df_filtered)
        st.sidebar.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"flowtrace_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
        st.session_state['generate_report'] = False

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals", len(df))
    col2.metric("High Conviction", len(df_filtered))
    
    avg_move = df_filtered['expected_move_pct'].mean() if not df_filtered.empty else 0
    col3.metric("Avg Exp Move", f"{avg_move:.2f}%")
    
    recent_ticker = df.iloc[0]['source_ticker'] if not df.empty else "N/A"
    col4.metric("Latest Event", recent_ticker)

    # Tabs
    # Reordered for Phase 4 Interface
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([
        "📡 Live Feed", "💬 Analyst Chat", "📊 Analysis & History", "🤖 Agent Logs",
        "💼 Portfolio", "🖼️ Gallery", "⚙️ Settings", "📈 Performance",
        "⚠️ Risk Analysis", "👀 Watchlist", "🌍 Macro Dashboard", "🧠 Learning & AI",
        "🐝 Trading Swarm", "🌊 Cross-Intelligence",
    ])

    with tab1:
        st.subheader("Latest Signals")
        
        # Morning Briefing Section
        if st.button("🎙️ Generate Morning Briefing"):
            with st.spinner("Compiling market briefing..."):
                # Convert filtered df to list of dicts
                signals_list = df_filtered.head(10).to_dict(orient='records')
                briefing = asyncio.run(generate_briefing(signals_list))
                st.success("Briefing Ready")
                st.markdown(f"""
                <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #FFC107;">
                    <h4>🌅 Morning Briefing</h4>
                    {briefing}
                </div>
                """, unsafe_allow_html=True)
                
                # TTS for Briefing
                try:
                    tts = gTTS(text=briefing, lang='en')
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format='audio/mp3')
                except Exception:
                    pass
        
        st.markdown("---")
        
        if df_filtered.empty:
            st.warning("No signals match current filters.")
        else:
            # Signal freshness classification
            try:
                from signal_intelligence import classify_freshness
                freshness_available = True
            except ImportError:
                freshness_available = False

            for index, row in df_filtered.head(10).iterrows():
                # Determine direction and colors
                move = row.get('expected_move_pct', 0) or 0
                direction = "BULLISH" if move > 0 else "BEARISH" if move < 0 else "NEUTRAL"
                dir_color = "#00FF00" if direction == "BULLISH" else "#FF4444" if direction == "BEARISH" else "#AAAAAA"

                # Signal freshness badge
                freshness = ""
                if freshness_available and row.get("timestamp"):
                    try:
                        freshness = classify_freshness(str(row["timestamp"]))
                    except Exception:
                        freshness = ""
                freshness_badge = {"fresh": "🟢", "aging": "🟡", "stale": "⚪"}.get(freshness, "")

                # Sector and regime from expanded columns (may be None for old signals)
                sector = row.get("sector", "") or ""
                regime = row.get("market_regime", "") or ""

                with st.container():
                    st.markdown(f"""
                    <div style="background-color:#1a1a2e;padding:16px;border-radius:10px;margin-bottom:12px;border-left:5px solid {dir_color};">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <h3 style="margin:0;">{freshness_badge} {direction}: {row['source_ticker']} → {row['target_ticker']}</h3>
                            <span style="color:{dir_color};font-size:1.2em;font-weight:bold;">{row.get('confidence', 0):.0f}%</span>
                        </div>
                        <p style="margin:4px 0;color:#888;">
                            Event: {row.get('event_type', 'Unknown')} | Expected: {move:+.1f}% |
                            Score: {row.get('unified_score', 0):.2f}
                            {f' | Sector: {sector}' if sector else ''}
                            {f' | {regime.replace("_", " ").title()}' if regime else ''}
                        </p>
                        <p><em>{row.get('summary', '')}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"View Full Analysis for {row['target_ticker']}"):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown(f"**Reasoning:**\n{row.get('reasoning', 'N/A')}")
                            st.markdown(f"**Expected Move:** {move:+.1f}%")

                            # Risk factors (from expanded schema)
                            risk_str = row.get("risk_factors", "")
                            if risk_str:
                                try:
                                    risks = json.loads(risk_str) if isinstance(risk_str, str) else risk_str
                                    if risks:
                                        st.markdown("**Risk Factors:**")
                                        for r in risks[:5]:
                                            st.markdown(f"- {r}")
                                except Exception:
                                    pass

                        with c2:
                            st.metric("Unified Score", f"{row.get('unified_score', 0):.2f}")
                            st.metric("Confidence", f"{row.get('confidence', 0):.0f}%")
                            if sector:
                                st.metric("Sector", sector)
                        
                        # Text-to-Speech for Summary
                        summary_text = str(row['summary'])
                        if summary_text and summary_text.lower() != "none":
                            if st.button("🔊 Read Summary", key=f"tts_{index}"):
                                with st.spinner("Generating audio..."):
                                    try:
                                        tts = gTTS(text=summary_text, lang='en')
                                        audio_fp = io.BytesIO()
                                        tts.write_to_fp(audio_fp)
                                        audio_fp.seek(0)
                                        st.audio(audio_fp, format='audio/mp3')
                                    except Exception as e:
                                        st.error(f"Error generating audio: {e}")

    with tab2:
        st.subheader("💬 Chat with Analyst Swarm")
        st.info("Ask the swarm to research tickers, check risks, or propose strategies. The CIO Agent will delegate tasks.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("E.g., 'Analyze the risks for NVDA' or 'Find suppliers for Tesla'"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("The Swarm is working..."):
                    # Run the agent workflow
                    full_response = asyncio.run(run_research_task(prompt))
                    
                    # Parse out the Final Report for the main chat bubble
                    final_answer = "Analysis Complete. See logs for details."
                    logs = full_response
                    
                    # Simple parsing logic based on agent_workflow.py logging format
                    if "🏁 **Final Report**:" in full_response:
                        parts = full_response.split("🏁 **Final Report**:")
                        logs = parts[0]
                        final_answer = parts[1].strip()
                    
                    st.markdown(final_answer)
                    
                    with st.expander("View Agent Thought Process & Logs"):
                        st.markdown(logs)
                        # Optional: Show graph
                        graph_dot = generate_execution_graph(logs)
                        if graph_dot:
                            st.graphviz_chart(graph_dot)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    with tab3:
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

    with tab4:
        st.subheader("Agent Research Logs")
        st.markdown("Raw output from the Agentic Research Layer for recent signals.")
        
        for index, row in df_filtered.head(5).iterrows():
            with st.expander(f"Research for {row['source_ticker']} -> {row['target_ticker']}"):
                if row['agent_data'] and row['agent_data'] != 'None':
                    content = row['agent_data']
                    
                    # Check for screenshots
                    screenshot_match = re.search(r"\[SCREENSHOT: (.*?)\]", content)
                    if screenshot_match:
                        image_path = screenshot_match.group(1)
                        if os.path.exists(image_path):
                            st.image(image_path, caption="Agent Screenshot", use_container_width=True)
                    
                    st.markdown(content)
                    
                    # Reasoning Graph
                    graph_dot = generate_execution_graph(content)
                    if graph_dot:
                        st.markdown("#### 🧠 Reasoning Flow")
                        st.code(graph_dot, language='DOT') # Use st.code with DOT language
                else:
                    st.info("No agent data available for this signal.")

    with tab5:
        st.subheader("Portfolio Overview")
        
        # --- Trade Execution ---
        with st.expander("💸 Manual Trade Execution", expanded=False):
            with st.form("trade_form"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    trade_ticker = st.text_input("Ticker", placeholder="AAPL").upper()
                with c2:
                    trade_action = st.selectbox("Action", ["BUY", "SELL"])
                with c3:
                    trade_qty = st.number_input("Quantity", min_value=1, step=1)
                with c4:
                    trade_price = st.number_input("Price ($)", min_value=0.01, step=0.01)
                
                submit_trade = st.form_submit_button("Execute Order")
                
                if submit_trade:
                    if not trade_ticker:
                        st.error("Please enter a ticker.")
                    else:
                        summary = portfolio_manager.get_portfolio_summary()
                        cash = summary['cash']
                        total_val = trade_qty * trade_price
                        
                        if trade_action == "BUY":
                            if cash >= total_val:
                                portfolio_manager.update_cash(-total_val)
                                portfolio_manager.add_position(trade_ticker, trade_qty, trade_price)
                                portfolio_manager.log_trade(trade_ticker, "BUY", trade_qty, trade_price)
                                st.success(f"Executed BUY: {trade_qty} {trade_ticker} @ ${trade_price}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Insufficient funds. Cash: ${cash:,.2f}, Cost: ${total_val:,.2f}")
                        
                        elif trade_action == "SELL":
                            positions = summary['positions']
                            pos = next((p for p in positions if p['ticker'] == trade_ticker), None)
                            owned = pos['quantity'] if pos else 0
                            
                            if owned >= trade_qty:
                                portfolio_manager.update_cash(total_val)
                                portfolio_manager.add_position(trade_ticker, -trade_qty, trade_price)
                                portfolio_manager.log_trade(trade_ticker, "SELL", trade_qty, trade_price)
                                st.success(f"Executed SELL: {trade_qty} {trade_ticker} @ ${trade_price}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Insufficient shares. Owned: {owned}, Selling: {trade_qty}")

        summary = portfolio_manager.get_portfolio_summary()
        
        # Calculate Realized P&L
        history = portfolio_manager.get_trade_history()
        realized_pnl = 0.0
        if history:
            realized_pnl = sum(item['pnl'] for item in history if item.get('pnl') is not None)

        # Metrics
        total_value = summary['cash']
        positions_df = pd.DataFrame(summary['positions'])
        invested_value = 0.0
        
        if not positions_df.empty:
            # Calculate current value (using cost basis for now as we don't have live price feed in this view easily)
            positions_df['value'] = positions_df['quantity'] * positions_df['avg_price']
            invested_value = positions_df['value'].sum()
            total_value += invested_value
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Liquidation Value", f"${total_value:,.2f}")
        c2.metric("Cash Balance", f"${summary['cash']:,.2f}")
        c3.metric("Invested Capital", f"${invested_value:,.2f}")
        c4.metric("Total Realized P&L", f"${realized_pnl:,.2f}")
        
        if not positions_df.empty:
            st.markdown("### Positions")
            st.dataframe(positions_df, use_container_width=True)
            
            # Allocation Chart
            if 'value' in positions_df.columns:
                fig = px.pie(positions_df, values='value', names='ticker', title='Portfolio Allocation')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active positions.")
            
        st.markdown("---")
        st.subheader("Trade Journal & History")
        if history:
            df_history = pd.DataFrame(history)
            
            # Ensure notes column exists and handle NaNs
            if 'notes' not in df_history.columns:
                df_history['notes'] = ""
            else:
                df_history['notes'] = df_history['notes'].fillna("")

            edited_df = st.data_editor(
                df_history,
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "timestamp": st.column_config.DatetimeColumn("Time", disabled=True, format="D MMM YYYY, h:mm a"),
                    "ticker": st.column_config.TextColumn("Ticker", disabled=True, width="small"),
                    "action": st.column_config.TextColumn("Action", disabled=True, width="small"),
                    "quantity": st.column_config.NumberColumn("Qty", disabled=True, width="small"),
                    "price": st.column_config.NumberColumn("Price", disabled=True, format="$%.2f", width="small"),
                    "pnl": st.column_config.NumberColumn("Realized P&L", disabled=True, format="$%.2f", width="small"),
                    "notes": st.column_config.TextColumn("Journal Notes", width="large")
                },
                column_order=["timestamp", "ticker", "action", "quantity", "price", "pnl", "notes"],
                use_container_width=True,
                hide_index=True,
                key="journal_editor"
            )

            if st.session_state.get("journal_editor"):
                edits = st.session_state["journal_editor"].get("edited_rows", {})
                for idx, change in edits.items():
                    if "notes" in change:
                        trade_id = df_history.iloc[idx]['id']
                        portfolio_manager.update_trade_note(trade_id, change['notes'])
        else:
            st.info("No trades executed yet.")

    with tab6:
        st.subheader("Agent Screenshot Gallery")
        screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        
        if os.path.exists(screenshot_dir):
            # Get list of images sorted by modification time (newest first)
            images = [f for f in os.listdir(screenshot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            images.sort(key=lambda x: os.path.getmtime(os.path.join(screenshot_dir, x)), reverse=True)
            
            if images:
                # Create a masonry-like grid using columns
                cols = st.columns(3)
                for idx, img_file in enumerate(images):
                    file_path = os.path.join(screenshot_dir, img_file)
                    with cols[idx % 3]:
                        st.image(file_path, caption=img_file, use_container_width=True)
            else:
                st.info("No screenshots found in the directory.")
        else:
            st.info("No screenshots directory found yet (Agents haven't run or taken shots).")

    with tab7:
        st.subheader("Configuration")

        # --- Trader Profile Section ---
        st.markdown("### Trader Profile")
        if not profile_exists():
            st.warning("No trader profile configured. Select your trading style to optimize the system for your needs.")

        current_profile = load_profile()
        current_style = current_profile.get("trading_style", "swing_trader")
        current_horizon = current_profile.get("swing_horizon", "multi_week")

        # Map to preset key
        style_options = {
            "Value Investor": "value_investor",
            "Swing Trader (Multi-Week)": "swing_multi_week",
            "Swing Trader (Single Week)": "swing_single_week",
            "Day Trader": "day_trader",
        }
        reverse_map = {v: k for k, v in style_options.items()}

        # Determine current display
        if current_style == "value_investor":
            current_key = "value_investor"
        elif current_style == "day_trader":
            current_key = "day_trader"
        elif current_horizon == "single_week":
            current_key = "swing_single_week"
        else:
            current_key = "swing_multi_week"

        current_display = reverse_map.get(current_key, "Swing Trader (Multi-Week)")

        selected_style = st.radio(
            "Trading Style",
            options=list(style_options.keys()),
            index=list(style_options.keys()).index(current_display),
            horizontal=True,
        )
        selected_key = style_options[selected_style]

        # Show cost warning
        cost = COST_WARNINGS.get(selected_key, {})
        if cost.get("level") == "high":
            st.error(cost.get("message", ""))
        elif cost.get("level") in ("medium", "low_medium"):
            st.warning(cost.get("message", ""))
        else:
            st.success(cost.get("message", ""))

        # Show preset details
        preset = PRESETS.get(selected_key, {})
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Fundamental Weight", f"{preset.get('fundamental_weight', 0.4):.0%}")
        pc2.metric("Technical Weight", f"{preset.get('technical_weight', 0.6):.0%}")
        pc3.metric("Swarm LLM Rec", str(preset.get("swarm_llm_recommendation", "auto")))

        if selected_key != current_key:
            if st.button("Apply Trading Profile"):
                apply_preset(selected_key)
                st.success(f"Profile set to **{selected_style}**. System will adapt on next cycle.")
                st.rerun()

        st.markdown("---")

        st.info("Note: Updates to API keys require a restart of the application/listener to take effect.")
        
        # Load current env vars to pre-fill
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(env_path, override=True)
        
        with st.form("settings_form"):
            st.markdown("### LLM API Keys")
            xai_key = st.text_input("xAI (Grok) API Key", value=os.getenv("XAI_API_KEY", ""), type="password")
            anthropic_key = st.text_input("Anthropic (Claude) API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password")
            openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
            google_key = st.text_input("Google (Gemini) API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")

            st.markdown("### Data API Keys")
            poly_key = st.text_input("Polygon.io API Key", value=os.getenv("POLYGON_API_KEY", ""), type="password")
            finn_key = st.text_input("Finnhub API Key", value=os.getenv("FINNHUB_API_KEY", ""), type="password")
            serp_key = st.text_input("Serper API Key", value=os.getenv("SERPER_API_KEY", ""), type="password")
            sec_email = st.text_input("SEC Email (User Agent)", value=os.getenv("SEC_EMAIL", ""))
            
            st.markdown("### System Settings")
            current_mode = os.getenv("TRADING_MODE", "PAPER")
            mode_index = 1 if current_mode == "LIVE" else 0
            trading_mode = st.radio("Trading Mode", ["Paper Trading", "Live Trading"], index=mode_index)
            
            submitted = st.form_submit_button("Save Settings")
            
            if submitted:
                # Create .env if it doesn't exist
                if not os.path.exists(env_path):
                    with open(env_path, 'w') as f: pass
                
                set_key(env_path, "XAI_API_KEY", xai_key)
                set_key(env_path, "ANTHROPIC_API_KEY", anthropic_key)
                set_key(env_path, "OPENAI_API_KEY", openai_key)
                set_key(env_path, "GOOGLE_API_KEY", google_key)
                set_key(env_path, "POLYGON_API_KEY", poly_key)
                set_key(env_path, "FINNHUB_API_KEY", finn_key)
                set_key(env_path, "SERPER_API_KEY", serp_key)
                set_key(env_path, "SEC_EMAIL", sec_email)
                
                new_mode = "LIVE" if "Live" in trading_mode else "PAPER"
                set_key(env_path, "TRADING_MODE", new_mode)
                
                st.success("Settings saved to .env!")
                time.sleep(0.5)
                st.rerun()

            st.markdown("---")
            st.markdown("### Prompt Optimization")
            if st.button("Suggest Prompt Improvements"):
                with st.spinner("Analyzing agent performance..."):
                    # Call the function to get prompt improvement suggestions
                    suggested_improvements = asyncio.run(identify_agents_for_prompt_improvement())

                    if suggested_improvements:
                        st.write("Suggested Prompt Improvements:")
                        st.json(suggested_improvements)
                    else:
                        st.info("No improvements suggested (or no performance data yet).")
        
        st.markdown("---")
        st.subheader("LLM Provider Configuration")
        st.caption("Choose which AI model powers each agent. Leave blank to use the default.")

        llm_cfg = load_llm_config()
        providers = get_available_providers()

        # Show which providers have keys
        prov_cols = st.columns(4)
        for i, (name, info) in enumerate(providers.items()):
            col = prov_cols[i % 4]
            status = "Configured" if info["configured"] else "No API Key"
            color = "#00ff88" if info["configured"] else "#ff4444"
            col.markdown(f"**{name.upper()}**: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

        configured_providers = [name for name, info in providers.items() if info["configured"]]

        if configured_providers:
            # Default provider
            current_default = llm_cfg.get("default_provider", "xai")
            default_idx = configured_providers.index(current_default) if current_default in configured_providers else 0
            new_default = st.selectbox("Default LLM Provider", configured_providers, index=default_idx)

            # Default model
            default_model = llm_cfg.get("default_model", "")
            provider_cfg = llm_cfg.get("providers", {}).get(new_default, {})
            new_model = st.text_input("Default Model", value=provider_cfg.get("default_model", default_model))

            # Per-agent assignments
            st.markdown("#### Per-Agent Overrides")
            st.caption("Assign specific agents to different providers. Unset = uses default.")

            agent_types = [
                "Supervisor", "ResearchAgent", "MacroAgent", "SentimentAgent",
                "FundamentalAgent", "TechnicalAgent", "ScoutAgent", "StrategyAgent",
                "AnalysisEngine", "DebateAgent", "KnowledgeGraph", "VisionModel",
            ]

            assignments = llm_cfg.get("agent_assignments", {})
            new_assignments = {}

            for agent_type in agent_types:
                current = assignments.get(agent_type)
                current_prov = current.get("provider", "") if isinstance(current, dict) else ""
                options = ["(default)"] + configured_providers
                idx = options.index(current_prov) if current_prov in options else 0
                selected = st.selectbox(f"{agent_type}", options, index=idx, key=f"llm_{agent_type}")
                if selected != "(default)":
                    agent_model = llm_cfg.get("providers", {}).get(selected, {}).get("default_model", "")
                    new_assignments[agent_type] = {"provider": selected, "model": agent_model}

            if st.button("Save LLM Configuration"):
                llm_cfg["default_provider"] = new_default
                llm_cfg["default_model"] = new_model
                llm_cfg["agent_assignments"] = new_assignments
                save_llm_config(llm_cfg)
                reload_config()
                st.success("LLM configuration saved! Changes take effect on next agent invocation.")
        else:
            st.warning("No LLM providers configured. Add at least one API key above and save settings.")

        st.markdown("---")
        st.subheader("Danger Zone")
        if st.button("Reset Portfolio (Clear All Data)"):
            portfolio_manager.reset_portfolio()
            st.success("Portfolio reset to initial state ($100,000 cash, no positions).")
            time.sleep(1)
            st.rerun()

    with tab8:
        st.subheader("Portfolio Performance (Realized)")
        equity_data = portfolio_manager.get_equity_curve()
        
        if equity_data:
            df_equity = pd.DataFrame(equity_data)
            df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'])
            
            # Equity Curve
            # fig_eq = px.line(df_equity, x='timestamp', y='equity', title='Total Portfolio Value (Realized)', markers=True)
            
            # Create dual-axis chart for Equity + Sentiment
            fig = go.Figure()
            
            # Trace 1: Equity
            fig.add_trace(go.Scatter(x=df_equity['timestamp'], y=df_equity['equity'], name="Portfolio Value", line=dict(color='#00FF00', width=2)))
            
            # Trace 2: Sentiment Overlay
            sentiment_df = portfolio_manager.get_daily_sentiment()
            if sentiment_df is not None and not sentiment_df.empty:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                fig.add_trace(go.Bar(x=sentiment_df['date'], y=sentiment_df['avg_score'], name="Avg Sentiment Score", yaxis="y2", opacity=0.3, marker_color='cyan'))
            
            fig.update_layout(
                title="Portfolio Value vs. News Sentiment",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                yaxis2=dict(title="Sentiment Score (0-1)", overlaying="y", side="right", range=[0, 1]),
                legend=dict(x=0, y=1.1, orientation="h")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cash vs Invested
            df_melted = df_equity.melt(id_vars=['timestamp'], value_vars=['cash', 'invested'], var_name='Type', value_name='Amount')
            fig_alloc = px.area(df_melted, x='timestamp', y='Amount', color='Type', title='Cash vs Invested Capital')
            st.plotly_chart(fig_alloc, use_container_width=True)
            
            # Stats
            start_val = 100000.0
            curr_val = df_equity.iloc[-1]['equity']
            pnl = curr_val - start_val
            pnl_pct = (pnl / start_val) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Net Profit/Loss", f"${pnl:,.2f}", f"{pnl_pct:.2f}%")
            c2.metric("Trade Count", len(df_equity))
            c3.metric("Current Equity", f"${curr_val:,.2f}")
        else:
            st.info("No trade history available to generate performance charts.")

    with tab9:
        st.subheader("Portfolio Risk Analysis")
        risk_metrics = portfolio_manager.get_risk_metrics()
        
        if risk_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
            c2.metric("VaR (95%)", f"{risk_metrics['var_95_pct']:.2%}")
            c3.metric("Annual Volatility", f"{risk_metrics['volatility_annual']:.2%}")
            c4.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
            
            st.info(f"Value at Risk (95%): There is a 5% chance the portfolio loses more than {abs(risk_metrics['var_95_pct']):.2%} in a single day.")
            
            # Distribution Chart
            curve = portfolio_manager.get_equity_curve()
            df = pd.DataFrame(curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            daily_returns = df['equity'].resample('D').last().ffill().pct_change().dropna()
            
            fig_hist = px.histogram(daily_returns, x="equity", nbins=50, title="Distribution of Daily Returns", labels={'equity': 'Daily Return'})
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Not enough data to calculate risk metrics (need at least 2 days of history).")

        # Holdings Correlation Matrix
        st.markdown("---")
        st.subheader("Holdings Correlation Matrix")
        
        with st.spinner("Calculating correlations (fetching historical data)..."):
            corr_matrix = portfolio_manager.get_holdings_correlation()
            
            if corr_matrix is not None and not corr_matrix.empty:
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    aspect="auto", 
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Correlation Heatmap (90 Days)"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough holdings or data to calculate correlation matrix.")

    with tab10:
        st.subheader("Market Watchlist")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            new_ticker = st.text_input("Add Ticker to Watchlist", placeholder="e.g. AMD")
        with c2:
            if st.button("Add Ticker"):
                if new_ticker:
                    portfolio_manager.add_to_watchlist(new_ticker)
                    st.success(f"Added {new_ticker}")
                    time.sleep(0.5)
                    st.rerun()

        with st.spinner("Fetching watchlist data..."):
            wl_data = portfolio_manager.get_watchlist_with_prices()
        
        if wl_data:
            df_wl = pd.DataFrame(wl_data)
            # Format price
            df_wl['Price'] = df_wl['Price'].apply(lambda x: f"${x:,.2f}")
            # Format Change
            df_wl['Change'] = df_wl['ChangePct'].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(df_wl, use_container_width=True)
            
            st.markdown("---")
            tickers_list = [w['Ticker'] for w in wl_data]
            to_remove = st.selectbox("Select Ticker to Remove", options=tickers_list)
            if st.button("Remove Ticker"):
                portfolio_manager.remove_from_watchlist(to_remove)
                st.success(f"Removed {to_remove}")
                time.sleep(0.5)
                st.rerun()
            
            st.markdown("---")
            st.subheader("🔔 Price Alerts")
            
            # Alert Setting Form
            with st.form("alert_form"):
                c_t, c_p, c_c = st.columns(3)
                with c_t:
                    alert_ticker = st.selectbox("Ticker", options=tickers_list, key="alert_ticker")
                with c_p:
                    alert_price = st.number_input("Target Price", min_value=0.0, step=0.01)
                with c_c:
                    alert_cond = st.selectbox("Condition", ["ABOVE", "BELOW"])
                
                if st.form_submit_button("Set Alert"):
                    portfolio_manager.set_price_alert(alert_ticker, alert_price, alert_cond)
                    st.success(f"Alert set for {alert_ticker} {alert_cond} ${alert_price}")
                    st.rerun()

            # Check Alerts
            if wl_data:
                # Convert list of dicts to dict {ticker: price} for checking
                current_prices_map = {item['Ticker']: float(str(item['Price']).replace('$','').replace(',','')) for item in wl_data}
                triggered_alerts = portfolio_manager.check_alerts(current_prices_map)
                
                if triggered_alerts:
                    st.error("🚨 ALERTS TRIGGERED:")
                    for alert in triggered_alerts:
                        st.write(f"**{alert['ticker']}** is {alert['price']} ({alert['condition']} target ${alert['target']})")
            
            # News Feed
            st.markdown("---")
            st.subheader("📰 Watchlist News Feed")
            
            poly_key = os.getenv("POLYGON_API_KEY")
            if poly_key and tickers_list:
                news_items = fetch_watchlist_news(tickers_list, poly_key)
                if news_items:
                    for item in news_items[:10]: # Show top 10 recent
                        with st.expander(f"{item['published_utc'][:10]} | {item['ticker']} | {item['title']}"):
                            st.write(item['description'])
                            st.markdown(f"[Read Full Article]({item['article_url']})")
                else:
                    st.info("No recent news found for watchlist tickers.")
            
            # Sector Heatmap
            st.markdown("---")
            st.subheader("Sector Performance Heatmap")
            
            tickers = [item['Ticker'] for item in wl_data]
            sectors = get_ticker_sectors(tickers)
            
            # Prepare data for Treemap
            heatmap_data = []
            for item in wl_data:
                t = item['Ticker']
                heatmap_data.append({
                    "Ticker": t,
                    "Sector": sectors.get(t, "Other"),
                    "ChangePct": item.get('ChangePct', 0.0),
                    "Price": item.get('Price', 0.0),
                    "Size": 1 # Equal size boxes for readability
                })
                
            if heatmap_data:
                fig_tree = px.treemap(
                    heatmap_data, 
                    path=[px.Constant("Watchlist"), 'Sector', 'Ticker'], 
                    values='Size',
                    color='ChangePct',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    hover_data=['Price', 'ChangePct'],
                    title="Watchlist Performance by Sector"
                )
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("Watchlist is empty. Add tickers to monitor them.")

        # Scenario Analysis
        st.markdown("---")
        st.subheader("Scenario Analysis (Stress Test)")
        
        scenario_type = st.radio("Scenario Type", ["Single Stock Shock", "Macro Factor Shock"], horizontal=True)
        
        portfolio_summary = portfolio_manager.get_portfolio_summary()
        holdings = [p['ticker'] for p in portfolio_summary.get('positions', [])]
        
        if scenario_type == "Single Stock Shock":
            if holdings:
                c1, c2 = st.columns(2)
                with c1:
                    selected_ticker = st.selectbox("Select Ticker to Stress", options=holdings)
                with c2:
                    pct_change = st.slider("Simulated Price Change (%)", -50, 50, -20)
                
                if st.button("Run Scenario"):
                    with st.spinner(f"Simulating a {pct_change}% move in {selected_ticker}..."):
                        stress_result = portfolio_manager.run_stress_test(selected_ticker, pct_change / 100.0)
                        
                        if stress_result:
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Initial Portfolio Value", f"${stress_result['initial_equity']:,.2f}")
                            sc2.metric("Stressed Portfolio Value", f"${stress_result['stressed_equity']:,.2f}")
                            sc3.metric(
                                "P&L Impact", 
                                f"${stress_result['pnl_impact']:,.2f}",
                                delta=f"{stress_result['pnl_impact_pct']:.2%}"
                            )
                        else:
                            st.error("Could not run scenario analysis. Check logs for yfinance errors.")
            else:
                st.info("No holdings in portfolio to run a scenario analysis.")
        
        else: # Macro Factor Shock
            if holdings:
                macro_options = {
                    "S&P 500 (Market)": "^GSPC",
                    "10-Year Treasury Yield (Rates)": "^TNX",
                    "Crude Oil": "CL=F",
                    "Gold": "GC=F",
                    "VIX (Volatility)": "^VIX"
                }
                
                c1, c2 = st.columns(2)
                with c1:
                    selected_macro_name = st.selectbox("Select Macro Factor", options=list(macro_options.keys()))
                    selected_macro_ticker = macro_options[selected_macro_name]
                with c2:
                    macro_pct_change = st.slider("Simulated Factor Change (%)", -30, 30, 5, help="E.g., +10% change in VIX or Oil price.")
                    
                if st.button("Run Macro Stress Test"):
                    with st.spinner(f"Calculating portfolio sensitivity to {selected_macro_name}..."):
                        macro_result = portfolio_manager.run_macro_stress_test(selected_macro_ticker, macro_pct_change / 100.0)
                        
                        if macro_result:
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Initial Portfolio Value", f"${macro_result['initial_equity']:,.2f}")
                            sc2.metric("Stressed Portfolio Value", f"${macro_result['stressed_equity']:,.2f}")
                            sc3.metric(
                                "Est. P&L Impact", 
                                f"${macro_result['pnl_impact']:,.2f}",
                                delta=f"{macro_result['pnl_impact_pct']:.2%}"
                            )
                            
                            with st.expander("View Beta & Impact Details"):
                                details = macro_result['details']
                                det_data = []
                                for t, d in details.items():
                                    det_data.append({
                                        "Ticker": t,
                                        "Beta": f"{d['beta']:.2f}",
                                        "Exp Move": f"{d['expected_move']:.2%}",
                                        "P&L": f"${d['pnl']:,.2f}"
                                    })
                                st.dataframe(pd.DataFrame(det_data), use_container_width=True)
                        else:
                            st.error("Analysis failed. Insufficient historical data or API error.")
            else:
                st.info("No holdings in portfolio to run a scenario analysis.")

    with tab11:
        st.subheader("Global Macro Dashboard")
        st.markdown("Key economic indicators fetched from the Federal Reserve Economic Data (FRED).")
        
        with st.spinner("Fetching macro data..."):
            macro_df = portfolio_manager.get_macro_history()
            
        if not macro_df.empty:
            # Latest Values
            latest = macro_df.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fed Funds Rate", f"{latest.get('Fed Funds Rate', 0):.2f}%")
            m2.metric("10Y Treasury", f"{latest.get('10Y Treasury Yield', 0):.2f}%")
            m3.metric("CPI (Index)", f"{latest.get('CPI (Inflation)', 0):.1f}")
            m4.metric("Unemployment", f"{latest.get('Unemployment Rate', 0):.1f}%")
            
            st.markdown("### US Treasury Yield Curve")
            yield_curve_data = []
            for maturity, col in [("3M", "3M Treasury Yield"), ("2Y", "2Y Treasury Yield"), ("5Y", "5Y Treasury Yield"), ("10Y", "10Y Treasury Yield"), ("30Y", "30Y Treasury Yield")]:
                val = latest.get(col)
                if pd.notnull(val):
                    yield_curve_data.append({"Maturity": maturity, "Yield": val})
            
            if yield_curve_data:
                df_yc = pd.DataFrame(yield_curve_data)
                fig_yc = px.line(df_yc, x="Maturity", y="Yield", markers=True, title=f"Current Yield Curve ({latest.name.strftime('%Y-%m-%d')})")
                st.plotly_chart(fig_yc, use_container_width=True)

            st.markdown("### Interest Rates & Yields History")
            cols_to_plot = [c for c in ['Fed Funds Rate', '3M Treasury Yield', '2Y Treasury Yield', '10Y Treasury Yield'] if c in macro_df.columns]
            if cols_to_plot:
                st.line_chart(macro_df[cols_to_plot])
            
            st.markdown("### Inflation & Employment")
            cols_macro = [c for c in ['CPI (Inflation)', 'Unemployment Rate'] if c in macro_df.columns]
            if cols_macro:
                fig_macro = px.line(macro_df, y=cols_macro, title="Inflation vs Unemployment")
                st.plotly_chart(fig_macro, use_container_width=True)
            
            # CME FedWatch Section
            st.markdown("---")
            st.subheader("🏦 CME FedWatch Tool")
            st.markdown("Market expectations for the Federal Reserve's next interest rate decision.")
            
            with st.spinner("Fetching FedWatch probabilities..."):
                fed_data = portfolio_manager.get_fed_watch_data()
            
            if fed_data:
                try:
                    # The API returns a list of meetings. We take the first one (Next Meeting).
                    next_meeting = fed_data[0]
                    meeting_date = next_meeting.get('meeting', 'Unknown Date')
                    probs = next_meeting.get('prob', [])
                    
                    st.write(f"**Next FOMC Meeting:** {meeting_date}")
                    
                    if probs:
                        df_probs = pd.DataFrame(probs)
                        df_probs['value'] = pd.to_numeric(df_probs['value'], errors='coerce')
                        df_probs.dropna(inplace=True)
                        df_probs = df_probs[df_probs['value'] > 0] # Filter out 0% probs
                        
                        fig_fed = px.bar(
                            df_probs, 
                            x='label', 
                            y='value', 
                            title=f"Target Rate Probabilities for {meeting_date}",
                            labels={'label': 'Target Rate (bps)', 'value': 'Probability (%)'},
                            text='value'
                        )
                        fig_fed.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_fed.update_layout(yaxis_range=[0, 110])
                        st.plotly_chart(fig_fed, use_container_width=True)
                    else:
                        st.info("No probability data found for the next meeting.")
                except Exception as e:
                    st.error(f"Error parsing FedWatch data: {e}")
            else:
                st.warning("Could not fetch data from CME FedWatch.")
            
            # Commodities Section
            st.markdown("---")
            st.subheader("🛢️ Commodities")
            
            with st.spinner("Fetching commodity prices..."):
                comm_df = portfolio_manager.get_commodities_history()
            
            if not comm_df.empty:
                latest = comm_df.iloc[-1]
                prev = comm_df.iloc[-2] if len(comm_df) > 1 else latest
                
                c1, c2, c3, c4 = st.columns(4)
                metrics = [
                    ("Crude Oil", c1),
                    ("Gold", c2),
                    ("Copper", c3),
                    ("Natural Gas", c4)
                ]
                
                for label, col in metrics:
                    if label in latest:
                        val = latest[label]
                        delta = ((val - prev[label]) / prev[label]) * 100
                        col.metric(label, f"${val:,.2f}", f"{delta:+.2f}%")
                
                st.markdown("### Commodities Trend (Rebased to 100)")
                st.line_chart(comm_df / comm_df.iloc[0] * 100)
        else:
            st.error("Failed to fetch macro data. Check internet connection or FRED availability.")

    with tab12:
        st.subheader("Learning & AI Network")

        # --- Node Identity ---
        identity = load_identity()
        learning_config = load_config()

        st.markdown("### Node Identity")
        id_c1, id_c2, id_c3, id_c4 = st.columns(4)
        id_c1.metric("Node Alias", identity.get("node_alias", "unknown"))
        id_c2.metric("Forum Status", get_forum_status().upper())
        id_c3.metric("Forum Connected", "Yes" if is_forum_configured() else "No (TBD)")
        id_c4.metric("Intensity", get_participation_intensity().upper())

        # Participation Intensity Setting
        st.markdown("### Participation Intensity")
        st.info(
            "Controls how actively your AI agents engage with the AgentForum network. "
            "Higher intensity = more debate, faster learning, better trade identification, "
            "but more LLM API calls and higher costs."
        )
        new_intensity = st.select_slider(
            "Set Intensity",
            options=["low", "medium", "high"],
            value=learning_config.get("participation_intensity", "medium"),
        )
        if new_intensity != learning_config.get("participation_intensity"):
            learning_config["participation_intensity"] = new_intensity
            save_config(learning_config)
            st.success(f"Participation intensity set to **{new_intensity.upper()}**")

        st.markdown("---")

        # --- Performance Overview ---
        st.markdown("### Learning Performance")
        perf = learning_config.get("performance_history", {})
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Total Signals", perf.get("total_signals", 0))
        p2.metric("Wins", perf.get("wins", 0))
        p3.metric("Losses", perf.get("losses", 0))
        p4.metric("Win Rate", f"{perf.get('win_rate', 0):.1%}")
        p5.metric("Avg Return", f"{perf.get('avg_return', 0):+.2f}%")

        last_review = perf.get("last_review")
        if last_review:
            st.caption(f"Last learning review: {last_review}")
        else:
            st.caption("No learning review has run yet. The system will review automatically.")

        st.markdown("---")

        # --- Agent Trust Weights ---
        st.markdown("### Agent Trust Weights")
        st.caption("Weights > 1.0 mean the system trusts this agent more. < 1.0 means less.")

        agent_weights = learning_config.get("agent_weights", {})
        if agent_weights:
            weights_df = pd.DataFrame(
                [(k, v) for k, v in sorted(agent_weights.items(), key=lambda x: x[1], reverse=True)],
                columns=["Agent", "Weight"],
            )
            fig_weights = px.bar(
                weights_df, x="Agent", y="Weight",
                title="Agent Trust Weights (adjusted by LearningAgent)",
                color="Weight",
                color_continuous_scale=["#FF4444", "#FFAA00", "#44FF44"],
                range_color=[0.5, 1.5],
            )
            fig_weights.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.5)
            fig_weights.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_weights, use_container_width=True)

        st.markdown("---")

        # --- Agent Execution Stats ---
        st.markdown("### Agent Execution Stats")
        agent_stats = calculate_agent_performance()
        if agent_stats:
            stats_data = []
            for name, s in agent_stats.items():
                stats_data.append({
                    "Agent": name,
                    "Runs": s["count"],
                    "Success": s["success"],
                    "Fail": s["fail"],
                    "Fail Rate": f"{s['fail_rate']:.1%}",
                    "Avg Duration (s)": s["avg_duration"],
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        else:
            st.info("No agent execution data yet. Run some research tasks to populate.")

        st.markdown("---")

        # --- Consensus Signals ---
        st.markdown("### Recent Consensus Signals")
        consensus_signals = get_recent_consensus_signals(limit=10)
        if consensus_signals:
            for sig in consensus_signals:
                direction_color = "#00FF00" if sig["direction"] == "BULLISH" else "#FF4444"
                st.markdown(
                    f'<div style="background-color:#262730;padding:15px;border-radius:10px;'
                    f'margin-bottom:10px;border-left:5px solid {direction_color};">'
                    f'<strong>{sig["ticker"]}</strong> — {sig["direction"]} '
                    f'<span style="color:{direction_color}">Score: {sig["consensus_score"]:.0%}</span><br/>'
                    f'Expected Move: {sig.get("expected_move_pct", 0):+.1f}% | '
                    f'Horizon: {sig.get("time_horizon_days", 5)}d | '
                    f'Raw Conf: {sig.get("raw_confidence", 0):.0f}% -> '
                    f'Adjusted: {sig.get("adjusted_confidence", 0):.0f}%<br/>'
                    f'<small>{sig.get("timestamp", "")}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No consensus signals yet. The system will generate them as data flows in.")

        st.markdown("---")

        # --- Event Type Weights ---
        st.markdown("### Event Type Performance Weights")
        event_weights = learning_config.get("event_type_weights", {})
        if event_weights:
            ew_df = pd.DataFrame(
                [(k, v) for k, v in event_weights.items()],
                columns=["Event Type", "Weight"],
            )
            st.dataframe(ew_df, use_container_width=True)

        st.markdown("---")

        # --- Market Regime (Phase 4) ---
        st.markdown("### Market Regime Detection")
        regime = detect_market_regime()
        if regime:
            r1, r2 = st.columns(2)
            r1.metric("Current Regime", regime.get("current_regime", "unknown").replace("_", " ").title())
            r2.metric("VIX Level", f"{regime.get('vix_level', 0):.1f}")
            st.caption("Regime affects confidence multipliers: Low VIX = +5%, Normal = neutral, High VIX = -15%")
        else:
            st.info("Market regime data unavailable.")

        st.markdown("---")

        # --- Confidence Calibration (Phase 4) ---
        st.markdown("### Confidence Calibration")
        cal = calibrate_confidence()
        if cal:
            cal_data = []
            for bucket, data in cal.items():
                cal_data.append({
                    "Confidence Bucket": bucket + "%",
                    "Predicted Win Rate": f"{data['predicted_confidence']}%",
                    "Actual Win Rate": f"{data['actual_win_rate']}%",
                    "Calibration Error": f"{data['calibration_error']:+.1f}%",
                    "Samples": data["sample_size"],
                })
            st.dataframe(pd.DataFrame(cal_data), use_container_width=True)
            st.caption("Positive error = overconfident, Negative = underconfident")
        else:
            st.info("Not enough resolved signals for calibration (need 20+).")

        st.markdown("---")

        # --- Network Trust (Phase 4) ---
        st.markdown("### Network Node Trust")
        network_trust = learning_config.get("network_node_trust", {})
        if network_trust:
            trust_df = pd.DataFrame(
                [(k, v) for k, v in sorted(network_trust.items(), key=lambda x: x[1], reverse=True)],
                columns=["Node Alias", "Trust Score"],
            )
            st.dataframe(trust_df, use_container_width=True)
        else:
            st.info("No network trust data yet. Connect to AgentForum to build trust scores.")

        st.markdown("---")

        # --- Prompt Improvement Suggestions (Phase 4) ---
        st.markdown("### Agent Improvement Suggestions")
        suggestions = generate_prompt_improvement_suggestions()
        if suggestions:
            for agent, suggestion in suggestions.items():
                st.warning(f"**{agent}**: {suggestion}")
        else:
            st.success("All agents performing within acceptable parameters.")

    with tab13:
        st.subheader("Trading Agent Swarm")

        swarm_cfg = load_swarm_config()

        # --- Swarm Status & Configuration ---
        st.markdown("### Swarm Configuration")

        sw_c1, sw_c2, sw_c3, sw_c4 = st.columns(4)
        sw_c1.metric("Status", "ENABLED" if swarm_cfg.get("enabled") else "DISABLED")
        sw_c2.metric("Swarm Size", swarm_cfg.get("swarm_size", 20))
        sw_c3.metric("Speed", swarm_cfg.get("simulation_speed", "normal").upper())
        sw_c4.metric("Round Interval", f"{swarm_cfg.get('round_interval_seconds', 30)}s")

        # Enable/disable toggle
        swarm_enabled = st.toggle("Enable Trading Swarm", value=swarm_cfg.get("enabled", False))
        if swarm_enabled != swarm_cfg.get("enabled"):
            swarm_cfg["enabled"] = swarm_enabled
            save_swarm_config(swarm_cfg)
            st.success(f"Swarm {'ENABLED' if swarm_enabled else 'DISABLED'}. Restart monitor to take effect.")

        # Swarm size slider
        new_size = st.slider(
            "Swarm Size (number of trading agents)",
            min_value=5, max_value=10000, value=swarm_cfg.get("swarm_size", 20),
            step=5,
            help="5-20: Full LLM per agent. 21-100: LLM for leaders. 100+: LLM per archetype representative.",
        )
        if new_size != swarm_cfg.get("swarm_size"):
            swarm_cfg["swarm_size"] = new_size
            save_swarm_config(swarm_cfg)
            st.info(f"Swarm size set to {new_size}. Will take effect on next persona generation.")

        # LLM calls per round
        st.markdown("#### LLM Budget per Round")
        st.caption(
            "Controls how many agents get full LLM reasoning per simulation round. "
            "More LLM calls = better swarm intelligence, but higher API costs. "
            '"Auto" uses efficient defaults. "All" gives every agent an LLM call.'
        )
        current_llm = swarm_cfg.get("llm_calls_per_round", "auto")
        llm_options = ["auto", "all", "10", "25", "50", "100", "250", "500", "1000"]
        current_display = str(current_llm) if str(current_llm) in llm_options else "auto"
        new_llm = st.selectbox(
            "LLM calls per round",
            options=llm_options,
            index=llm_options.index(current_display) if current_display in llm_options else 0,
            help="auto = tiered by swarm size | all = every agent | number = exact count",
        )
        if new_llm != str(current_llm):
            swarm_cfg["llm_calls_per_round"] = new_llm if new_llm in ("auto", "all") else int(new_llm)
            save_swarm_config(swarm_cfg)
            st.success(f"LLM budget set to **{new_llm}** calls per round.")

        st.markdown("---")

        # --- Archetype Distribution ---
        st.markdown("### Archetype Distribution")
        archetypes = swarm_cfg.get("archetypes", {})
        if archetypes:
            arch_data = []
            for name, cfg in archetypes.items():
                count = max(1, round(cfg.get("weight", 0.1) * swarm_cfg.get("swarm_size", 20)))
                arch_data.append({
                    "Archetype": name.replace("_", " ").title(),
                    "Weight": f"{cfg.get('weight', 0):.0%}",
                    "Est. Agents": count,
                    "Bias": cfg.get("bias", ""),
                    "Indicators": ", ".join(cfg.get("indicators", [])[:3]),
                })
            st.dataframe(pd.DataFrame(arch_data), use_container_width=True)

        st.markdown("---")

        # --- Swarm Performance (from DB) ---
        st.markdown("### Swarm Performance")

        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row

            # Overall stats
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_agents,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_agents,
                    SUM(lifetime_trades) as total_trades,
                    SUM(lifetime_wins) as total_wins,
                    SUM(lifetime_losses) as total_losses
                FROM swarm_agents
            """).fetchone()

            if stats and stats["total_agents"] > 0:
                sp1, sp2, sp3, sp4, sp5 = st.columns(5)
                sp1.metric("Active Agents", stats["active_agents"] or 0)
                sp2.metric("Total Trades", stats["total_trades"] or 0)
                sp3.metric("Wins", stats["total_wins"] or 0)
                sp4.metric("Losses", stats["total_losses"] or 0)
                total_t = stats["total_trades"] or 0
                total_w = stats["total_wins"] or 0
                win_rate = total_w / total_t if total_t > 0 else 0
                sp5.metric("Win Rate", f"{win_rate:.1%}")

                # Archetype performance chart
                arch_perf = conn.execute("""
                    SELECT archetype,
                           SUM(lifetime_wins) as wins,
                           SUM(lifetime_losses) as losses,
                           SUM(lifetime_trades) as trades,
                           AVG(reputation_score) as avg_rep
                    FROM swarm_agents WHERE is_active = 1
                    GROUP BY archetype
                """).fetchall()

                if arch_perf:
                    perf_data = []
                    for r in arch_perf:
                        trades = r["trades"] or 0
                        wins = r["wins"] or 0
                        perf_data.append({
                            "Archetype": r["archetype"].replace("_", " ").title(),
                            "Win Rate": round(wins / trades, 3) if trades > 0 else 0,
                            "Trades": trades,
                            "Avg Reputation": round(r["avg_rep"], 3) if r["avg_rep"] else 0.5,
                        })

                    perf_df = pd.DataFrame(perf_data)
                    if not perf_df.empty and perf_df["Trades"].sum() > 0:
                        fig_arch = px.bar(
                            perf_df, x="Archetype", y="Win Rate",
                            title="Win Rate by Archetype",
                            color="Win Rate",
                            color_continuous_scale=["#FF4444", "#FFAA00", "#44FF44"],
                            range_color=[0.3, 0.7],
                            text="Trades",
                        )
                        fig_arch.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5)
                        fig_arch.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_arch, use_container_width=True)

                st.markdown("---")

                # Agent Leaderboard
                st.markdown("### Agent Leaderboard (Top 20)")
                leaders = conn.execute("""
                    SELECT persona_name, archetype, win_rate, lifetime_trades,
                           lifetime_wins, lifetime_losses, reputation_score
                    FROM swarm_agents
                    WHERE is_active = 1 AND lifetime_trades >= 5
                    ORDER BY win_rate DESC, reputation_score DESC
                    LIMIT 20
                """).fetchall()

                if leaders:
                    leader_data = []
                    for r in leaders:
                        leader_data.append({
                            "Agent": r["persona_name"],
                            "Archetype": r["archetype"].replace("_", " ").title(),
                            "Win Rate": f"{r['win_rate']:.1%}",
                            "Trades": r["lifetime_trades"],
                            "W/L": f"{r['lifetime_wins']}/{r['lifetime_losses']}",
                            "Reputation": f"{r['reputation_score']:.2f}",
                        })
                    st.dataframe(pd.DataFrame(leader_data), use_container_width=True)
                else:
                    st.info("No agents with enough trades for leaderboard yet.")

                st.markdown("---")

                # Recent Simulation Rounds
                st.markdown("### Recent Simulation Rounds")
                rounds = conn.execute("""
                    SELECT round_number, cycle_number, posts_created,
                           positions_opened, positions_resolved, completed_at
                    FROM swarm_rounds
                    ORDER BY round_number DESC
                    LIMIT 20
                """).fetchall()

                if rounds:
                    round_data = []
                    for r in rounds:
                        round_data.append({
                            "Round": r["round_number"],
                            "Cycle": r["cycle_number"],
                            "Posts": r["posts_created"],
                            "Positions Opened": r["positions_opened"],
                            "Positions Resolved": r["positions_resolved"],
                            "Completed": r["completed_at"] or "in progress",
                        })
                    st.dataframe(pd.DataFrame(round_data), use_container_width=True)

                st.markdown("---")

                # Active Debates
                st.markdown("### Active Debates")
                debates = conn.execute("""
                    SELECT t.ticker, t.direction, t.confidence, t.content,
                           COUNT(c.post_id) as challenges
                    FROM swarm_posts t
                    LEFT JOIN swarm_posts c ON c.in_reply_to = t.post_id
                    WHERE t.channel = 'theses'
                    GROUP BY t.post_id
                    HAVING challenges > 0
                    ORDER BY challenges DESC
                    LIMIT 10
                """).fetchall()

                if debates:
                    for d in debates:
                        direction_color = "#00FF00" if d["direction"] == "BULLISH" else "#FF4444" if d["direction"] == "BEARISH" else "#AAAAAA"
                        st.markdown(
                            f'<div style="background-color:#262730;padding:12px;border-radius:8px;'
                            f'margin-bottom:8px;border-left:4px solid {direction_color};">'
                            f'<strong>{d["ticker"]}</strong> — {d["direction"]} '
                            f'(conf: {d["confidence"]:.0%}) — '
                            f'{d["challenges"]} challenge(s)<br/>'
                            f'<small>{d["content"][:200]}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No active debates yet. Enable the swarm to start simulations.")

            else:
                st.info(
                    "No swarm data yet. Enable the Trading Swarm above and start the "
                    "ContinuousMonitorAgent to begin swarm simulations."
                )

            conn.close()
        except Exception as e:
            st.warning(f"Could not load swarm data: {e}")

        st.markdown("---")

        # Anti-Convergence Settings
        st.markdown("### Anti-Convergence Settings")
        anti_conv = swarm_cfg.get("anti_convergence", {})
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Max Agreement Ratio", f"{anti_conv.get('max_agreement_ratio', 0.85):.0%}")
        ac2.metric("Forced Contrarian %", f"{anti_conv.get('forced_contrarian_pct', 0.10):.0%}")
        ac3.metric("Mutation Rate", f"{anti_conv.get('opinion_mutation_rate', 0.05):.0%}")

    # ── Tab 14: Cross-Intelligence (TidalFlowBridge) ──
    with tab14:
        st.subheader("Cross-Intelligence — TidalShift Connection")

        try:
            from bridge.bridge_startup import get_bridge_orchestrator, get_bridge_adapter, is_bridge_enabled
            from bridge.cascade_evidence import format_pathways_for_streamlit, format_warnings_for_streamlit

            bridge_orch = get_bridge_orchestrator()
            bridge_adapt = get_bridge_adapter()

            if not is_bridge_enabled():
                st.info(
                    "TidalFlowBridge is not active. To enable:\n"
                    "1. Set `BRIDGE_PASSPHRASE` environment variable\n"
                    "2. Run `tidalflowbridge-setup` in the FlowTrace directory\n"
                    "3. Pair with TidalShift: `tidalflowbridge-pair --peer-url http://<host>:8200`\n"
                    "4. Restart FlowTrace"
                )
            else:
                bridge_status = bridge_orch.get_status()

                # --- Status Metrics ---
                st.markdown("### Bridge Status")
                bs1, bs2, bs3, bs4, bs5, bs6 = st.columns(6)
                bs1.metric("Status", "ONLINE" if bridge_status.get("running") else "OFFLINE")
                identity = bridge_status.get("identity", {})
                bs2.metric("Node", identity.get("node_alias", "N/A"))
                conns = bridge_status.get("connections", {})
                bs3.metric("Peers", f"{conns.get('online_peers', 0)}/{conns.get('total_peers', 0)}")
                pub = bridge_status.get("publisher", {})
                bs4.metric("Published", pub.get("total_published", 0))
                sub = bridge_status.get("subscriber", {})
                bs5.metric("Received", sub.get("total_received", 0))
                budget = bridge_status.get("budget", {})
                bs6.metric("Budget Left", f"${budget.get('remaining_usd', 0):.2f}")

                st.markdown("---")

                # --- Cascade Pathways from TidalShift ---
                st.markdown("### Cascading Pathways from TidalShift")
                cascade_cache = bridge_adapt.get_cascade_cache() if bridge_adapt else []
                if cascade_cache:
                    pathway_data = format_pathways_for_streamlit(cascade_cache)
                    st.dataframe(pd.DataFrame(pathway_data), use_container_width=True)

                    # Detailed view in expander
                    for pw in reversed(cascade_cache[-5:]):
                        with st.expander(f"{pw.get('label', 'Unknown')[:80]} — {pw.get('plausibility', 0):.0%}"):
                            pw_c1, pw_c2, pw_c3 = st.columns(3)
                            pw_c1.metric("Plausibility", f"{pw.get('plausibility', 0):.0%}")
                            pw_c2.metric("Domains", ", ".join(pw.get("domains_involved", [])))
                            pw_c3.metric("Horizon", pw.get("time_horizon", "weeks"))
                            st.markdown(f"**Sectors:** {', '.join(pw.get('affected_sectors', []))}")
                            st.markdown(f"**Entities:** {', '.join(pw.get('affected_entities', [])[:10])}")
                            if pw.get("summary"):
                                st.markdown(f"**Summary:** {pw['summary'][:500]}")
                            if pw.get("feedback_loops"):
                                st.markdown(f"**Feedback Loops:** {', '.join(str(fl) for fl in pw['feedback_loops'][:3])}")
                else:
                    st.caption("No pathways received from TidalShift yet.")

                st.markdown("---")

                # --- Cascade Warnings ---
                st.markdown("### Active Cascade Warnings")
                warnings = bridge_adapt.get_active_warnings() if bridge_adapt else []
                if warnings:
                    warning_data = format_warnings_for_streamlit(warnings)
                    st.dataframe(pd.DataFrame(warning_data), use_container_width=True)
                else:
                    st.caption("No active cascade warnings.")

                st.markdown("---")

                # --- Recent Signals ---
                st.markdown("### Recent Bridge Signals")
                recent = bridge_adapt.get_recent_received(limit=15) if bridge_adapt else []
                if recent:
                    for sig in recent:
                        sig_type = sig.get("signal_type", "unknown")
                        conf = sig.get("confidence", 0)
                        color = "#3b82f6" if conf > 0.7 else ("#f59e0b" if conf > 0.4 else "#64748b")
                        st.markdown(
                            f'<div style="background-color:#262730;padding:10px;border-radius:8px;'
                            f'margin-bottom:8px;border-left:4px solid {color};">'
                            f'<strong style="color:{color};">{sig_type.replace("_", " ").upper()}</strong> '
                            f'<span style="color:#94a3b8;">— {sig.get("summary", "")}</span><br/>'
                            f'<small style="color:#64748b;">'
                            f'Confidence: {conf:.0%} | From: {sig.get("source", "?")} | {sig.get("timestamp", "")[:19]}'
                            f'</small></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No signals received yet.")

                st.markdown("---")

                # --- Ask TidalShift ---
                st.markdown("### Ask TidalShift")
                with st.form("bridge_query_form"):
                    query_text = st.text_area(
                        "Question",
                        placeholder="e.g., What policy risks affect NVDA in the next 30 days?",
                        height=80,
                    )
                    bq1, bq2 = st.columns(2)
                    query_type = bq1.selectbox(
                        "Query Type",
                        options=["custom", "explain_move", "cascade_forecast", "entity_exposure", "pathway_status"],
                    )
                    query_depth = bq2.selectbox(
                        "Depth",
                        options=["cached", "lightweight", "standard", "full"],
                        index=0,
                    )
                    submitted = st.form_submit_button("Send Query")

                if submitted and query_text and bridge_orch:
                    with st.spinner("Querying TidalShift..."):
                        import asyncio
                        try:
                            resp = asyncio.run(bridge_orch.query_client.ask_peer(
                                question=query_text,
                                target_node_type="tidalshift",
                                query_type=query_type,
                                depth=query_depth,
                                user_initiated=True,
                            ))
                            if resp.status == "success":
                                st.success(f"Response (confidence: {resp.confidence:.0%}, cost: ${resp.cost_usd:.3f})")
                                st.json(resp.response)
                            elif resp.status == "partial":
                                st.warning(f"Partial response (confidence: {resp.confidence:.0%})")
                                st.json(resp.response)
                            else:
                                st.error(f"Query {resp.status}: {resp.error_message or 'Unknown error'}")
                        except Exception as e:
                            st.error(f"Query failed: {e}")

                st.markdown("---")

                # --- Peer Details ---
                st.markdown("### Peer Connections")
                peers = conns.get("peers", [])
                if peers:
                    for peer in peers:
                        status_icon = "+" if peer.get("is_online") else "-"
                        local_tag = " (local)" if peer.get("is_local") else ""
                        latency = f" — {peer.get('latency_ms', 0):.0f}ms" if peer.get("latency_ms") else ""
                        st.markdown(
                            f"**[{status_icon}] {peer.get('node_alias', '?')}** "
                            f"({peer.get('node_type', '?')}){local_tag}{latency}"
                        )
                else:
                    st.caption("No peers connected.")

        except ImportError:
            st.info(
                "TidalFlowBridge package is not installed. To enable cross-intelligence:\n\n"
                "`pip install -e ../TidalFlowBridge`"
            )
        except Exception as e:
            st.error(f"Bridge tab error: {e}")

# Auto-refresh logic
time.sleep(refresh_rate)
st.rerun()