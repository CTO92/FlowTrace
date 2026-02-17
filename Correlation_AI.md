**Small-Cap FlowTrace**  
**Version: Enhanced with Comprehensive Multi-Factor Correlation Analysis + Agentic Research Capabilities**

---

## **Executive Summary**

This project develops a high-performance, local Python application designed to detect high-probability trading opportunities in small-cap stocks (typically market caps under $2 billion, such as those in the Russell 2000 index) by analyzing real-time news events from their Fortune 500 customers. The core innovation lies in leveraging supply-chain dynamics: major announcements from large corporations (e.g., earnings beats, contract expansions, or strategic partnerships) often cascade positive impacts to their smaller vendors, suppliers, or partners, leading to short-term price momentum in those small-caps.

The enhanced version introduces a **comprehensive multi-factor correlation analysis** that integrates **all key data points**: news events, supply-chain relationships, price action (historical and real-time, including rolling correlations and technical indicators), options market data (IV, OI, flow), fundamentals (e.g., EPS, P/E, ROE, inventory turnover, DIO), macroeconomic/geopolitical indicators (e.g., interest rates, inflation, commodity prices, trade policies), market sentiment/alternative data (e.g., social media buzz, earnings transcripts, short interest), volatility metrics (e.g., historical vol, VIX correlations), sector/peer/intermarket data (e.g., sector indices, insider activity), and advanced statistical enhancements (e.g., nonlinear measures like mutual information, outlier adjustments, graph-based connectedness). These factors are combined into a unified correlation strength score (e.g., via weighted averaging or machine learning in backtesting), capturing nonlinear, time-varying, and external influences for superior prediction of cascading effects.

**New: Agentic Research Layer**  
To achieve true depth and adaptability, the system now includes **agentic capabilities**. On-demand AI agents are dynamically spun up to perform targeted research, crawl for missing data, and generate supplementary analyses. This turns the application from a reactive signal generator into a proactive intelligence platform — agents can autonomously gather real-time filings, analyze earnings calls, monitor geopolitical shifts, or validate correlations with fresh external sources.

By switching from Gemini 1.5 Pro to Grok (via the xAI API), we dramatically enhance capabilities. Grok's strengths include ultra-large context windows (up to 2 million tokens), low-latency reasoning (optimized for fast-reasoning models), cost efficiency ($0.20 per million input tokens, $0.50 per million output), and superior structured output handling for quantitative analysis. This enables more accurate correlation predictions, reduced hallucinations through evidence-based prompting, and integration of historical precedents in a single call.

Key improvements over the original plan:
- **Dynamic Mapping**: Replace static CSV with an auto-updating knowledge graph built from EDGAR filings, API data, and vector embeddings for semantic relationships.  
- **Agentic Research**: Autonomous agents crawl and analyze on-demand (e.g., scrape new 10-Qs, extract transcript insights, monitor X sentiment in real-time).  
- **Comprehensive Multi-Factor Correlations**: Analyze and correlate all factors (price, options, fundamentals, macro, sentiment, technical, sector, advanced stats) between F500 and small-caps, combining into a unified score.  
- **Tiered Analysis**: Asynchronous pipeline with pre-filtering, Grok-powered reasoning (over all factors), and real-time multi-validation to ensure signals have immediate market confirmation.  
- **Backtesting Integration**: Built-in historical replay (including all factors) to calibrate confidence thresholds, aiming for 60-70% directional accuracy and positive expectancy (e.g., average 1-5% abnormal returns on validated signals).  
- **Latency Optimization**: End-to-end under 3 seconds using WebSockets and efficient models.  
- **UI and Alerts**: Interactive dashboard for live feed, signal history, and performance metrics.  
- **Scalability**: Modular design supports expansion to more APIs, custom prompts, or even multi-model ensembles if needed.

The result is a tool that generates actionable trade ideas with statistical edge, suitable for day/swing traders focusing on event-driven strategies. It's not for autonomous trading but for informed decision-making. Estimated setup time: 6-8 hours for core functionality, plus 3-5 hours for backtesting, UI, and agent orchestration.

## **Technical Architecture**

The application runs locally as an asynchronous, modular pipeline using Python 3.10+. It processes news in real-time, maps impacts to small-caps, analyzes with Grok (over comprehensive multi-factor correlations), deploys agents for deeper research, validates with market data, and reports signals. Core components:

### **A. Ingestion Layer (News and Data Aggregator)**

* **Primary Sources**:  
  * Polygon.io: Institutional-grade real-time news via WebSocket (`/vX/reference/news` endpoint). Includes ticker tagging, sentiment metadata, and low-latency quotes/trades/volumes/options/fundamentals/technical indicators for validation. Filter for Fortune 500 tickers (e.g., AAPL, WMT). Enhanced to fetch options snapshots (`/v3/snapshot/options/{ticker}`), fundamentals (`/vX/reference/financials`), and technical data (e.g., via aggregates for RSI/MACD).  
  * Finnhub.io: Complementary WebSocket for supply-chain data (`/stock/supply-chain?symbol=XXX`), company-specific news, fundamentals (`/stock/fundamental-metric`), short interest (`/stock/short-interest`), insider activity (`/stock/insider-transactions`), economic calendar (for macro like CPI), and sector/peer metrics.  
* **Backup/Historical**: Tiingo for deduplicated historical feeds (backtesting) and StockNewsAPI for additional sentiment scoring. Use Polygon historical APIs for past options, prices, fundamentals, and technicals; integrate FRED (via `pandas_datareader`) for macro (e.g., rates, inflation); Alpha Vantage or Quandl for commodities/intermarket data.  
* **Additional Sources**:  
  * Macro/Geopolitical: FRED API for rates/yields/inflation; web scraping or APIs like Trading Economics for tariffs/policies (use `requests` with user keys).  
  * Sentiment/Alternative: X API (via `tweepy` or semantic search tools) for social buzz; Alpha Vantage for earnings transcripts; Ortex via API for advanced short interest/borrow rates.  
  * Sector/Intermarket: Finnhub for sector indices; Yahoo Finance (via `yfinance`) for peer groups and intermarket (e.g., ETFs, currencies).  
* **Data Flow**: Asynchronous listeners fetch JSON-structured news every 5-60 seconds (configurable). Pre-filter for relevance: Must mention a Fortune 500 ticker and keywords like "contract," "partnership," "expansion," or "earnings." Upon trigger, fetch all data (price, options, fundamentals, macro, sentiment, technical, sector) for both F500 and small-cap tickers.  
* **Additional Inputs**: Real-time price/volume snapshots, options metrics (IV, OI, call/put ratios), fundamentals (e.g., EPS delta, inventory turnover changes), macro indicators (e.g., rate changes, commodity prices), sentiment scores (e.g., X mentions), technical indicators (e.g., RSI crossovers), short interest, insider activity.

### **B. Knowledge Graph Layer (Dynamic Customer-Vendor Mapping)**

* **Structure**: Hybrid graph database using NetworkX (for in-memory traversal) + SQLite (persistent storage) + ChromaDB (vector embeddings for semantic queries).  
  * Nodes: Companies (Fortune 500 as "hubs," small-caps as "spokes") with attributes like latest fundamentals (EPS, P/E, ROE, inventory turnover, DIO, cash-to-cash cycle, debt-to-equity, ROA), technical indicators (RSI, MACD), short interest, insider activity, sector affiliation.  
  * Edges: Relationships (e.g., "supplier," "partner") with attributes like % revenue dependency, historical price correlation coefficient (including rolling/cointegration), options IV correlation, fundamental correlations (e.g., EPS beta, inventory turnover linkage), macro betas (e.g., interest rate sensitivity, commodity exposure), sentiment correlations (e.g., social buzz linkage), volatility measures (e.g., vol beta), sector/peer betas, intermarket correlations (e.g., FX exposure), nonlinear measures (mutual information), graph connectedness scores, last update date.  
* **Auto-Building/Updating**:  
  * Seed from Finnhub supply-chain API.  
  * Enhance via SEC EDGAR filings: Use `sec-edgar-downloader` to fetch recent 10-K/10-Q for small-caps; extract customer mentions from "Business" and "Risk Factors" sections using Grok for structured parsing; include earnings transcripts for qualitative correlations.  
  * Embed relationship descriptions with sentence-transformers for fuzzy matching (e.g., query "cloud security providers for AWS").  
  * Compute Correlations: Use historical data from Polygon/Finnhub/FRED to calculate comprehensive multi-factor correlations (e.g., Pearson for linear, scipy `mutual_info_regression` for nonlinear, statsmodels for cointegration; apply outlier trimming via robust stats; use GCN for graph propagation).  
  * Update quarterly or on-demand (e.g., via cron job or manual trigger), refreshing all data points and recalculating correlations.  
* **Benefits**: Handles thousands of relationships dynamically, reducing false positives from outdated static data. Comprehensive multi-factor correlations provide a holistic view of dependency strength across regimes.

### **C. Agentic Research Layer (New: On-Demand Intelligence)**

* **Core Concept**: A **multi-agent system** powered by LangGraph (stateful workflows) and Grok as the central LLM. Agents are "spun up" asynchronously when a news event triggers deeper investigation.
* **Agent Types** (dynamically instantiated via Python multiprocessing or Celery for parallelism):
  - **Research Crawler Agent**: Uses Playwright/Selenium + BeautifulSoup to scrape EDGAR, company websites, or news archives for missing data (e.g., latest 8-K filings).
  - **Transcript Analyst Agent**: Downloads and parses earnings call transcripts (via Alpha Vantage or SEC) to extract supplier mentions and tone analysis.
  - **Macro/Geopolitical Agent**: Queries FRED, Trading Economics, and news APIs for real-time impacts (e.g., "How does this tariff affect WMT suppliers?").
  - **Sentiment Deep-Dive Agent**: Uses X API + semantic search to track buzz, plus web search for analyst reports.
  - **Validation Agent**: Runs code execution (via local REPL) to recompute correlations or simulate scenarios.
* **Orchestration**: Grok acts as **Supervisor Agent** using tool-calling:
  - On high-confidence signal, it decides which agents to spawn (e.g., "Event involves cloud spending → spin up Transcript Agent + Macro Agent").
  - Agents return structured results (JSON) that feed back into the knowledge graph and Grok's final reasoning.
  - Parallel execution keeps core signal latency <3s; deeper research runs in background and updates signals in real-time.
* **Tools**: LangChain tools for:
  - Web search (Serper/DuckDuckGo API)
  - Browser automation (Playwright)
  - Code interpreter (local Python REPL with pandas/numpy)
  - Custom API wrappers (Polygon, Finnhub)
* **Benefits**: Agents make the system "alive" — it self-improves by fetching fresh data, reducing stale correlations, and uncovering hidden insights (e.g., a new partnership buried in a 10-Q).

### **D. Analysis Layer (Grok-Powered Reasoning)**

* **Tiered Processing**:
  - **Stage 1 (Fast Pre-Filter)**: Rule-based or lightweight model (e.g., local spaCy for NER) to extract entities, classify events (e.g., "New Contract," "Supply Chain Expansion"), and query the graph for connected small-caps.
  - **Stage 2 (Grok Core + Agents)**: Single API call to `grok-4-1-fast-reasoning` (or equivalent fast model) with 2M+ token context:
    * Inputs: Full news text, graph subgraph (relevant edges, including all correlations), recent data for all factors (price/volume/options/fundamentals/macro/sentiment/technical/sector/intermarket), agent research outputs, 3-5 few-shot examples of past events/outcomes (including comprehensive factors).
    * System Prompt: "You are a quantitative hedge fund analyst specializing in supply-chain event studies, options markets, fundamental analysis, macro/sentiment/technical factors, and advanced stats. Given this news about [F500 Company], classify the event type and estimate impact on connected small-cap vendors. Incorporate agent research outputs. Perform comprehensive multi-factor correlation analysis: Correlate price action (historical/rolling betas, cointegration), options (IV/OI/flow), fundamentals (e.g., EPS growth, ROE, inventory turnover transmission), macro (rates, inflation, commodities, tariffs), sentiment (social buzz, transcripts, short interest), technical (RSI/MACD/vol), sector/peer/intermarket (indices, insider activity), and advanced stats (nonlinear mutual info, outlier-adjusted, graph connectedness). Combine all into a unified correlation strength score (0-1, e.g., weighted average). For each: Project revenue lift (%), expected 1/5/20-day abnormal return, unified multi-factor corr strength, historical precedent strength, and overall confidence (0-100). Output strict JSON: {'targets': [{'ticker': str, 'expected_move_pct': float, 'revenue_impact_est_pct': float, 'unified_multi_factor_corr_strength': float (0-1), 'price_corr_beta': float, 'options_corr_strength': float, 'fundamentals_corr_examples': list[str], 'macro_corr_examples': list[str], 'sentiment_corr_strength': float, 'technical_corr_examples': list[str], 'sector_intermarket_corr': float, 'advanced_stats_notes': str, 'confidence': int, 'reasoning_chain': str, 'comparable_events': list[str]}]}."
    * Output: Structured JSON for easy parsing, focusing on causal chains (e.g., "F500 EPS beat correlates 0.75 with small-cap price pop; inflation spike amplifies commodity exposure, unified score 0.85").
  - **Validation Sub-Layer**: Post-Grok, cross-check with real-time data:
    * Volume spike >3x 20-day average.
    * Positive abnormal return in last 1-5 minutes.
    * Comprehensive Checks: Options/price/fundamentals/macro/sentiment/technical correlations > historical avgs; require unified multi-factor score >0.75.

### **E. Reporting and UI Layer**

* **Outputs**: For each signal above threshold (e.g., confidence >80%, unified multi-factor corr >0.75, calibrated via backtest):
  * Headline: Triggering event summary.
  * Targets: List of small-caps with expected move, confidence, reasoning, and comprehensive insights (e.g., "Unified corr 0.88: Price beta 1.2, options sync 0.8, fundamentals EPS strong, macro rates sensitivity 0.7, sentiment buzz high").
  * Metrics: Projected returns, risk factors (e.g., market volatility, macro mismatches).
  * Visuals: Integrated charts (e.g., via Matplotlib/Plotly) for price action, options IV, fundamental correlations, macro overlays, sentiment trends, technical indicators.
  * Agent Insights: Dedicated section showing research outputs (e.g., "Research Agent found: new supplier mention in latest 10-Q").
* **Interface**: Streamlit dashboard for live feed, signal history table (with realized returns and multi-factor metrics), backtest results (Sharpe ratio, win rate by bucket, correlation accuracy), and "Agent Activity Log" tab for transparency.
* **Alerts**: Desktop notifications (plyer library), Telegram bot, or email for high-confidence signals, including multi-factor highlights.

### **F. Backtesting Module**

* **Purpose**: Validate and optimize the pipeline with comprehensive multi-factor correlations.
* **Data**: Historical news/prices/options/fundamentals/macro/sentiment/technical/sector from Polygon/Tiingo/Finnhub/FRED/Alpha Vantage (e.g., last 2 years; compute past correlations including nonlinear/graph-based).
* **Process**: Replay events through the pipeline, compute cumulative abnormal returns (CAR), win rate, expectancy, and multi-factor hit rates (e.g., did predicted correlations hold across factors?). Bucket by confidence and unified corr strength to set thresholds (e.g., >85% for alerts, multi-factor >0.8). Simulate agent behaviors in replay (e.g., "What if agents had fetched this filing?").
* **Metrics Table Example** (hypothetical post-backtest):

| Confidence Bucket | Win Rate (%) | Avg CAR (1-Day) | Avg CAR (5-Day) | Avg Unified Multi-Factor Corr Strength | Expectancy (%) | Sample Size |
|-------------------|--------------|-----------------|-----------------|----------------------------------------|----------------|-------------|
| 70-80             | 55           | 1.2             | 2.8             | 0.65                                   | 0.66           | 150         |
| 81-90             | 65           | 2.5             | 4.1             | 0.78                                   | 1.63           | 100         |
| 91+               | 75           | 3.8             | 6.2             | 0.88                                   | 2.85           | 50          |

## **Detailed Implementation Plan**

Follow this phased rollout to build and deploy. Assume basic Python proficiency; total code ~900-1800 lines (added comprehensive factors and agents).

### **Phase 1: Environment Setup (30-60 minutes)**

1. Obtain API Keys:  
   * xAI: Sign up at console.x.ai, generate key.  
   * Polygon.io: Free tier for starters; upgrade for WebSockets, options, fundamentals.  
   * Finnhub: Premium for supply-chain, fundamentals, short interest, macro.  
   * Additional: Alpha Vantage (free for transcripts/commodities), FRED (no key), Ortex (if premium short data).  
2. Install Dependencies:  
   ```bash
   pip install xai-sdk openai aiohttp websockets polygon-api-client streamlit pandas numpy networkx chromadb sentence-transformers sec-edgar-downloader plyer requests scipy statsmodels pandas_datareader yfinance tweepy langgraph langchain langchain-community playwright streamlit-aggrid plotly
   ```
   (Note: Use OpenAI-compatible client for xAI by setting `base_url="https://api.x.ai/v1"`. Added LangGraph/LangChain for agents, Streamlit extras for UI.)  
3. Configure Environment: Create `.env` file with keys (e.g., `XAI_API_KEY=sk-...`).

### **Phase 2: Build Knowledge Graph (1-2 hours)**

1. Seed Data: Script to fetch Finnhub supply-chain for top 100 Fortune 500 tickers; store in SQLite (`companies` and `relationships` tables).  
2. EDGAR Integration: Download recent filings/transcripts; chunk text and prompt Grok to extract structured relationships and qualitative correlations.  
3. Comprehensive Correlations: Fetch all historical data; compute and store correlations (e.g., via scipy pearsonr/mutual_info, statsmodels cointegration, robust stats for outliers, networkx for GCN connectedness).  
4. Vector Embeddings: Use sentence-transformers (`all-MiniLM-L6-v2`) to embed edge descriptions; store in ChromaDB.  
5. Update Script: Function to refresh graph quarterly, including all factors.

### **Phase 3: Ingestion and Pre-Processing (1 hour)**

1. Async Listener: Use `asyncio` and Polygon/Finnhub WebSockets to subscribe to news/quotes/options/fundamentals/technical/macro/sentiment for Fortune 500 list.  
2. Pre-Filter: On news receipt, use regex/spaCy to confirm F500 mention and event keywords; query graph for connected small-caps (discard if none). Fetch all comprehensive data for both tickers.

### **Phase 4: Grok Analysis Integration (1-2 hours)**

1. Client Setup: Initialize xAI client with fast-reasoning model.  
2. Prompt Engineering: Define system prompt and few-shot examples (including all factors).  
3. API Call: Async function to build context payload (add all factors) and parse JSON response.  
4. Validation: Fetch 1-min data; apply rules (e.g., `if volume > 3 * avg_20d and return > 0 and unified_multi_factor_corr > 0.75: signal = True`). Use scipy/statsmodels for real-time calcs.

### **Phase 4.5: Agentic Layer Setup (1 hour)**

1. Define agent classes (e.g., `ResearchAgent` with tools).  
2. Implement Supervisor prompt for Grok to route tasks (e.g., "Decide which agents to spawn and return JSON plan").  
3. Create LangGraph workflow: Supervisor → parallel sub-agents → merge results → feed back to Grok.  
4. Test parallel agent execution with `concurrent.futures` or Celery (local broker).

### **Phase 5: Reporting and UI (2.5–4 hours)**

**Goal**: Build a professional, interactive, local Streamlit dashboard that serves as the primary user interface — displaying live signals, detailed correlation analysis popovers, agent research logs, signal history, backtest results, and configuration controls.

**Sub-steps**:

1. **Project Structure & Streamlit Setup (20–30 min)**  
   - Create `app.py` as the main entry point.  
   - Set up basic layout: sidebar for navigation/config, main area for content.  
   - Use `st.set_page_config(layout="wide", page_title="Correlation Edge")` for full-screen experience.  
   - Install: `pip install streamlit streamlit-aggrid plotly` (for better tables and charts).  
   - Add custom CSS for dark/neon theme (holographic cards, glowing buttons, thin scrollbars).  

2. **Live Signals Feed & Cards (45–75 min)**  
   - Use `st.container()` + columns to create a scrollable feed of signal cards.  
   - Each card shows: trigger headline, target ticker(s), expected move %, confidence %, mini radar chart (via `st.plotly_chart` or Chart.js embed), short reasoning snippet.  
   - Make cards clickable → open modal/popover with full analysis (use `st.experimental_get_query_params` + session state or a custom modal component).  
   - Real-time updates: use `st.rerun()` in a background thread or `st.session_state` polling loop (every 5–10 s).  
   - Integrate agent insights: small badge or expander showing “Agent found: new 10-Q mention”.

3. **Detailed Signal Popover / Modal (45–60 min)**  
   - Build expandable modal using `st.dialog` (Streamlit 1.38+) or custom HTML/JS component via `st.components.v1.html`.  
   - Top: compact header (trigger → target, move %, confidence %, 1-line reasoning).  
   - Middle: large radar chart + price action mini-chart (Plotly or Chart.js).  
   - Bottom: scrollable “Full Multi-Factor Correlation Analysis” section (grid layout: left = charts/score, right = detailed text breakdown per factor).  
   - Ensure scroll container (`div` with `overflow-y: auto`) prevents clipping on smaller screens.

4. **Agent Activity Log Tab (20–30 min)**  
   - New tab in sidebar or top nav.  
   - Table (via `st.dataframe` or `streamlit-aggrid`) showing: timestamp, agent type, task, status, key output snippet.  
   - Color-code rows (green = success, yellow = partial, red = error).

5. **Signal History & Backtest Tabs (30–45 min)**  
   - **History**: SQLite-backed table with filters (date, confidence, ticker, unified score). Columns: time, trigger, target, move %, confidence, realized return (if available), agent notes.  
   - **Backtest**: Summary stats cards + interactive Plotly charts (equity curve, win-rate by bucket, factor importance heatmap). Use `vectorbt` or `pandas` for quick calculations.

6. **Alerts & Settings (20–30 min)**  
   - Settings sidebar: sliders for min confidence/unified score, toggle agent depth (light/medium/deep), API key inputs (masked).  
   - Alerts: `plyer` desktop notifications + optional `python-telegram-bot` integration (send high-confidence signals with link to dashboard).

7. **Polish & Testing (30–45 min)**  
   - Add loading spinners (`st.spinner`) during agent runs or data refresh.  
   - Theme toggle (dark/light) via session state + custom CSS.  
   - Mobile responsiveness (test on phone).  
   - End-to-end test: trigger fake news → see card appear → click → verify popover scrolls and charts render.

**Key Libraries**:
- `streamlit`, `streamlit-aggrid` (advanced tables), `plotly` (interactive charts), `plyer` (notifications), `python-telegram-bot` (optional)

**Gotchas & Best Practices**:
- Streamlit reruns the entire script on every interaction — use `st.session_state` and `@st.cache_data` / `@st.cache_resource` aggressively.
- Heavy charts/agents → run in background threads (e.g., `threading` or `concurrent.futures`) to avoid blocking UI.
- Modal/popover: if using custom HTML/JS, ensure it works in Streamlit’s iframe sandbox.
- Keep dashboard modular — separate files for components (e.g., `components/signal_card.py`) and import them.

**Estimated Total Time**: 2.5–4 hours (depending on how polished you want the visuals and how many agent insights you surface).

### **Phase 6: Backtesting and Optimization (1-2 hours)**

1. Historical Fetch: Script to download past data across all factors (e.g., Polygon bulk, FRED, yfinance).  
2. Replay Loop: Process historical events; log predicted vs. actual outcomes, including multi-factor correlations.  
3. Analysis: Use pandas to compute metrics; adjust thresholds/prompts based on results (e.g., weight factors in unified score).

### **Phase 7: Testing and Launch (30-60 minutes)**

1. Unit Tests: Mock news events and all data; verify end-to-end flow.  
2. Live Run: Start async loop; monitor for 1-2 days, logging signals.  
3. Iterate: Review hallucinations (update graph), latency (profile with cProfile), accuracy (compare to manual analysis, focus on multi-factor correlations).

## **Key Tools and Sources**

| Source/API | Key Required? | Best Feature | Usage in Project |
|------------|---------------|--------------|------------------|
| Polygon.io | Yes | Ultra-low latency WebSockets for news + market/options/fundamentals/technical data | Primary ingestion + validation + multi-factor correlations |
| Finnhub.io | Yes | Supply-chain, fundamentals, short interest, macro, insider endpoints | Graph seeding + historical betas across factors |
| xAI (Grok) | Yes | Fast reasoning with large context | Core analysis + structured predictions incl. comprehensive multi-factor |
| Tiingo | Yes | Deduplicated historical news | Backtesting |
| SEC EDGAR (via downloader) | No | Free access to filings/transcripts | Dynamic mapping + qualitative correlations |
| FRED (via pandas_datareader) | No | Macro indicators (rates, inflation) | Macro correlations |
| Alpha Vantage/yfinance | Yes/Free | Earnings transcripts, commodities, sector indices, intermarket | Alternative data + sector/peer correlations |
| X API (tweepy) | Yes | Social sentiment | Sentiment correlations |
| **LangGraph / LangChain** | No (local) | Multi-agent orchestration | Agentic research layer |

## **Risks and Mitigations**

* **Latency Issues**: Mitigated by async design, WebSockets, and tiered filtering; target <3s, monitor with logging. Additional fetches add ~1-2s, optimized via caching/batching.  
* **Hallucinations/False Positives**: Use structured outputs, graph validation, comprehensive confirmation, and backtested thresholds; quarterly graph refreshes.  
* **API Costs/Rate Limits**: Grok is cheap (~$0.01 per signal); expanded APIs increase costs—start with free tiers, monitor usage; fallback to polling if throttled.  
* **Data Quality**: Deduplicate across sources; handle incomplete data by prompting Grok for best-effort; validate illiquid metrics (e.g., skip sparse factors).  
* **Market Risks**: Emphasize as "ideas only"; include disclaimers on correlation vs. causation; track realized performance to avoid overtrading.  
* **Complexity/Scalability**: Modular code allows selective factor toggling; if volume high, consider cloud deployment (e.g., AWS Lambda) later.  
* **Agent Latency/Cost**: Agents run async/background; cap concurrent agents; use cheap local models for simple tasks.  
* **Data Privacy**: All crawling local; no external data storage.  
* **Over-Research**: Supervisor limits agents to a configurable amount (default to 3) per signal.
