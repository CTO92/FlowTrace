# FlowTrace: Autonomous Financial Intelligence Agent

**THIS TOOL IS AN ALPHA VERSION MEANT TO PROVIDE A PLACE FOR DEVELOPERS TO WORK FROM IT IS NOT PRODUCTION-READY**

**FlowTrace** is an advanced, local AI platform designed to act as an autonomous "Hedge Fund Analyst Swarm." It combines real-time event monitoring with deep agentic research to uncover trading opportunities, risks, and market intelligence.

The system leverages **Grok (xAI)** for reasoning, **LangGraph** for multi-agent orchestration, and is evolving to integrate **OpenClaw** capabilities for robust, stealthy web intelligence.


---

## ⚡ Key Features

-   **Multi-Agent Swarm**: Specialized agents for Macro, Technical, Fundamental, and Sentiment analysis.
- **Real-Time Ingestion**: Listens to institutional-grade news feeds via Polygon.io WebSockets.
- **OpenClaw Integration**: "Scout" agent uses stealth browser technology to scrape alternative data (Web Traffic, App Ranks, Job Trends).
- **Knowledge Graph**: Maps market relationships (Supply Chain, Competitors, Sector Peers) across all market caps using Finnhub data and SEC Filings.
- **Agentic Research Layer**: Spins up autonomous AI agents (LangGraph + Playwright) to search the web and scrape data when context is missing.
- **Grok Analysis**: Uses xAI's Grok model to calculate a "Unified Correlation Score" based on price, fundamentals, and sentiment.
-   **Peer Comparison**: Compares companies against their competitors on key metrics.
-   **SEC Filing Analysis**: Searches and retrieves specific sections from SEC filings.
-   **News Aggregation**: Fetches news from RSS feeds.
-   **Short Interest Tracking**: Retrieves short interest data and days-to-cover metrics.
-   **Earnings Analysis**: Retrieves upcoming earnings dates and estimates.
-   **Seasonality Analysis**: Identifies seasonal patterns in stock returns.
-   **Correlation Analysis**: Calculates rolling correlations between assets and benchmarks.
-   **Volatility Analysis**: Gauges market fear through VIX term structure analysis.
-   **Sector Rotation**: Analyzes sector ETF momentum to suggest allocations.
- **Interactive Dashboard**: A Streamlit UI for live monitoring, signal history, and manual agent triggers.
-   **Supply Chain Visualization**: Generates Graphviz DOT code to visualize supply chain relationships.
- **Portfolio Management**: Integrated paper trading system to track positions, performance, and equity curves.
- **Risk Analysis**: Real-time calculation of Value at Risk (VaR), Sharpe Ratio, and holdings correlation matrices.
- **Macro Dashboard**: Visualizes key economic indicators (Fed Rates, Yield Curve, CPI) and commodity prices.
- **Analyst Chat**: Conversational interface to task the agent swarm with custom research requests.
- **Backtesting**: Historical replay module to validate strategies against past events.
- **Desktop Alerts**: Native notifications for high-confidence signals.

---

## 🏗️ Architecture

The system is composed of six main layers:

1.  **Ingestion Layer** (`ingestion_listener.py`): Connects to Polygon.io, filters news for market-moving events, and queries the graph for related assets.
2.  **Knowledge Graph** (`knowledge_graph.db`): SQLite database storing company nodes and relationship edges.
3.  **Agentic Layer** (`agent_workflow.py`): A Supervisor Agent delegates tasks to a swarm of specialized agents to build a comprehensive thesis.
4.  **Analysis Layer** (`grok_analysis.py`): Sends aggregated context (News + Graph + Agent Findings) to Grok for a structured prediction.
5.  **Portfolio Layer** (`portfolio_manager.py`): Manages paper trading accounts, calculates risk metrics, and fetches macro data.
6.  **UI Layer** (`app.py`): Streamlit dashboard for visualization and control.

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- xAI API Key (for Grok)
- Polygon.io API Key (for News/Market Data)
- Finnhub API Key (for Supply Chain Data)
- Serper API Key (Optional, for Agent Web Search)

### Local Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd PortfolioResearch
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install pyvis
    ```

3.  **Install Browser Binaries** (for Agent scraping):
    ```bash
    python -m playwright install chromium
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory (see `.env` template in code) and add your API keys:
    ```ini
    XAI_API_KEY=your_key_here
    POLYGON_API_KEY=your_key_here
    FINNHUB_API_KEY=your_key_here
    SERPER_API_KEY=your_key_here
    SEC_EMAIL=user@example.com
    ```

5.  **Verify Setup**:
    Run the environment check script to ensure everything is ready.
    ```bash
    python check_env.py
    ```

---

## 🛠️ Usage

### Phase 1: Build the Knowledge Graph
Initialize the database and seed it with supply chain data from Finnhub.
```bash
python build_knowledge_graph.py
```
*Note: This creates `knowledge_graph.db` locally.*

### Phase 2: Start the Ingestion Listener
Run the backend listener. This process monitors the news feed, triggers agents, performs analysis, and saves signals to the DB.
```bash
python ingestion_listener.py
```
*Keep this running in a separate terminal window.*

### Phase 3: Launch the Dashboard
Start the Streamlit interface to view live signals, interact with agents, and manage the portfolio.
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

---

## 🐳 Docker Deployment

You can run the entire stack (Listener + Dashboard) using Docker Compose.

1.  Ensure Docker Desktop is running.
2.  Run the compose command:
    ```bash
    docker-compose up --build
    ```
3.  Access the dashboard at `http://localhost:8501`.

*Note: Desktop notifications (Plyer) may not work inside the containerized environment.*

---

## 🧪 Backtesting & Maintenance

### Run Backtest
Validate the strategy using historical data (last 90 days).
```bash
python backtest.py
```
Results are saved to `backtest_results.csv`.

### Update Graph via EDGAR
Download and parse the latest 10-Q filings to find new supplier relationships using Grok.
```bash
python update_knowledge_graph.py
```

### Export Graph for Visualization
Export the nodes and edges to CSV files compatible with Gephi for network analysis.
```bash
python export_graph.py
```

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `ingestion_listener.py` | Main entry point. Listens for news, triggers workflow. |
| `app.py` | Streamlit dashboard source code. |
| `agent_workflow.py` | LangGraph definition for the Agentic Research Layer. |
| `agent_tools.py` | Tools for agents (Web Search, Scraper). |
| `agent_tools_scout.py` | **New**: Alternative data tools using OpenClaw stealth tech. |
| `openclaw_wrapper.py` | **New**: Stealth browser session manager. |
| `grok_analysis.py` | Interface for xAI API interaction. |
| `agent_workflow.py` | Defines multi-agent workflows orchestrated by the Supervisor. |
| `build_knowledge_graph.py` | Scripts to seed SQLite DB from Finnhub. |
| `update_knowledge_graph.py` | Scripts to update DB from SEC EDGAR filings. |
| `backtest.py` | Historical simulation and validation script. |
| `check_env.py` | Environment verification utility. |
| `requirements.txt` | Python dependencies. |
| `Dockerfile` | Container definition. |
| `docker-compose.yml` | Multi-container orchestration. |
| `report_generator.py` | Generates PDF reports of signal data. |


---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Trading stocks, especially small-caps, involves significant risk. The authors are not responsible for any financial losses incurred while using this software.

## 📄 License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE.md) file for details.
```

<!--
[PROMPT_SUGGESTION]Implement a 'Simulation Mode' in the dashboard to manually input a fake news headline and see how the system analyzes it.[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Add a feature to export the Knowledge Graph to a Gephi-compatible format for visualization.[/PROMPT_SUGGESTION]
-->