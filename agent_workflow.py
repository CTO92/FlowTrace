import os
import json
import time
import asyncio
import datetime
import operator
from typing import Annotated, Sequence, TypedDict, List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_tools_agent, AgentExecutor
from dotenv import load_dotenv

from agent_tools import web_search, scrape_web_page, get_competitors, get_insider_trades, get_analyst_ratings, get_fundamental_ratios, get_earnings_calendar, get_short_interest, fetch_rss_feed, visualize_supply_chain, get_sec_filing_section, compare_peers, get_comprehensive_fundamentals, get_financial_statements, get_earnings_history
from agent_tools_advanced import get_macro_rates, get_reddit_sentiment, get_portfolio_exposure, execute_trade, propose_option_strategy, calculate_option_greeks, calculate_optimal_entry, validate_signal_robustness, calculate_atr_stop_loss, analyze_sentiment_bert, optimize_portfolio_mean_variance, analyze_sector_momentum, analyze_vix_term_structure, calculate_rolling_correlation, analyze_seasonality, analyze_technicals
from agent_tools_scout import get_web_traffic_metrics, get_app_store_rankings, get_job_market_trends, get_google_trends
from agent_tools_technical import analyze_chart_pattern
from node_identity import generate_agent_id, compute_persona_hash, get_node_id
from llm_config import get_langchain_llm

load_dotenv()

# --- Configuration ---
PERFORMANCE_LOG_FILE = os.path.join(os.path.dirname(__file__), "agent_performance.log")

# --- Agent Identity Registry ---
# Maps agent_type -> { agent_id, persona_hash } for the current session
_agent_identities = {}

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    swarm_brief: dict  # Optional: SwarmBrief from Trading Agent Swarm

# --- Agent Constructors ---

def _register_agent_identity(agent_type: str, system_prompt: str) -> dict:
    """Register or retrieve the identity for an agent type in this session."""
    if agent_type not in _agent_identities:
        _agent_identities[agent_type] = {
            "agent_id": generate_agent_id(agent_type),
            "persona_hash": compute_persona_hash(system_prompt),
            "agent_type": agent_type,
            "node_id": get_node_id(),
        }
    return _agent_identities[agent_type]


def get_agent_identities() -> dict:
    """Return the full agent identity registry for this session."""
    return dict(_agent_identities)


def create_agent(tools, system_prompt, agent_type=None):
    """Generic helper to create an agent with specific tools and persona.

    Uses the multi-LLM factory to get the correct model for this agent type.
    The provider/model is determined by llm_config.json — each agent can be
    assigned a different LLM backend if the trader desires.
    """
    # Register identity if agent_type provided
    if agent_type:
        _register_agent_identity(agent_type, system_prompt)

    # Get the LLM for this specific agent type (may differ per agent)
    agent_llm = get_langchain_llm(agent_type=agent_type, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
 ("user", "Please adhere to the system prompt and respond in a JSON format."),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(agent_llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# --- Specialized Agent Nodes ---
async def log_agent_performance(agent_name, start_time, success, error=None):
    """Logs agent performance metrics to a file."""
    end_time = time.time()
    duration = end_time - start_time
    
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent_name": agent_name,
        "duration": duration,
        "success": success,
        "error": error
    }
    
    with open(PERFORMANCE_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

async def research_node(state):
    """General Web Research Agent."""
    system_prompt = "You are a Research Agent. You can search the web, scrape pages, look up competitors, and check analyst ratings to find missing financial data, supplier relationships, or news details."
    agent = create_agent([web_search, scrape_web_page, get_competitors, get_analyst_ratings], system_prompt, agent_type="ResearchAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ResearchAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ResearchAgent")]}
    except Exception as e:
        await log_agent_performance("ResearchAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ResearchAgent failed with error: {e}", name="ResearchAgent")]}

async def macro_node(state):
    """Macro-Economic Agent using FRED and web search."""
    system_prompt = "You are a Macro Strategy Agent. You have a tool to fetch current interest rates from FRED. Focus on interest rates, inflation, central bank policy, and geopolitical events. Use your tools to answer questions."
    agent = create_agent([web_search, get_macro_rates], system_prompt, agent_type="MacroAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("MacroAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="MacroAgent")]}
    except Exception as e:
        await log_agent_performance("MacroAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"MacroAgent failed with error: {e}", name="MacroAgent")]}

async def sentiment_node(state):
    """Sentiment & Alternative Data Agent for Reddit and web scraping."""
    system_prompt = "You are a Sentiment Analyst. You have a tool to scrape Reddit for ticker sentiment. You investigate social media buzz, consumer sentiment, and alternative data trends. You also check insider trading activity to gauge management confidence."
    agent = create_agent([web_search, scrape_web_page, get_reddit_sentiment, get_insider_trades], system_prompt, agent_type="SentimentAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("SentimentAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="SentimentAgent")]}
    except Exception as e:
        await log_agent_performance("SentimentAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"SentimentAgent failed with error: {e}", name="SentimentAgent")]}

async def risk_node(state):
    """Risk Management Agent."""
    system_prompt = "You are a Risk Manager. Your job is to check portfolio exposure and assess concentration risk. Use the get_portfolio_exposure tool to check current holdings before we commit to a trade."
    agent = create_agent([get_portfolio_exposure], system_prompt, agent_type="RiskManagerAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("RiskManagerAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="RiskManagerAgent")]}
    except Exception as e:
        await log_agent_performance("RiskManagerAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"RiskManagerAgent failed with error: {e}", name="RiskManagerAgent")]}

async def execution_node(state):
    """Execution Agent."""
    system_prompt = "You are an Execution Trader. Your job is to execute trades based on instructions. Use the execute_trade tool."
    agent = create_agent([execute_trade], system_prompt, agent_type="ExecutionAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ExecutionAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ExecutionAgent")]}
    except Exception as e:
        await log_agent_performance("ExecutionAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ExecutionAgent failed with error: {e}", name="ExecutionAgent")]}

async def strategy_node(state):
    """Strategy Agent."""
    system_prompt = "You are a Derivatives Strategist. Your job is to propose complex option structures (spreads, condors, etc.) that match the analysis outlook. You can calculate Greeks, find optimal entries via technical support, set ATR-based stops, and run full technical analysis (RSI, Bollinger Bands, ADX, etc.) to inform strategy selection."
    agent = create_agent([propose_option_strategy, calculate_option_greeks, calculate_optimal_entry, calculate_atr_stop_loss, analyze_technicals], system_prompt, agent_type="StrategyAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("StrategyAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="StrategyAgent")]}
    except Exception as e:
        await log_agent_performance("StrategyAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"StrategyAgent failed with error: {e}", name="StrategyAgent")]}

async def scout_node(state):
    """Scout Agent using OpenClaw tools."""
    system_prompt = "You are an Alternative Data Scout. You use OpenClaw stealth technology to scrape web traffic, app rankings, job trends, and Google Trends."
    agent = create_agent([get_web_traffic_metrics, get_app_store_rankings, get_job_market_trends, get_google_trends], system_prompt, agent_type="ScoutAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ScoutAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ScoutAgent")]}
    except Exception as e:
        await log_agent_performance("ScoutAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ScoutAgent failed with error: {e}", name="ScoutAgent")]}

async def technical_node(state):
    """Technical Agent using Vision and programmatic indicators."""
    system_prompt = "You are a Technical Analyst. You have two complementary tools: (1) analyze_technicals for programmatic indicator readings (RSI, MACD, ADX, Bollinger, etc.) and (2) analyze_chart_pattern for visual chart pattern identification via AI vision. Use BOTH tools together: programmatic indicators give you precise numbers, while visual analysis catches patterns the numbers might miss."
    agent = create_agent([analyze_technicals, analyze_chart_pattern], system_prompt, agent_type="TechnicalAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("TechnicalAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="TechnicalAgent")]}
    except Exception as e:
        await log_agent_performance("TechnicalAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"TechnicalAgent failed with error: {e}", name="TechnicalAgent")]}

async def validation_node(state):
    """Validation Agent."""
    system_prompt = "You are a Validation Agent. Your job is to backtest the specific signal type found to ensure it has a positive historical expectancy before we risk capital. Use the validate_signal_robustness tool."
    agent = create_agent([validate_signal_robustness], system_prompt, agent_type="ValidationAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ValidationAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ValidationAgent")]}
    except Exception as e:
        await log_agent_performance("ValidationAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ValidationAgent failed with error: {e}", name="ValidationAgent")]}

async def news_sentiment_node(state):
    """News Sentiment Agent using local BERT."""
    system_prompt = "You are a News Sentiment Specialist. You use a local FinBERT model to score headlines and news snippets. Use the analyze_sentiment_bert tool."
    agent = create_agent([analyze_sentiment_bert], system_prompt, agent_type="NewsSentimentAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("NewsSentimentAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="NewsSentimentAgent")]}
    except Exception as e:
        await log_agent_performance("NewsSentimentAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"NewsSentimentAgent failed with error: {e}", name="NewsSentimentAgent")]}

async def portfolio_optimizer_node(state):
    """Portfolio Optimizer Agent."""
    system_prompt = "You are a Portfolio Optimizer. You use Mean-Variance Optimization to suggest rebalancing weights for the current portfolio to maximize risk-adjusted returns. Use the optimize_portfolio_mean_variance tool."
    agent = create_agent([optimize_portfolio_mean_variance], system_prompt, agent_type="PortfolioOptimizerAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("PortfolioOptimizerAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="PortfolioOptimizerAgent")]}
    except Exception as e:
        await log_agent_performance("PortfolioOptimizerAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"PortfolioOptimizerAgent failed with error: {e}", name="PortfolioOptimizerAgent")]}

async def sector_rotation_node(state):
    """Sector Rotation Agent."""
    system_prompt = "You are a Sector Rotation Strategist. You analyze relative strength between sectors (e.g., Tech vs Energy) to suggest overweight/underweight allocations. Use the analyze_sector_momentum tool."
    agent = create_agent([analyze_sector_momentum], system_prompt, agent_type="SectorRotationAgent")
    
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("SectorRotationAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="SectorRotationAgent")]}
    except Exception as e:
        await log_agent_performance("SectorRotationAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"SectorRotationAgent failed with error: {e}", name="SectorRotationAgent")]}

async def volatility_node(state):
    """Volatility Agent."""
    system_prompt = "You are a Volatility Strategist. You analyze the VIX term structure to gauge market fear and hedging costs. Use the analyze_vix_term_structure tool."
    agent = create_agent([analyze_vix_term_structure], system_prompt, agent_type="VolatilityAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="VolatilityAgent")]}

async def correlation_matrix_node(state):
    """Correlation Matrix Agent."""
    system_prompt = "You are a Correlation Analyst. You calculate rolling correlations between assets and benchmarks to assess decoupling or coupling trends. Use the calculate_rolling_correlation tool."
    agent = create_agent([calculate_rolling_correlation], system_prompt, agent_type="CorrelationMatrixAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="CorrelationMatrixAgent")]}

async def seasonality_node(state):
    """Seasonality Agent."""
    system_prompt = "You are a Seasonality Analyst. You analyze historical monthly returns to identify recurring seasonal patterns (e.g., 'Sell in May'). Use the analyze_seasonality tool."
    agent = create_agent([analyze_seasonality], system_prompt, agent_type="SeasonalityAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SeasonalityAgent")]}

async def fundamental_node(state):
    """Fundamental Analysis Agent."""
    system_prompt = "You are a Fundamental Analyst. You have access to comprehensive financial data: (1) get_comprehensive_fundamentals for valuation, profitability, growth, financial health, dividends, and per-share metrics, (2) get_financial_statements for quarterly income statement, balance sheet, and cash flow trends, (3) get_earnings_history for actual vs estimated EPS and surprise percentages. Analyze all three to build a complete financial picture."
    agent = create_agent([get_comprehensive_fundamentals, get_financial_statements, get_earnings_history], system_prompt, agent_type="FundamentalAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="FundamentalAgent")]}

async def earnings_node(state):
    """Earnings Analysis Agent."""
    system_prompt = "You are an Earnings Analyst. You track upcoming earnings dates and consensus estimates to anticipate volatility events. Use the get_earnings_calendar tool."
    agent = create_agent([get_earnings_calendar], system_prompt, agent_type="EarningsAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="EarningsAgent")]}

async def short_interest_node(state):
    """Short Interest Analysis Agent."""
    system_prompt = "You are a Short Interest Analyst. You track short interest levels and days-to-cover to identify potential short squeezes or bearish sentiment. Use the get_short_interest tool."
    agent = create_agent([get_short_interest], system_prompt, agent_type="ShortInterestAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ShortInterestAgent")]}

async def news_aggregator_node(state):
    """News Aggregator Agent."""
    system_prompt = "You are a News Aggregator. You fetch news from various RSS feeds to get a broader market perspective. Use the fetch_rss_feed tool."
    agent = create_agent([fetch_rss_feed], system_prompt, agent_type="NewsAggregatorAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="NewsAggregatorAgent")]}

async def supply_chain_visualizer_node(state):
    """Supply Chain Visualizer Agent."""
    system_prompt = "You are a Supply Chain Visualizer. You generate Graphviz DOT code to visualize the relationships (suppliers, customers, competitors) for a ticker. Use the visualize_supply_chain tool."
    agent = create_agent([visualize_supply_chain], system_prompt, agent_type="SupplyChainVisualizerAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SupplyChainVisualizerAgent")]}

async def sec_filings_node(state):
    """SEC Filings Agent."""
    system_prompt = "You are an SEC Filings Analyst. You search 10-K and 10-Q filings for specific sections like 'Risk Factors' or 'MD&A' to uncover hidden risks or opportunities. Use the get_sec_filing_section tool."
    agent = create_agent([get_sec_filing_section], system_prompt, agent_type="SECFilingsAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SECFilingsAgent")]}

async def peer_comparison_node(state):
    """Peer Comparison Agent."""
    system_prompt = "You are a Peer Comparison Analyst. You compare companies against their competitors to identify relative value. Use compare_peers for a quick comparison table, and get_comprehensive_fundamentals on the target and key peers for deeper valuation, profitability, and health analysis."
    agent = create_agent([compare_peers, get_comprehensive_fundamentals], system_prompt, agent_type="PeerComparisonAgent")
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="PeerComparisonAgent")]}

async def business_model_node(state):
    """Business Model Analysis Agent."""
    system_prompt = """You are a Business Model Analyst. Your job is to understand HOW a company makes money, not just its financial ratios. You analyze:
1. Revenue model: segments, geographic mix, customer concentration, recurring vs one-time revenue
2. Competitive position: moat type (switching costs, network effects, scale, brand), moat strength, market share trends
3. Total addressable market: TAM size, growth rate, company's market share, expansion vectors
4. Business quality: management alignment, capital allocation, competitive threats

Use get_sec_filing_section to read the company's 10-K (Business section Item 1, Risk Factors Item 1A, MD&A Item 7).
Use get_comprehensive_fundamentals for financial data and sector/industry classification.
Use web_search for recent strategic announcements and competitive landscape.
Use get_competitors to identify key rivals.

Output a structured business model assessment with quality score (0-10)."""
    agent = create_agent(
        [get_sec_filing_section, get_comprehensive_fundamentals, web_search, get_competitors],
        system_prompt, agent_type="BusinessModelAgent"
    )

    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("BusinessModelAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="BusinessModelAgent")]}
    except Exception as e:
        await log_agent_performance("BusinessModelAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"BusinessModelAgent failed with error: {e}", name="BusinessModelAgent")]}


# --- Supervisor Node ---
async def supervisor_node(state):
    """
    The Supervisor (Grok) acts as the Portfolio Manager.
    It decides which specialized agent to call next or if the task is finished.
    """
    messages = state["messages"]
    
    system_prompt = """
    You are a Chief Investment Officer (CIO) and Head of Research.
    Your goal is to orchestrate a "Deep Dive" analysis by coordinating a swarm of specialized agents.
    
    You have the following workers:
    1. ResearchAgent: General web search, finding contracts, news, and filings.
    2. MacroAgent: Interest rates, central banks, geopolitics. Can fetch rates from FRED.
    3. SentimentAgent: Social buzz, consumer trends, alternative data. Can scrape Reddit.
    4. RiskManagerAgent: Checks portfolio exposure and concentration limits.
    5. ExecutionAgent: Executes trades using the execute_trade tool.
    6. StrategyAgent: Proposes multi-leg option strategies and calculates Greeks (Delta, Gamma, Theta).
    7. ScoutAgent: Uses OpenClaw stealth tech to scrape alternative data (web traffic, app ranks, job postings, Google Trends).
    8. TechnicalAgent: Fetches charts and uses Computer Vision to identify patterns (Head & Shoulders, etc).
    9. ValidationAgent: Runs a backtest validation on the signal type to ensure historical robustness.
    10. NewsSentimentAgent: Uses a local FinBERT model to score headlines and text for sentiment.
    11. PortfolioOptimizerAgent: Uses Mean-Variance Optimization to suggest rebalancing weights.
    12. SectorRotationAgent: Analyzes sector ETF momentum to suggest allocations.
    13. VolatilityAgent: Analyzes VIX term structure to gauge market fear.
    14. CorrelationMatrixAgent: Calculates rolling correlations between assets and benchmarks.
    15. SeasonalityAgent: Analyzes historical monthly returns to identify seasonal patterns.
    16. FundamentalAgent: Analyzes key financial ratios (P/E, PEG, Debt/Equity) using Finnhub.
    17. EarningsAgent: Retrieves upcoming earnings dates and estimates using Finnhub.
    18. ShortInterestAgent: Retrieves short interest data and days-to-cover metrics.
    19. NewsAggregatorAgent: Fetches news from RSS feeds.
    20. SupplyChainVisualizerAgent: Generates Graphviz DOT code to visualize supply chain relationships.
    21. SECFilingsAgent: Searches and retrieves specific sections (like Risk Factors) from SEC filings.
    22. PeerComparisonAgent: Generates comparison tables for tickers vs competitors on key metrics.
    23. BusinessModelAgent: Analyzes how a company makes money -- revenue segments, competitive moat, TAM, customer concentration, business quality.

    **Execution Strategy:**
    1. **Plan**: Analyze the user's request. Break it down into specific questions.
    2. **Delegate**: Choose the best agent for the immediate next step. Provide clear, specific instructions.
    3. **Synthesize**: Review agent outputs in the conversation history. If info is missing, call another agent.
    4. **Report**: When you have a complete picture, output FINISH with a comprehensive report.
    
    **Risk & Compliance:**
    - Before recommending a trade (ExecutionAgent), you MUST consult the RiskManagerAgent.
    - Before finalizing a trade idea, you MUST consult the ValidationAgent to check historical win rates.
    - If a signal is BULLISH, ask the StrategyAgent to calculate an optimal entry price and an ATR-based stop-loss.
    
    **Output Format (JSON ONLY):**
    {
        "thought_process": "Briefly explain your reasoning. What do you know? What is missing? Why choose this next agent?",
        "next": "AGENT_NAME" OR "FINISH",
        "instructions": "Instructions for the agent (if next is an agent)",
        "final_answer": "The final report (if next is FINISH)"
    }
    """
    
    # Inject trader profile context
    active_system_prompt = system_prompt
    try:
        from trader_profile import get_trading_style, get_fundamental_weight, get_technical_weight
        style = get_trading_style()
        fund_w = get_fundamental_weight()
        tech_w = get_technical_weight()
        active_system_prompt += f"""

    TRADER PROFILE:
    Trading Style: {style.replace('_', ' ').title()}
    Fundamental Weight: {fund_w:.0%} | Technical Weight: {tech_w:.0%}
    Prioritize {'fundamental analysis (business quality, valuation, earnings)' if fund_w > tech_w else 'technical analysis (indicators, chart patterns, momentum)'} for this trader.
    """
    except ImportError:
        pass

    # Inject SwarmBrief if available
    swarm_brief = state.get("swarm_brief")
    if swarm_brief and swarm_brief.get("swarm_size", 0) > 0:
        try:
            from swarm_synthesizer import format_swarm_brief_for_supervisor
            swarm_text = format_swarm_brief_for_supervisor(swarm_brief)
            active_system_prompt += f"""

    TRADING AGENT SWARM INTELLIGENCE:
    The following is a synthesized brief from the local Trading Agent Swarm — a population
    of {swarm_brief.get('swarm_size', 'N')} autonomous agents with diverse trading
    philosophies that have been continuously debating and sharing results.

    {swarm_text}

    Use this swarm intelligence to inform your delegation decisions. If the swarm has strong
    consensus on a ticker, consider deploying specialists to validate. If the swarm is deeply
    divided, investigate both sides. If the swarm flags an anomaly, prioritize it.
    """
        except ImportError:
            pass

    # We construct a prompt for the supervisor using its configured LLM
    supervisor_llm = get_langchain_llm(agent_type="Supervisor", temperature=0)
    response = await supervisor_llm.ainvoke([
        SystemMessage(content=active_system_prompt),
        *messages
    ])
    
    try:
        decision = json.loads(response.content)
        return {
            "next": decision.get("next", "FINISH"),
            "messages": [AIMessage(content=response.content)]
        }

    except Exception:
        # Fallback if JSON parsing fails
        return {"next": "FINISH", "messages": [AIMessage(content=response.content)]}

async def identify_agents_for_prompt_improvement():
    """Identifies poorly performing agents for prompt improvement."""
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return {}
        
    agent_stats = {}
    try:
        with open(PERFORMANCE_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    name = entry['agent_name']
                    if name not in agent_stats:
                        agent_stats[name] = {'success': 0, 'fail': 0, 'total_time': 0, 'count': 0}
                    
                    if entry['success']:
                        agent_stats[name]['success'] += 1
                    else:
                        agent_stats[name]['fail'] += 1
                    
                    agent_stats[name]['total_time'] += entry['duration']
                    agent_stats[name]['count'] += 1
                except:
                    continue
    except Exception:
        return {}
    
    suggestions = {}
    for name, stats in agent_stats.items():
        fail_rate = stats['fail'] / stats['count'] if stats['count'] > 0 else 0
        avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
        
        if fail_rate > 0.2:
            suggestions[name] = f"High failure rate ({fail_rate:.1%}). Consider reviewing tool inputs or error handling."
        if avg_time > 30:
            suggestions[name] = suggestions.get(name, "") + f" Slow execution ({avg_time:.1f}s). Check tool latency."
            
    return suggestions

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("ResearchAgent", research_node)
workflow.add_node("MacroAgent", macro_node)
workflow.add_node("SentimentAgent", sentiment_node)
workflow.add_node("RiskManagerAgent", risk_node)
workflow.add_node("ExecutionAgent", execution_node)
workflow.add_node("StrategyAgent", strategy_node)
workflow.add_node("ScoutAgent", scout_node)
workflow.add_node("TechnicalAgent", technical_node)
workflow.add_node("ValidationAgent", validation_node)
workflow.add_node("NewsSentimentAgent", news_sentiment_node)
workflow.add_node("PortfolioOptimizerAgent", portfolio_optimizer_node)
workflow.add_node("SectorRotationAgent", sector_rotation_node)
workflow.add_node("VolatilityAgent", volatility_node)
workflow.add_node("CorrelationMatrixAgent", correlation_matrix_node)
workflow.add_node("SeasonalityAgent", seasonality_node)
workflow.add_node("FundamentalAgent", fundamental_node)
workflow.add_node("EarningsAgent", earnings_node)
workflow.add_node("ShortInterestAgent", short_interest_node)
workflow.add_node("NewsAggregatorAgent", news_aggregator_node)
workflow.add_node("SupplyChainVisualizerAgent", supply_chain_visualizer_node)
workflow.add_node("SECFilingsAgent", sec_filings_node)
workflow.add_node("PeerComparisonAgent", peer_comparison_node)
workflow.add_node("BusinessModelAgent", business_model_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "ResearchAgent": "ResearchAgent",
        "MacroAgent": "MacroAgent",
        "SentimentAgent": "SentimentAgent",
        "RiskManagerAgent": "RiskManagerAgent",
        "ExecutionAgent": "ExecutionAgent",
        "StrategyAgent": "StrategyAgent",
        "ScoutAgent": "ScoutAgent",
        "TechnicalAgent": "TechnicalAgent",
        "ValidationAgent": "ValidationAgent",
        "NewsSentimentAgent": "NewsSentimentAgent",
        "PortfolioOptimizerAgent": "PortfolioOptimizerAgent",
        "SectorRotationAgent": "SectorRotationAgent",
        "VolatilityAgent": "VolatilityAgent",
        "CorrelationMatrixAgent": "CorrelationMatrixAgent",
        "SeasonalityAgent": "SeasonalityAgent",
        "FundamentalAgent": "FundamentalAgent",
        "EarningsAgent": "EarningsAgent",
        "ShortInterestAgent": "ShortInterestAgent",
        "NewsAggregatorAgent": "NewsAggregatorAgent",
        "SupplyChainVisualizerAgent": "SupplyChainVisualizerAgent",
        "SECFilingsAgent": "SECFilingsAgent",
        "PeerComparisonAgent": "PeerComparisonAgent",
        "BusinessModelAgent": "BusinessModelAgent",
        "FINISH": END
    }
)

workflow.add_edge("ResearchAgent", "Supervisor")
workflow.add_edge("MacroAgent", "Supervisor")
workflow.add_edge("SentimentAgent", "Supervisor")
workflow.add_edge("RiskManagerAgent", "Supervisor")
workflow.add_edge("ExecutionAgent", "Supervisor")
workflow.add_edge("StrategyAgent", "Supervisor")
workflow.add_edge("ScoutAgent", "Supervisor")
workflow.add_edge("TechnicalAgent", "Supervisor")
workflow.add_edge("ValidationAgent", "Supervisor")
workflow.add_edge("NewsSentimentAgent", "Supervisor")
workflow.add_edge("PortfolioOptimizerAgent", "Supervisor")
workflow.add_edge("SectorRotationAgent", "Supervisor")
workflow.add_edge("VolatilityAgent", "Supervisor")
workflow.add_edge("CorrelationMatrixAgent", "Supervisor")
workflow.add_edge("SeasonalityAgent", "Supervisor")
workflow.add_edge("FundamentalAgent", "Supervisor")
workflow.add_edge("EarningsAgent", "Supervisor")
workflow.add_edge("ShortInterestAgent", "Supervisor")
workflow.add_edge("NewsAggregatorAgent", "Supervisor")
workflow.add_edge("SupplyChainVisualizerAgent", "Supervisor")
workflow.add_edge("SECFilingsAgent", "Supervisor")
workflow.add_edge("PeerComparisonAgent", "Supervisor")
workflow.add_edge("BusinessModelAgent", "Supervisor")

app = workflow.compile()

# --- Public Interface ---
async def run_research_task(query: str):
    """
    Entry point to run a research task.
    Example: run_research_task("Find the latest supplier contract for Company X")
    """
    print(f"[*] Starting Agentic Research: {query}")
    
    # Inject swarm brief if available from the ContinuousMonitorAgent
    swarm_brief = None
    try:
        from agent_continuous_monitor import get_monitor
        monitor = get_monitor()
        swarm_brief = getattr(monitor, "_latest_swarm_brief", None)
    except Exception:
        pass

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next": "Supervisor",
        "swarm_brief": swarm_brief or {},
    }
    
    execution_log = []
    
    async for event in app.astream(initial_state):
        for key, value in event.items():
            if key == "Supervisor":
                # Check if supervisor provided a final answer
                if "messages" in value:
                    try:
                        content = json.loads(value["messages"][0].content)
                        
                        # Log the Brain's thought process
                        if "thought_process" in content:
                            print(f"🧠 [CIO Brain]: {content['thought_process']}")
                            execution_log.append(f"🧠 **CIO Thought**: {content['thought_process']}")
                            
                        if "final_answer" in content:
                            execution_log.append(f"🏁 **Final Report**: {content['final_answer']}")
                    except:
                        pass
            elif key in ["ResearchAgent", "MacroAgent", "SentimentAgent", "RiskManagerAgent", "ExecutionAgent", "StrategyAgent", "ScoutAgent", "TechnicalAgent", "ValidationAgent", "NewsSentimentAgent", "PortfolioOptimizerAgent", "SectorRotationAgent", "VolatilityAgent", "CorrelationMatrixAgent", "SeasonalityAgent", "FundamentalAgent", "EarningsAgent", "ShortInterestAgent", "NewsAggregatorAgent", "SupplyChainVisualizerAgent", "SECFilingsAgent", "PeerComparisonAgent", "BusinessModelAgent"]:
                # Log agent activity
                msg = value["messages"][0].content
                # Ensure msg is a string before slicing
                if isinstance(msg, str):
                    print(f"    [{key}]: {msg[:100]}...")
                else:
                    print(f"    [{key}]: {str(msg)[:100]}...")
                
                execution_log.append(f"🤖 **{key}**: {msg}")
                
    return "\n\n".join(execution_log)

# Test block
if __name__ == "__main__":
    import asyncio
    # Example test
    # asyncio.run(run_research_task("Who are the top 3 suppliers for Walmart mentioned in 2024 news?"))