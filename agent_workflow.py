import os
import json
import time
import operator
from typing import Annotated, Sequence, TypedDict, List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_tools_agent, AgentExecutor
from dotenv import load_dotenv

from agent_tools import web_search, scrape_web_page, get_competitors, get_insider_trades, get_analyst_ratings, get_fundamental_ratios, get_earnings_calendar, get_short_interest, fetch_rss_feed, visualize_supply_chain, get_sec_filing_section, compare_peers
from agent_tools_advanced import get_macro_rates, get_reddit_sentiment, get_portfolio_exposure, execute_trade, propose_option_strategy, calculate_option_greeks, calculate_optimal_entry, validate_signal_robustness, calculate_atr_stop_loss, analyze_sentiment_bert, optimize_portfolio_mean_variance, analyze_sector_momentum, analyze_vix_term_structure, calculate_rolling_correlation, analyze_seasonality
from agent_tools_scout import get_web_traffic_metrics, get_app_store_rankings, get_job_market_trends, get_google_trends
from agent_tools_technical import analyze_chart_pattern

load_dotenv()

# --- Configuration ---
XAI_API_KEY = os.getenv("XAI_API_KEY")
PERFORMANCE_LOG_FILE = os.path.join(os.path.dirname(__file__), "agent_performance.log")

# Initialize Grok (xAI) as the LLM
llm = ChatOpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    model="grok-beta", # or grok-2
    temperature=0
)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# --- Agent Constructors ---

def create_agent(tools, system_prompt):
    """Generic helper to create an agent with specific tools and persona."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
 ("user", "Please adhere to the system prompt and respond in a JSON format."),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
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
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ResearchAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ResearchAgent")]}
    except Exception as e:
        await log_agent_performance("ResearchAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ResearchAgent failed with error: {e}", name="ResearchAgent")]}

    """General Web Research Agent."""
    system_prompt = "You are a Research Agent. You can search the web, scrape pages, look up competitors, and check analyst ratings to find missing financial data, supplier relationships, or news details."
    agent = create_agent([web_search, scrape_web_page, get_competitors, get_analyst_ratings], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ResearchAgent")]}

async def macro_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("MacroAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="MacroAgent")]}
    except Exception as e:
        await log_agent_performance("MacroAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"MacroAgent failed with error: {e}", name="MacroAgent")]}
    """Macro-Economic Agent using FRED and web search."""
    system_prompt = "You are a Macro Strategy Agent. You have a tool to fetch current interest rates from FRED. Focus on interest rates, inflation, central bank policy, and geopolitical events. Use your tools to answer questions."
    agent = create_agent([web_search, get_macro_rates], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="MacroAgent")]}


async def sentiment_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("SentimentAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="SentimentAgent")]}
    except Exception as e:
        await log_agent_performance("SentimentAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"SentimentAgent failed with error: {e}", name="SentimentAgent")]}
    """Sentiment & Alternative Data Agent for Reddit and web scraping."""
    system_prompt = "You are a Sentiment Analyst. You have a tool to scrape Reddit for ticker sentiment. You investigate social media buzz, consumer sentiment, and alternative data trends. You also check insider trading activity to gauge management confidence."
    agent = create_agent([web_search, scrape_web_page, get_reddit_sentiment, get_insider_trades], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SentimentAgent")]}


async def risk_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("RiskManagerAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="RiskManagerAgent")]}
    except Exception as e:
        await log_agent_performance("RiskManagerAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"RiskManagerAgent failed with error: {e}", name="RiskManagerAgent")]}
    """Risk Management Agent."""
    system_prompt = "You are a Risk Manager. Your job is to check portfolio exposure and assess concentration risk. Use the get_portfolio_exposure tool to check current holdings before we commit to a trade."
    agent = create_agent([get_portfolio_exposure], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="RiskManagerAgent")]}


async def execution_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ExecutionAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ExecutionAgent")]}
    except Exception as e:
        await log_agent_performance("ExecutionAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ExecutionAgent failed with error: {e}", name="ExecutionAgent")]}
    """Execution Agent."""
    system_prompt = "You are an Execution Trader. Your job is to execute trades based on instructions. Use the execute_trade tool."
    agent = create_agent([execute_trade], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ExecutionAgent")]}


async def strategy_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("StrategyAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="StrategyAgent")]}
    except Exception as e:
        await log_agent_performance("StrategyAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"StrategyAgent failed with error: {e}", name="StrategyAgent")]}
    """Strategy Agent."""
    system_prompt = "You are a Derivatives Strategist. Your job is to propose complex option structures (spreads, condors, etc.) that match the analysis outlook. You can also calculate the Greeks for specific options using calculate_option_greeks to assess risk. You can also calculate optimal entry prices based on technical support and suggest stop-loss levels using ATR."
    agent = create_agent([propose_option_strategy, calculate_option_greeks, calculate_optimal_entry, calculate_atr_stop_loss], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="StrategyAgent")]}


async def scout_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ScoutAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ScoutAgent")]}
    except Exception as e:
        await log_agent_performance("ScoutAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ScoutAgent failed with error: {e}", name="ScoutAgent")]}
    """Scout Agent using OpenClaw tools."""
    system_prompt = "You are an Alternative Data Scout. You use OpenClaw stealth technology to scrape web traffic, app rankings, job trends, and Google Trends."
    agent = create_agent([get_web_traffic_metrics, get_app_store_rankings, get_job_market_trends, get_google_trends], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ScoutAgent")]}


async def technical_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("TechnicalAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="TechnicalAgent")]}
    except Exception as e:
        await log_agent_performance("TechnicalAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"TechnicalAgent failed with error: {e}", name="TechnicalAgent")]}
    """Technical Agent using Vision."""
    system_prompt = "You are a Technical Analyst. You use a vision model to look at charts and identify patterns. Use the analyze_chart_pattern tool."
    agent = create_agent([analyze_chart_pattern], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="TechnicalAgent")]}


async def validation_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("ValidationAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="ValidationAgent")]}
    except Exception as e:
        await log_agent_performance("ValidationAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"ValidationAgent failed with error: {e}", name="ValidationAgent")]}
    """Validation Agent."""
    system_prompt = "You are a Validation Agent. Your job is to backtest the specific signal type found to ensure it has a positive historical expectancy before we risk capital. Use the validate_signal_robustness tool."
    agent = create_agent([validate_signal_robustness], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ValidationAgent")]}


async def news_sentiment_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("NewsSentimentAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="NewsSentimentAgent")]}
    except Exception as e:
        await log_agent_performance("NewsSentimentAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"NewsSentimentAgent failed with error: {e}", name="NewsSentimentAgent")]}
    """News Sentiment Agent using local BERT."""
    system_prompt = "You are a News Sentiment Specialist. You use a local FinBERT model to score headlines and news snippets. Use the analyze_sentiment_bert tool."
    agent = create_agent([analyze_sentiment_bert], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="NewsSentimentAgent")]}


async def portfolio_optimizer_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("PortfolioOptimizerAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="PortfolioOptimizerAgent")]}
    except Exception as e:
        await log_agent_performance("PortfolioOptimizerAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"PortfolioOptimizerAgent failed with error: {e}", name="PortfolioOptimizerAgent")]}
    """Portfolio Optimizer Agent."""
    system_prompt = "You are a Portfolio Optimizer. You use Mean-Variance Optimization to suggest rebalancing weights for the current portfolio to maximize risk-adjusted returns. Use the optimize_portfolio_mean_variance tool."
    agent = create_agent([optimize_portfolio_mean_variance], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="PortfolioOptimizerAgent")]}


async def sector_rotation_node(state):
    start_time = time.time()
    try:
        result = await agent.ainvoke({"messages": state["messages"]})
        await log_agent_performance("SectorRotationAgent", start_time, True)
        return {"messages": [AIMessage(content=result["output"], name="SectorRotationAgent")]}
    except Exception as e:
        await log_agent_performance("SectorRotationAgent", start_time, False, str(e))
        return {"messages": [AIMessage(content=f"SectorRotationAgent failed with error: {e}", name="SectorRotationAgent")]}
    """Sector Rotation Agent."""
    system_prompt = "You are a Sector Rotation Strategist. You analyze relative strength between sectors (e.g., Tech vs Energy) to suggest overweight/underweight allocations. Use the analyze_sector_momentum tool."
    agent = create_agent([analyze_sector_momentum], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SectorRotationAgent")]}

async def volatility_node(state):
    """Volatility Agent."""
    system_prompt = "You are a Volatility Strategist. You analyze the VIX term structure to gauge market fear and hedging costs. Use the analyze_vix_term_structure tool."
    agent = create_agent([analyze_vix_term_structure], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="VolatilityAgent")]}

async def correlation_matrix_node(state):
    """Correlation Matrix Agent."""
    system_prompt = "You are a Correlation Analyst. You calculate rolling correlations between assets and benchmarks to assess decoupling or coupling trends. Use the calculate_rolling_correlation tool."
    agent = create_agent([calculate_rolling_correlation], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="CorrelationMatrixAgent")]}

async def seasonality_node(state):
    """Seasonality Agent."""
    system_prompt = "You are a Seasonality Analyst. You analyze historical monthly returns to identify recurring seasonal patterns (e.g., 'Sell in May'). Use the analyze_seasonality tool."
    agent = create_agent([analyze_seasonality], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SeasonalityAgent")]}

async def fundamental_node(state):
    """Fundamental Analysis Agent."""
    system_prompt = "You are a Fundamental Analyst. You analyze company financial health using key ratios like P/E, PEG, ROE, and Debt-to-Equity. Use the get_fundamental_ratios tool."
    agent = create_agent([get_fundamental_ratios], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="FundamentalAgent")]}

async def earnings_node(state):
    """Earnings Analysis Agent."""
    system_prompt = "You are an Earnings Analyst. You track upcoming earnings dates and consensus estimates to anticipate volatility events. Use the get_earnings_calendar tool."
    agent = create_agent([get_earnings_calendar], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="EarningsAgent")]}

async def short_interest_node(state):
    """Short Interest Analysis Agent."""
    system_prompt = "You are a Short Interest Analyst. You track short interest levels and days-to-cover to identify potential short squeezes or bearish sentiment. Use the get_short_interest tool."
    agent = create_agent([get_short_interest], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ShortInterestAgent")]}

async def news_aggregator_node(state):
    """News Aggregator Agent."""
    system_prompt = "You are a News Aggregator. You fetch news from various RSS feeds to get a broader market perspective. Use the fetch_rss_feed tool."
    agent = create_agent([fetch_rss_feed], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="NewsAggregatorAgent")]}

async def supply_chain_visualizer_node(state):
    """Supply Chain Visualizer Agent."""
    system_prompt = "You are a Supply Chain Visualizer. You generate Graphviz DOT code to visualize the relationships (suppliers, customers, competitors) for a ticker. Use the visualize_supply_chain tool."
    agent = create_agent([visualize_supply_chain], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SupplyChainVisualizerAgent")]}

async def sec_filings_node(state):
    """SEC Filings Agent."""
    system_prompt = "You are an SEC Filings Analyst. You search 10-K and 10-Q filings for specific sections like 'Risk Factors' or 'MD&A' to uncover hidden risks or opportunities. Use the get_sec_filing_section tool."
    agent = create_agent([get_sec_filing_section], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SECFilingsAgent")]}

async def peer_comparison_node(state):
    """Peer Comparison Agent."""
    system_prompt = "You are a Peer Comparison Analyst. You compare companies against their competitors to identify relative value. Use the compare_peers tool."
    agent = create_agent([compare_peers], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="PeerComparisonAgent")]}

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
    
    # We construct a prompt for the supervisor
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
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

app = workflow.compile()

# --- Public Interface ---
async def run_research_task(query: str):
    """
    Entry point to run a research task.
    Example: run_research_task("Find the latest supplier contract for Company X")
    """
    print(f"[*] Starting Agentic Research: {query}")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next": "Supervisor"
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
            elif key in ["ResearchAgent", "MacroAgent", "SentimentAgent", "RiskManagerAgent", "ExecutionAgent", "StrategyAgent", "ScoutAgent", "TechnicalAgent", "ValidationAgent", "NewsSentimentAgent", "PortfolioOptimizerAgent", "SectorRotationAgent", "VolatilityAgent", "CorrelationMatrixAgent", "SeasonalityAgent", "FundamentalAgent", "EarningsAgent", "ShortInterestAgent", "NewsAggregatorAgent", "SupplyChainVisualizerAgent", "SECFilingsAgent", "PeerComparisonAgent"]:
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