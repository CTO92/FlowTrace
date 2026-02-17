import os
import json
import operator
from typing import Annotated, Sequence, TypedDict, List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import create_openai_tools_agent, AgentExecutor
from dotenv import load_dotenv

from agent_tools import web_search, scrape_web_page
from agent_tools_advanced import get_macro_rates, get_reddit_sentiment, get_portfolio_exposure, execute_trade, propose_option_strategy, calculate_option_greeks
from agent_tools_scout import get_web_traffic_metrics, get_app_store_rankings, get_job_market_trends, get_google_trends
from agent_tools_technical import analyze_chart_pattern

load_dotenv()

# --- Configuration ---
XAI_API_KEY = os.getenv("XAI_API_KEY")

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
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# --- Specialized Agent Nodes ---

async def research_node(state):
    """General Web Research Agent."""
    system_prompt = "You are a Research Agent. You can search the web and scrape pages to find missing financial data, supplier relationships, or news details."
    agent = create_agent([web_search, scrape_web_page], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ResearchAgent")]}

async def macro_node(state):
    """Macro-Economic Agent using FRED and web search."""
    system_prompt = "You are a Macro Strategy Agent. You have a tool to fetch current interest rates from FRED. Focus on interest rates, inflation, central bank policy, and geopolitical events. Use your tools to answer questions."
    agent = create_agent([web_search, get_macro_rates], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="MacroAgent")]}

async def sentiment_node(state):
    """Sentiment & Alternative Data Agent for Reddit and web scraping."""
    system_prompt = "You are a Sentiment Analyst. You have a tool to scrape Reddit for ticker sentiment. You investigate social media buzz, consumer sentiment, and alternative data trends."
    agent = create_agent([web_search, scrape_web_page, get_reddit_sentiment], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="SentimentAgent")]}

async def risk_node(state):
    """Risk Management Agent."""
    system_prompt = "You are a Risk Manager. Your job is to check portfolio exposure and assess concentration risk. Use the get_portfolio_exposure tool to check current holdings before we commit to a trade."
    agent = create_agent([get_portfolio_exposure], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="RiskManagerAgent")]}

async def execution_node(state):
    """Execution Agent."""
    system_prompt = "You are an Execution Trader. Your job is to execute trades based on instructions. Use the execute_trade tool."
    agent = create_agent([execute_trade], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ExecutionAgent")]}

async def strategy_node(state):
    """Strategy Agent."""
    system_prompt = "You are a Derivatives Strategist. Your job is to propose complex option structures (spreads, condors, etc.) that match the analysis outlook. You can also calculate the Greeks for specific options using calculate_option_greeks to assess risk."
    agent = create_agent([propose_option_strategy, calculate_option_greeks], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="StrategyAgent")]}

async def scout_node(state):
    """Scout Agent using OpenClaw tools."""
    system_prompt = "You are an Alternative Data Scout. You use OpenClaw stealth technology to scrape web traffic, app rankings, job trends, and Google Trends."
    agent = create_agent([get_web_traffic_metrics, get_app_store_rankings, get_job_market_trends, get_google_trends], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ScoutAgent")]}

async def technical_node(state):
    """Technical Agent using Vision."""
    system_prompt = "You are a Technical Analyst. You use a vision model to look at charts and identify patterns. Use the analyze_chart_pattern tool."
    agent = create_agent([analyze_chart_pattern], system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="TechnicalAgent")]}

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
    
    **Execution Strategy:**
    1. **Plan**: Analyze the user's request. Break it down into specific questions.
    2. **Delegate**: Choose the best agent for the immediate next step. Provide clear, specific instructions.
    3. **Synthesize**: Review agent outputs in the conversation history. If info is missing, call another agent.
    4. **Report**: When you have a complete picture, output FINISH with a comprehensive report.
    
    **Risk & Compliance:**
    - Before recommending a trade (ExecutionAgent), you MUST consult the RiskManagerAgent.
    
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
            elif key in ["ResearchAgent", "MacroAgent", "SentimentAgent", "RiskManagerAgent", "ExecutionAgent", "StrategyAgent", "ScoutAgent", "TechnicalAgent"]:
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