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

# --- Research Agent Node ---
def create_research_agent():
    """Creates the Research Agent that can search and scrape."""
    tools = [web_search, scrape_web_page]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Research Agent. You can search the web and scrape pages to find missing financial data, supplier relationships, or news details."),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

async def research_node(state):
    """Node function for the Research Agent."""
    research_agent = create_research_agent()
    result = await research_agent.ainvoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name="ResearchAgent")]}

# --- Supervisor Node ---
async def supervisor_node(state):
    """
    The Supervisor (Grok) decides which agent to call next or if the task is finished.
    """
    messages = state["messages"]
    
    system_prompt = """
    You are a Supervisor managing a research workflow.
    You have the following workers: [ResearchAgent].
    
    Your goal is to answer the user's request by delegating tasks to workers.
    
    1. If you need more information, output: {"next": "ResearchAgent", "instructions": "..."}
    2. If you have the answer or the research is complete, output: {"next": "FINISH", "final_answer": "..."}
    
    Output STRICT JSON only.
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

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "ResearchAgent": "ResearchAgent",
        "FINISH": END
    }
)

workflow.add_edge("ResearchAgent", "Supervisor")

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
    
    final_output = None
    
    async for event in app.astream(initial_state):
        for key, value in event.items():
            if key == "Supervisor":
                # Check if supervisor provided a final answer
                if "messages" in value:
                    try:
                        content = json.loads(value["messages"][0].content)
                        if "final_answer" in content:
                            final_output = content["final_answer"]
                    except:
                        pass
            elif key == "ResearchAgent":
                # Log agent activity
                msg = value["messages"][0].content
                print(f"    [ResearchAgent]: {msg[:100]}...")
                final_output = msg
                
    return final_output

# Test block
if __name__ == "__main__":
    import asyncio
    # Example test
    # asyncio.run(run_research_task("Who are the top 3 suppliers for Walmart mentioned in 2024 news?"))