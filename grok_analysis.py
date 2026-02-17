import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XAI_API_KEY = os.getenv("XAI_API_KEY")

# Initialize Client (using OpenAI-compatible SDK for xAI)
client = None
if XAI_API_KEY:
    client = AsyncOpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )
else:
    logger.warning("XAI_API_KEY not found. Grok analysis will be skipped.")

SYSTEM_PROMPT = """
You are a quantitative hedge fund analyst specializing in supply-chain event studies.
Given news about a Fortune 500 company (Hub), analyze the impact on its connected small-cap partners (Spokes).

Your task:
1. Classify the event type (e.g., Contract, Earnings, Merger, Regulatory).
2. Estimate the impact on the specific small-cap partners provided.
3. Calculate a 'Unified Multi-Factor Correlation Strength' (0.0 to 1.0) based on:
   - Price Action (Beta, Cointegration)
   - Options Flow (IV, OI)
   - Fundamentals (Revenue dependency)
   - Macro/Sector tailwinds
   - Sentiment (News/Social)
4. Predict the expected % move for the small-cap stock over the next 1-5 days.

WEIGHTING INSTRUCTIONS:
- If event_type is 'Macro' (e.g., Rates, Inflation) or 'Regulatory', prioritize findings from the MacroAgent (rates, geopolitics).
- If event_type is 'Earnings', 'Product Launch', or 'Scandal', prioritize findings from the SentimentAgent (social buzz, retail sentiment).
- If event_type is 'Contract', 'Merger', or 'Partnership', prioritize the ResearchAgent (deal specifics) and Knowledge Graph connections.

Output STRICT JSON format only. No markdown, no conversational text.
Structure:
{
  "analysis_summary": "Brief summary of the event and macro implication",
  "targets": [
    {
      "ticker": "SMALL_CAP_TICKER",
      "event_type": "Contract/Earnings/etc",
      "expected_move_pct": 5.5,
      "confidence": 85,
      "unified_correlation_score": 0.88,
      "reasoning": "Detailed reasoning...",
      "risk_factors": ["List", "of", "risks"]
    }
  ]
}
"""

async def analyze_impact(source_ticker, partners, news_item, agent_data=None, market_data=None):
    """
    Sends news and partner context to Grok for analysis.
    """
    if not client:
        logger.error("Grok client not initialized. Check API Key.")
        return None

    # Extract news details
    title = getattr(news_item, "title", "No Title")
    description = getattr(news_item, "description", "No Description")
    published_utc = getattr(news_item, "published_utc", "Unknown Time")

    # Format partner context for the LLM
    partners_context = []
    for p in partners:
        partners_context.append(f"- {p['ticker']} ({p['name']}): Relationship={p['relationship']}")
    
    partners_str = "\n".join(partners_context)

    market_section = ""
    if market_data:
        market_section = "REAL-TIME MARKET DATA:\n"
        for t, d in market_data.items():
            market_section += f"- {t}: Price=${d.get('price')}, Change={d.get('change_pct')}%, Vol={d.get('volume')}\n"

    agent_section = ""
    if agent_data:
        agent_section = f"""
    AGENT RESEARCH FINDINGS:
    {agent_data}
    """

    user_content = f"""
    NEWS ALERT:
    Source: {source_ticker}
    Title: {title}
    Description: {description}
    Time: {published_utc}

    POTENTIAL AFFECTED PARTNERS:
    {partners_str}

    {market_section}

    {agent_section}

    Analyze the impact of this news on the partners listed above.
    """

    try:
        logger.info(f"Sending analysis request to Grok for {source_ticker} -> {len(partners)} partners...")
        
        response = await client.chat.completions.create(
            model="grok-beta", # Adjust model name as needed (e.g., grok-2)
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2, # Low temperature for analytical precision
            response_format={"type": "json_object"} 
        )

        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from Grok response.")
            logger.debug(f"Raw response: {content}")
            return None

    except Exception as e:
        logger.error(f"Error during Grok analysis: {e}")
        return None

async def generate_briefing(signals):
    """
    Generates a morning briefing based on high-conviction signals.
    signals: List of dictionaries containing signal data.
    """
    if not client:
        return "Grok client not initialized."
    
    if not signals:
        return "No significant market signals found for a briefing."
    
    # Prepare context
    signals_text = ""
    for s in signals[:10]: # Limit to top 10 to fit context/focus
        signals_text += f"- {s['source_ticker']} -> {s['target_ticker']}: {s['summary']} (Conf: {s['confidence']}%)\n"
    
    prompt = f"""
    You are a Portfolio Manager giving a morning briefing to your trading desk.
    Summarize the following key market signals into a concise, actionable, audio-script style briefing.
    Group them by themes (e.g., "Tech Supply Chain", "Macro Headwinds") if possible.
    
    SIGNALS:
    {signals_text}
    """
    
    try:
        response = await client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are a senior financial reporter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating briefing: {e}"