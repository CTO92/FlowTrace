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
You are a Chief Investment Officer (CIO) and quantitative analyst.
Given news about a specific ticker (Source), analyze the impact on the ticker itself and its related assets (Competitors, Suppliers, Peers).

Your task:
1. Classify the event type (e.g., Contract, Earnings, Merger, Regulatory).
2. Estimate the impact on:
   - The Source Ticker (Direct Impact)
   - Related Assets (Sympathetic or Second-Order Impact)
3. Calculate a 'Unified Multi-Factor Correlation Strength' (0.0 to 1.0) based on:
   - Price Action (Beta, Cointegration)
   - Options Flow (IV, OI)
   - Fundamentals (Revenue dependency, Competitive advantage)
   - Macro/Sector tailwinds
   - Sentiment (News/Social)
4. Predict the expected % move for the targets over the next 1-5 days.

WEIGHTING INSTRUCTIONS:
- If event_type is 'Earnings' for a Competitor, specifically analyze for "Sympathetic Moves" (e.g., if Competitor A beats, does Ticker B rise in sympathy or fall due to market share loss?).
- If event_type is 'Macro' (e.g., Rates, Inflation) or 'Regulatory', prioritize findings from the MacroAgent (rates, geopolitics).
- If event_type is 'Earnings', 'Product Launch', or 'Scandal', prioritize findings from the SentimentAgent (social buzz, retail sentiment).
- If event_type is 'Contract', 'Merger', or 'Partnership', prioritize the ResearchAgent (deal specifics) and Knowledge Graph connections.

Output STRICT JSON format only. No markdown, no conversational text.
Structure:
{
  "analysis_summary": "Brief summary of the event and macro implication",
  "targets": [
    {
      "ticker": "TICKER",
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

async def analyze_impact(source_ticker, related_assets, news_item, agent_data=None, market_data=None):
    """
    Sends news and related asset context to Grok for analysis.
    """
    if not client:
        logger.error("Grok client not initialized. Check API Key.")
        return None

    # Extract news details
    title = getattr(news_item, "title", "No Title")
    description = getattr(news_item, "description", "No Description")
    published_utc = getattr(news_item, "published_utc", "Unknown Time")

    # Format partner context for the LLM
    assets_context = []
    for p in related_assets:
        assets_context.append(f"- {p['ticker']} ({p['name']}): Relationship={p['relationship']}")
    
    assets_str = "\n".join(assets_context)

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

    RELATED ASSETS (Competitors/Suppliers/Peers):
    {assets_str}

    {market_section}

    {agent_section}

    Analyze the impact of this news on the source ticker and the related assets listed above.
    """

    try:
        logger.info(f"Sending analysis request to Grok for {source_ticker}...")
        
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