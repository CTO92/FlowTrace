import os
import time
import base64
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openclaw_wrapper import OpenClawSession
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")

# Initialize Vision Model
# Note: Assuming 'grok-vision-beta' or similar is available via the xAI endpoint.
# If not, this mimics the OpenAI Vision API structure which xAI is compatible with.
vision_llm = ChatOpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    model="grok-vision-beta", 
    temperature=0.1
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@tool("analyze_chart_pattern")
async def analyze_chart_pattern(ticker: str):
    """
    Fetches a price chart for the given ticker using OpenClaw, takes a screenshot,
    and uses a Vision AI model to identify technical patterns (e.g., Head & Shoulders, Breakouts).
    """
    # Use a public chart URL (TradingView widget or similar accessible page)
    # For demo purposes, we use a generic chart view
    url = f"https://www.tradingview.com/chart/?symbol={ticker}"
    
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    
    try:
        async with OpenClawSession(headless=True) as session:
            page = await session.new_stealth_page()
            
            # Navigate
            await page.goto(url, timeout=20000, wait_until="domcontentloaded")
            
            # Wait for chart to render (simulated wait)
            await page.wait_for_timeout(5000)
            
            # Take Screenshot
            timestamp = int(time.time())
            filename = f"chart_{ticker}_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)
            await page.screenshot(path=filepath)
            
            # Encode for Vision Model
            base64_image = encode_image(filepath)
            
            # Analyze with Vision Model
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": f"You are a Technical Analyst. Look at this chart for {ticker}. Identify the trend, support/resistance levels, and any visible chart patterns (e.g., Double Bottom, Head and Shoulders). Be concise."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ]
            )
            
            response = await vision_llm.ainvoke([msg])
            analysis = response.content
            
            return f"Technical Analysis for {ticker}:\n{analysis}\n\n[SCREENSHOT: {filepath}]"
            
    except Exception as e:
        return f"Error analyzing chart for {ticker}: {str(e)}"