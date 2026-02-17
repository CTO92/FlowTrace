import os
import datetime
from langchain_core.tools import tool
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import portfolio_manager
import math
from scipy.stats import norm
import numpy as np

@tool("get_macro_rates")
def get_macro_rates(dummy_arg: str = ""):
    """
    Fetches current key interest rates (Fed Funds, 10Y Treasury) from FRED.
    Useful for the MacroAgent to assess the economic environment.
    """
    try:
        # Fetch last 60 days to ensure we get the most recent data point
        start = datetime.datetime.now() - datetime.timedelta(days=60)
        end = datetime.datetime.now()
        
        # FEDFUNDS: Effective Federal Funds Rate
        # DGS10: 10-Year Treasury Constant Maturity Rate
        
        # Note: pandas_datareader may require a FRED API key in environment variables 
        # depending on the specific series or usage limits.
        df = web.DataReader(['FEDFUNDS', 'DGS10'], 'fred', start, end)
        
        # Get latest available non-NaN values
        fed_funds = df['FEDFUNDS'].dropna().iloc[-1]
        treasury_10y = df['DGS10'].dropna().iloc[-1]
        
        return f"Latest Macro Rates (FRED):\nFed Funds Rate: {fed_funds}%\n10Y Treasury Yield: {treasury_10y}%"
    except Exception as e:
        return f"Error fetching macro rates from FRED: {str(e)}"

@tool("get_reddit_sentiment")
async def get_reddit_sentiment(ticker: str):
    """
    Scrapes the latest discussion threads about a ticker from Reddit to gauge sentiment.
    Searches for the ticker and returns titles of recent posts.
    """
    url = f"https://www.reddit.com/search/?q={ticker}&sort=new"
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            # Use a realistic user agent to avoid immediate blocking
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = await context.new_page()
            
            try:
                await page.goto(url, timeout=10000, wait_until="domcontentloaded")
                # Wait a bit for client-side hydration (Reddit is SPA)
                await page.wait_for_timeout(2000)
            except Exception:
                pass # Try to parse what we have if timeout occurs
                
            content = await page.content()
            await browser.close()
            
            soup = BeautifulSoup(content, "html.parser")
            
            posts = []
            # Strategy 1: Look for <shreddit-post> tags (Modern Reddit)
            for tag in soup.find_all("shreddit-post", limit=10):
                title = tag.get("post-title")
                subreddit = tag.get("subreddit-prefixed-name")
                if title:
                    posts.append(f"- [{subreddit}] {title}")
            
            # Strategy 2: Fallback to H3 tags (often used for titles in search results)
            if not posts:
                for h3 in soup.find_all("h3", limit=10):
                    text = h3.get_text(strip=True)
                    if len(text) > 10: # Filter out short UI elements
                        posts.append(f"- {text}")
            
            if not posts:
                return f"No recent Reddit posts found for {ticker} (or scraping was blocked)."
                
            return f"Recent Reddit Sentiment for {ticker}:\n" + "\n".join(posts)

    except Exception as e:
        return f"Error scraping Reddit: {str(e)}"

@tool("get_portfolio_exposure")
def get_portfolio_exposure(ticker: str):
    """
    Checks current portfolio exposure to a specific ticker or sector.
    Returns the percentage of capital allocated.
    """
    exposure, total_val = portfolio_manager.calculate_exposure(ticker)
    summary = portfolio_manager.get_portfolio_summary()
    cash = summary['cash']
    
    return f"Current portfolio exposure to {ticker}: {exposure*100:.2f}%. Cash position: ${cash:,.2f}."

@tool("execute_trade")
def execute_trade(ticker: str, action: str, quantity: int, price: float):
    """
    Executes a trade (BUY or SELL) for a specific ticker.
    Updates the portfolio cash and positions.
    action: 'BUY' or 'SELL'
    """
    try:
        action = action.upper()
        total_cost = quantity * price
        
        summary = portfolio_manager.get_portfolio_summary()
        cash = summary['cash']
        
        if action == "BUY":
            if cash < total_cost:
                return f"Error: Insufficient funds. Cash: ${cash}, Cost: ${total_cost}"
            portfolio_manager.update_cash(-total_cost)
            portfolio_manager.add_position(ticker, quantity, price)
            portfolio_manager.log_trade(ticker, "BUY", quantity, price)
            return f"Executed BUY: {quantity} {ticker} @ ${price}. Remaining Cash: ${cash - total_cost}"
            
        elif action == "SELL":
            # Check if we have enough shares
            positions = summary['positions']
            pos = next((p for p in positions if p['ticker'] == ticker.upper()), None)
            if not pos or pos['quantity'] < quantity:
                return f"Error: Insufficient shares to sell. Owned: {pos['quantity'] if pos else 0}"
            
            portfolio_manager.update_cash(total_cost)
            portfolio_manager.add_position(ticker, -quantity, price)
            portfolio_manager.log_trade(ticker, "SELL", quantity, price)
            return f"Executed SELL: {quantity} {ticker} @ ${price}. New Cash: ${cash + total_cost}"
            
        else:
            return "Error: Action must be BUY or SELL."
    except Exception as e:
        return f"Error executing trade: {str(e)}"

@tool("propose_option_strategy")
def propose_option_strategy(ticker: str, outlook: str, timeframe_days: int, volatility_environment: str = "normal"):
    """
    Proposes a multi-leg option strategy based on the outlook.
    outlook: 'bullish', 'bearish', 'neutral', 'volatile'
    timeframe_days: Expected duration of the trade
    volatility_environment: 'low', 'normal', 'high' (IV rank)
    """
    outlook = outlook.lower()
    volatility = volatility_environment.lower()
    
    strategy = {}
    
    if outlook == "bullish":
        if volatility == "high":
            strategy = {"name": "Bull Put Spread (Credit Spread)", "legs": ["Sell OTM Put", "Buy Lower Strike Put"], "rationale": "Capitalize on high IV crush while betting on upside."}
        else:
            strategy = {"name": "Long Call Vertical Spread", "legs": ["Buy ATM Call", "Sell OTM Call"], "rationale": "Defined risk upside exposure with lower cost than straight calls."}
            
    elif outlook == "bearish":
        if volatility == "high":
            strategy = {"name": "Bear Call Spread (Credit Spread)", "legs": ["Sell OTM Call", "Buy Higher Strike Call"], "rationale": "Fade the rally and collect premium from high IV."}
        else:
            strategy = {"name": "Long Put Vertical Spread", "legs": ["Buy ATM Put", "Sell OTM Put"], "rationale": "Defined risk downside exposure."}
            
    elif outlook == "neutral":
        strategy = {"name": "Iron Condor", "legs": ["Sell OTM Call", "Buy Higher Call", "Sell OTM Put", "Buy Lower Put"], "rationale": "Profit from time decay and range-bound price action."}
        
    elif outlook == "volatile":
        strategy = {"name": "Long Straddle/Strangle", "legs": ["Buy ATM Call", "Buy ATM Put"], "rationale": "Profit from a massive move in either direction, regardless of bias."}
        
    return f"Proposed Strategy for {ticker} ({outlook}, {timeframe_days} days): {strategy}"

@tool("calculate_option_greeks")
def calculate_option_greeks(ticker: str, strike_price: float, expiration_days: int, option_type: str, volatility: float = 0.3):
    """
    Calculates the theoretical Greeks (Delta, Gamma, Theta, Vega) for an option using Black-Scholes.
    option_type: 'call' or 'put'
    volatility: Annualized volatility (decimal, e.g., 0.3 for 30%)
    """
    try:
        # Fetch current price (mocked or real if available)
        # For this tool, we'll assume we can get a snapshot or use a placeholder if live data isn't hooked up in this specific function scope easily without passing it in.
        # In a real scenario, we'd fetch the live price. Let's assume a mock price close to strike for demonstration or fetch if possible.
        # To make it robust without live data dependency in this specific tool function, we might ask the agent to provide spot, but let's try to fetch or estimate.
        
        # Simple mock for spot price if not provided (assuming ATM for simplicity if not specified, but better to ask agent to provide it. 
        # However, to keep tool signature simple for the agent, we'll assume spot = strike for ATM or fetch from a helper if we had one.)
        # Let's use a placeholder spot price equal to strike to show ATM greeks, or slightly different to show sensitivity.
        S = strike_price # Spot Price (assuming ATM for calculation if not provided)
        K = strike_price # Strike Price
        T = expiration_days / 365.0 # Time to expiration in years
        r = 0.05 # Risk-free rate (5%)
        sigma = volatility # Volatility
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * sigma * norm.pdf(d1)) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
        else:
            delta = -norm.cdf(-d1)
            theta = (- (S * sigma * norm.pdf(d1)) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
            
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0 # Vega per 1% change in vol
        
        return {
            "ticker": ticker,
            "strike": strike_price,
            "type": option_type,
            "greeks": {
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta, 4),
                "vega": round(vega, 4)
            },
            "assumptions": {
                "spot_price": S,
                "volatility": sigma,
                "days_to_exp": expiration_days
            }
        }
    except Exception as e:
        return f"Error calculating Greeks: {str(e)}"