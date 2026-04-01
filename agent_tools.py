import os
import json
import sqlite3
import asyncio
import requests
import datetime
import glob
import feedparser
import yfinance as yf
from sec_edgar_downloader import Downloader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Ensure SERPER_API_KEY is loaded
if not os.getenv("SERPER_API_KEY"):
    print("[-] Warning: SERPER_API_KEY not found. Web search will fail.")

@tool("web_search")
def web_search(query: str):
    """
    Search the web for recent news, filings, or specific information.
    Useful for finding 'latest contracts', 'earnings reports', or 'supplier relationships'.
    """
    search = GoogleSerperAPIWrapper()
    return search.run(query)

@tool("scrape_web_page")
async def scrape_web_page(url: str):
    """
    Scrape the text content from a specific URL. 
    Useful for reading full news articles, press releases, or SEC filing pages.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            # Create a new context with a realistic user agent to avoid blocking
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = await context.new_page()
            
            try:
                await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            except Exception:
                # If timeout, try to proceed with what we have
                pass
                
            content = await page.content()
            await browser.close()
            
            # Parse HTML to text
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text(separator="\n")
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit length to avoid token limits
            return clean_text[:8000]
            
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

@tool("get_competitors")
def get_competitors(ticker: str):
    """
    Get a list of competitors and sector peers for a given ticker.
    Useful for analyzing relative value or finding sympathetic moves in the sector.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    url = "https://finnhub.io/api/v1/stock/peers"
    params = {'symbol': ticker, 'token': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            peers = response.json()
            return f"Competitors/Peers for {ticker}: {', '.join(peers)}"
        return f"Error fetching competitors: {response.status_code}"
    except Exception as e:
        return f"Exception fetching competitors: {str(e)}"

@tool("get_insider_trades")
def get_insider_trades(ticker: str):
    """
    Fetches insider sentiment data for a given ticker to gauge management confidence.
    Returns monthly share purchase/sale ratios.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    now = datetime.datetime.now()
    start_date = (now - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    
    url = "https://finnhub.io/api/v1/stock/insider-sentiment"
    params = {'symbol': ticker, 'from': start_date, 'to': end_date, 'token': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json().get('data', [])
            if not data:
                return f"No insider sentiment data found for {ticker} in the last 90 days."
            
            result = f"Insider Sentiment (Last 90 Days) for {ticker}:\n"
            for item in data:
                month = item.get('month')
                change = item.get('change')
                mspr = item.get('mspr')
                result += f"- Month {month}: Net Change={change}, MSPR={mspr}\n"
            return result
        return f"Error fetching insider sentiment: {response.status_code}"
    except Exception as e:
        return f"Exception fetching insider sentiment: {str(e)}"

@tool("get_analyst_ratings")
def get_analyst_ratings(ticker: str):
    """
    Fetches the latest analyst recommendation trends (Buy/Hold/Sell) for a ticker.
    Returns the consensus for the current period.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    url = "https://finnhub.io/api/v1/stock/recommendation"
    params = {'symbol': ticker, 'token': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return f"No analyst ratings found for {ticker}."
            
            # Finnhub returns a list of objects, usually sorted by period. We take the latest.
            latest = data[0]
            period = latest.get('period', 'Unknown')
            
            return (f"Analyst Ratings for {ticker} (Period: {period}):\n"
                    f"- Strong Buy: {latest.get('strongBuy', 0)}\n"
                    f"- Buy: {latest.get('buy', 0)}\n"
                    f"- Hold: {latest.get('hold', 0)}\n"
                    f"- Sell: {latest.get('sell', 0)}\n"
                    f"- Strong Sell: {latest.get('strongSell', 0)}")
        return f"Error fetching ratings: {response.status_code}"
    except Exception as e:
        return f"Exception fetching ratings: {str(e)}"

@tool("get_fundamental_ratios")
def get_fundamental_ratios(ticker: str):
    """
    Fetches key fundamental ratios (P/E, PEG, Debt/Equity, ROE) from Finnhub.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {'symbol': ticker, 'metric': 'all', 'token': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get('metric', {})
            
            if not metrics:
                return f"No fundamental data found for {ticker}."
            
            # Extract key metrics
            pe = metrics.get('peBasicExclExtraTTM', 'N/A')
            peg = metrics.get('pegTTM', 'N/A')
            debt_equity = metrics.get('totalDebt/totalEquityQuarterly', 'N/A')
            roe = metrics.get('roeTTM', 'N/A')
            beta = metrics.get('beta', 'N/A')
            div_yield = metrics.get('dividendYieldIndicatedAnnual', 'N/A')
            
            return (f"Fundamental Ratios for {ticker}:\n"
                    f"- P/E (TTM): {pe}\n"
                    f"- PEG (TTM): {peg}\n"
                    f"- Debt/Equity (Quarterly): {debt_equity}\n"
                    f"- ROE (TTM): {roe}%\n"
                    f"- Beta: {beta}\n"
                    f"- Dividend Yield: {div_yield}%")
        return f"Error fetching fundamentals: {response.status_code}"
    except Exception as e:
        return f"Exception fetching fundamentals: {str(e)}"

@tool("get_earnings_calendar")
def get_earnings_calendar(ticker: str):
    """
    Fetches upcoming earnings date and estimates for a ticker.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    # Look ahead 3 months
    now = datetime.datetime.now()
    start_date = now.strftime("%Y-%m-%d")
    end_date = (now + datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {'symbol': ticker, 'from': start_date, 'to': end_date, 'token': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            earnings_list = data.get('earningsCalendar', [])
            
            if not earnings_list:
                return f"No upcoming earnings found for {ticker} in the next 90 days."
            
            # Sort by date just in case
            earnings_list.sort(key=lambda x: x.get('date', '9999-99-99'))
            
            next_earnings = earnings_list[0]
            date = next_earnings.get('date', 'N/A')
            estimate = next_earnings.get('epsEstimate', 'N/A')
            quarter = next_earnings.get('quarter', 'N/A')
            year = next_earnings.get('year', 'N/A')
            
            return (f"Next Earnings for {ticker}:\n"
                    f"- Date: {date}\n"
                    f"- Quarter: {quarter} {year}\n"
                    f"- EPS Estimate: {estimate}")
        return f"Error fetching earnings: {response.status_code}"
    except Exception as e:
        return f"Exception fetching earnings: {str(e)}"

@tool("get_short_interest")
def get_short_interest(ticker: str):
    """
    Fetches short interest data including float short, days to cover, and short interest ratio.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    url = "https://finnhub.io/api/v1/stock/short-interest"
    params = {'symbol': ticker, 'token': api_key}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data or 'data' not in data or not data['data']:
                return f"No short interest data found for {ticker}."
            
            # Finnhub returns a list of historical data. We take the latest.
            latest = data['data'][0]
            
            date = latest.get('date', 'N/A')
            short_int = latest.get('shortInterest', 'N/A')
            days_cover = latest.get('daysToCover', 'N/A')
            
            return (f"Short Interest for {ticker} (Date: {date}):\n"
                    f"- Short Interest: {short_int}\n"
                    f"- Days to Cover: {days_cover}")
        return f"Error fetching short interest: {response.status_code}"
    except Exception as e:
        return f"Exception fetching short interest: {str(e)}"

@tool("fetch_rss_feed")
def fetch_rss_feed(feed_url: str, limit: int = 5):
    """
    Fetches the latest news items from a specific RSS feed URL.
    Returns a list of titles and links.
    """
    try:
        feed = feedparser.parse(feed_url)
        # Check for bozo bit (malformed XML) but sometimes it parses anyway
        if not feed.entries and feed.bozo:
             return f"Error parsing feed: {feed.bozo_exception}"
        
        entries = feed.entries[:limit]
        if not entries:
            return "No entries found in feed."
            
        result = f"Latest news from {feed.feed.get('title', feed_url)}:\n"
        for entry in entries:
            published = entry.get('published', 'N/A')
            result += f"- {entry.title} ({published}) - {entry.link}\n"
            
        return result
    except Exception as e:
        return f"Error fetching RSS feed: {str(e)}"

@tool("visualize_supply_chain")
def visualize_supply_chain(ticker: str):
    """
    Generates a Graphviz DOT string representing the supply chain and competitive landscape for a ticker based on the local Knowledge Graph.
    Returns the DOT string which can be rendered into a graph.
    """
    db_path = os.path.join(os.path.dirname(__file__), "knowledge_graph.db")
    if not os.path.exists(db_path):
        return "Error: Knowledge Graph database not found."
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get direct relationships where the ticker is either source or target
        cursor.execute("""
            SELECT source_ticker, target_ticker, relationship_type 
            FROM relationships 
            WHERE source_ticker = ? OR target_ticker = ?
        """, (ticker, ticker))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return f"No relationships found for {ticker} in the Knowledge Graph."
            
        dot = "digraph SupplyChain {\n"
        dot += "  rankdir=LR;\n"
        dot += "  node [shape=box, style=filled, fontname=\"Arial\"];\n"
        dot += f'  "{ticker}" [fillcolor="#FFC107", style=filled];\n'
        
        for source, target, rel_type in rows:
            color = "black"
            if rel_type == "Supplier":
                color = "#4CAF50" # Green
            elif rel_type == "Customer":
                color = "#2196F3" # Blue
            elif rel_type == "Competitor":
                color = "#F44336" # Red
                
            dot += f'  "{source}" -> "{target}" [label="{rel_type}", color="{color}", fontcolor="{color}"];\n'
            
        dot += "}"
        return dot
        
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

@tool("get_sec_filing_section")
def get_sec_filing_section(ticker: str, section: str = "Risk Factors", filing_type: str = "10-K"):
    """
    Downloads the latest SEC filing (10-K or 10-Q) for a ticker and extracts a specific section.
    Common sections: 'Risk Factors', 'Management Discussion', 'Business'.
    """
    email = os.getenv("SEC_EMAIL", "user@example.com")
    # Use a temp directory for agent downloads
    dl_path = os.path.join(os.path.dirname(__file__), "sec_filings_agent_temp")
    
    # Initialize downloader
    dl = Downloader("FlowTrace_Agent", email, dl_path)
    
    try:
        # Download latest filing
        dl.get(filing_type, ticker, limit=1)
        
        # Find the downloaded file
        path_pattern = os.path.join(dl_path, "sec-edgar-filings", ticker, filing_type, "*", "*.html")
        files = glob.glob(path_pattern)
        
        if not files:
             # Fallback to text if html not found
            path_pattern = os.path.join(dl_path, "sec-edgar-filings", ticker, filing_type, "*", "*.txt")
            files = glob.glob(path_pattern)
            
        if not files:
            return f"No {filing_type} found for {ticker}."
            
        file_path = files[0]
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator="\n")
        
        # Heuristic Extraction
        search_term = ""
        if "risk" in section.lower():
            search_term = "Item 1A"
        elif "business" in section.lower():
            search_term = "Item 1."
        elif "management" in section.lower() or "md&a" in section.lower():
            search_term = "Item 7."
            
        if not search_term:
             start_idx = text.lower().find(section.lower())
             if start_idx != -1:
                 return f"Extracted '{section}' (heuristic match):\n{text[start_idx:start_idx+5000]}..."
             return f"Section '{section}' not mapped to standard Item. Returning filing summary:\n{text[:3000]}..."

        # Find the Item (skip potential TOC entries by looking deeper in text)
        matches = [i for i in range(len(text)) if text.startswith(search_term, i)]
        target_idx = matches[-1] if matches else text.find(search_term)
        
        if target_idx == -1:
             return f"Could not find '{search_term}' in text. Returning summary:\n{text[:3000]}..."
            
        extracted = text[target_idx:target_idx+10000] # Return 10k chars
        
        return f"Extracted '{section}' ({search_term}) from {ticker} {filing_type}:\n{extracted}..."
        
    except Exception as e:
        return f"Error retrieving SEC filing: {str(e)}"

@tool("compare_peers")
def compare_peers(ticker: str):
    """
    Generates a markdown table comparing a ticker against its competitors on P/E, Revenue Growth, and Margins.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "Finnhub API key not configured."
        
    # Get Peers
    peers_url = "https://finnhub.io/api/v1/stock/peers"
    try:
        r = requests.get(peers_url, params={'symbol': ticker, 'token': api_key})
        peers = r.json() if r.status_code == 200 else []
    except Exception as e:
        return f"Error fetching peers: {str(e)}"
        
    if not peers:
        peers = [ticker]
    
    # Limit to top 5 (Ticker + 4 peers)
    target_peers = peers[:5]
    
    rows = []
    headers = ["Ticker", "P/E", "Rev Growth", "Net Margin", "ROE"]
    
    for t in target_peers:
        metric_url = "https://finnhub.io/api/v1/stock/metric"
        try:
            r = requests.get(metric_url, params={'symbol': t, 'metric': 'all', 'token': api_key})
            if r.status_code == 200:
                m = r.json().get('metric', {})
                pe = m.get('peBasicExclExtraTTM', 'N/A')
                rev = m.get('revenueGrowthTTMYoy', 'N/A')
                margin = m.get('netProfitMarginTTM', 'N/A')
                roe = m.get('roeTTM', 'N/A')
                
                # Formatting
                pe = f"{pe:.2f}" if isinstance(pe, (int, float)) else pe
                rev = f"{rev:.2f}%" if isinstance(rev, (int, float)) else rev
                margin = f"{margin:.2f}%" if isinstance(margin, (int, float)) else margin
                roe = f"{roe:.2f}%" if isinstance(roe, (int, float)) else roe
                
                rows.append(f"| {t} | {pe} | {rev} | {margin} | {roe} |")
            else:
                rows.append(f"| {t} | N/A | N/A | N/A | N/A |")
        except:
            rows.append(f"| {t} | Error | Error | Error | Error |")
            
    table = f"| {' | '.join(headers)} |\n|---|---|---|---|---|\n" + "\n".join(rows)
    return f"Peer Comparison Table:\n{table}"


@tool("get_comprehensive_fundamentals")
def get_comprehensive_fundamentals(ticker: str):
    """
    Fetch comprehensive fundamental data for a ticker using yfinance.
    Returns valuation, profitability, growth, financial health, dividend,
    and per-share metrics as a structured JSON string.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("trailingPegRatio") is None and info.get("marketCap") is None:
            # Minimal check that we got real data
            pass

        def _get(key):
            """Extract a value from info, returning None if missing."""
            return info.get(key)

        data = {
            "company_name": _get("longName"),
            "sector": _get("sector"),
            "industry": _get("industry"),
            "valuation": {
                "marketCap": _get("marketCap"),
                "enterpriseValue": _get("enterpriseValue"),
                "trailingPE": _get("trailingPE"),
                "forwardPE": _get("forwardPE"),
                "pegRatio": _get("pegRatio"),
                "priceToBook": _get("priceToBook"),
                "priceToSalesTrailing12Months": _get("priceToSalesTrailing12Months"),
                "enterpriseToRevenue": _get("enterpriseToRevenue"),
                "enterpriseToEbitda": _get("enterpriseToEbitda"),
            },
            "profitability": {
                "profitMargins": _get("profitMargins"),
                "operatingMargins": _get("operatingMargins"),
                "grossMargins": _get("grossMargins"),
                "returnOnEquity": _get("returnOnEquity"),
                "returnOnAssets": _get("returnOnAssets"),
            },
            "growth": {
                "revenueGrowth": _get("revenueGrowth"),
                "earningsGrowth": _get("earningsGrowth"),
                "earningsQuarterlyGrowth": _get("earningsQuarterlyGrowth"),
            },
            "financial_health": {
                "debtToEquity": _get("debtToEquity"),
                "currentRatio": _get("currentRatio"),
                "quickRatio": _get("quickRatio"),
                "totalCash": _get("totalCash"),
                "totalDebt": _get("totalDebt"),
                "freeCashflow": _get("freeCashflow"),
                "operatingCashflow": _get("operatingCashflow"),
            },
            "dividends": {
                "dividendYield": _get("dividendYield"),
                "dividendRate": _get("dividendRate"),
                "payoutRatio": _get("payoutRatio"),
                "fiveYearAvgDividendYield": _get("fiveYearAvgDividendYield"),
            },
            "per_share": {
                "bookValue": _get("bookValue"),
                "revenuePerShare": _get("revenuePerShare"),
                "trailingEps": _get("trailingEps"),
                "forwardEps": _get("forwardEps"),
            },
        }

        return json.dumps(data, indent=2, default=str)

    except Exception as e:
        return f"Error fetching comprehensive fundamentals for {ticker}: {str(e)}"


@tool("get_financial_statements")
def get_financial_statements(ticker: str):
    """
    Fetch the last 4 quarters of income statement, balance sheet, and cash flow
    data for a ticker using yfinance. Returns a formatted readable string with
    quarterly columns.
    """
    try:
        stock = yf.Ticker(ticker)

        def _format_statement(title, df, row_keys):
            """Format a DataFrame into a readable quarterly table."""
            if df is None or df.empty:
                return f"\n{title}:\n  No data available.\n"

            # Limit to 4 most recent quarters (columns)
            df = df.iloc[:, :4]

            # Build header with quarter labels
            col_labels = [col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col) for col in df.columns]
            header = f"{'Metric':<35}" + "".join(f"{lbl:>18}" for lbl in col_labels)

            lines = [f"\n{title}:", header, "-" * len(header)]

            for key in row_keys:
                if key in df.index:
                    row = df.loc[key]
                    values = []
                    for val in row:
                        if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "NaTType"):
                            values.append(f"{'N/A':>18}")
                        else:
                            try:
                                values.append(f"{float(val):>18,.0f}")
                            except (ValueError, TypeError):
                                values.append(f"{str(val):>18}")
                    lines.append(f"{key:<35}" + "".join(values))
                else:
                    lines.append(f"{key:<35}" + "".join(f"{'N/A':>18}" for _ in col_labels))

            return "\n".join(lines) + "\n"

        income_keys = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
        balance_keys = ["Total Assets", "Total Liabilities Net Minority Interest",
                        "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents"]
        cashflow_keys = ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"]

        result = f"Financial Statements for {ticker} (Last 4 Quarters):\n"
        result += _format_statement("Income Statement", stock.quarterly_income_stmt, income_keys)
        result += _format_statement("Balance Sheet", stock.quarterly_balance_sheet, balance_keys)
        result += _format_statement("Cash Flow", stock.quarterly_cashflow, cashflow_keys)

        return result

    except Exception as e:
        return f"Error fetching financial statements for {ticker}: {str(e)}"


@tool("get_earnings_history")
def get_earnings_history(ticker: str):
    """
    Fetch the last 4 quarters of earnings history (actual vs estimated EPS and
    surprise percentage) for a ticker using yfinance. Returns a formatted string.
    """
    try:
        stock = yf.Ticker(ticker)

        # Try earnings_dates first (contains EPS estimate and actual)
        try:
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                # Filter to rows that have both Reported EPS and EPS Estimate
                cols_available = ed.columns.tolist()
                has_reported = "Reported EPS" in cols_available
                has_estimate = "EPS Estimate" in cols_available

                if has_reported and has_estimate:
                    # Drop rows where Reported EPS is NaN (future dates)
                    filtered = ed.dropna(subset=["Reported EPS"]).head(4)

                    if not filtered.empty:
                        lines = [f"Earnings History for {ticker} (Last {len(filtered)} Quarters):\n"]
                        lines.append(f"{'Date':<22}{'EPS Estimate':>14}{'Reported EPS':>14}{'Surprise':>12}{'Surprise %':>12}")
                        lines.append("-" * 74)

                        for idx, row in filtered.iterrows():
                            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                            estimate = row.get("EPS Estimate")
                            reported = row.get("Reported EPS")

                            est_str = f"{estimate:.2f}" if estimate is not None and not (isinstance(estimate, float) and str(estimate) == "nan") else "N/A"
                            rep_str = f"{reported:.2f}" if reported is not None and not (isinstance(reported, float) and str(reported) == "nan") else "N/A"

                            # Calculate surprise
                            if (estimate is not None and reported is not None
                                    and not (isinstance(estimate, float) and str(estimate) == "nan")
                                    and not (isinstance(reported, float) and str(reported) == "nan")):
                                surprise = reported - estimate
                                surprise_pct = (surprise / abs(estimate) * 100) if estimate != 0 else 0.0
                                surp_str = f"{surprise:+.2f}"
                                pct_str = f"{surprise_pct:+.1f}%"
                            else:
                                surp_str = "N/A"
                                pct_str = "N/A"

                            lines.append(f"{date_str:<22}{est_str:>14}{rep_str:>14}{surp_str:>12}{pct_str:>12}")

                        return "\n".join(lines)
        except Exception:
            pass  # Fall through to alternative method

        # Fallback: construct from quarterly income statement EPS
        try:
            inc = stock.quarterly_income_stmt
            if inc is not None and not inc.empty and "Basic EPS" in inc.index:
                eps_row = inc.loc["Basic EPS"].iloc[:4]
                lines = [f"Earnings History for {ticker} (from quarterly income statement):\n"]
                lines.append(f"{'Quarter Ending':<22}{'Basic EPS':>14}")
                lines.append("-" * 36)

                for col, val in eps_row.items():
                    date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                    val_str = f"{float(val):.2f}" if val is not None else "N/A"
                    lines.append(f"{date_str:<22}{val_str:>14}")

                lines.append("\nNote: Estimate and surprise data not available; showing reported EPS only.")
                return "\n".join(lines)
        except Exception:
            pass

        return f"No earnings history data available for {ticker}."

    except Exception as e:
        return f"Error fetching earnings history for {ticker}: {str(e)}"