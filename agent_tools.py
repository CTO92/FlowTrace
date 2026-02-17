import os
import asyncio
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