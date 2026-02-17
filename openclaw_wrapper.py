import asyncio
import random
from playwright.async_api import async_playwright

class OpenClawSession:
    """
    A wrapper around Playwright to mimic OpenClaw's stealth capabilities.
    Provides anti-detect browsing sessions for agents to scrape difficult targets.
    """
    def __init__(self, headless=True):
        self.headless = headless
        self.browser = None
        self.playwright = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        # Launch with stealth args to avoid detection by anti-bot systems
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--window-position=0,0",
                "--ignore-certificate-errors",
                "--ignore-certificate-errors-spki-list",
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def new_stealth_page(self):
        """Creates a new page with stealth context (randomized viewport, masked webdriver)."""
        # Randomize viewport slightly to avoid fingerprinting
        width = 1920 + random.randint(-50, 50)
        height = 1080 + random.randint(-50, 50)
        
        context = await self.browser.new_context(
            viewport={"width": width, "height": height},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="America/New_York",
            java_script_enabled=True
        )
        
        # Add init script to mask webdriver property (Anti-detect)
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        return page