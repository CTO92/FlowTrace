import sys
import importlib.util
import os

def check_package(package_name, import_name=None):
    """Checks if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    if importlib.util.find_spec(import_name) is None:
        print(f"[-] {package_name} is NOT installed.")
        return False
    else:
        print(f"[+] {package_name} is installed.")
        return True

def check_env_file():
    """Checks if .env file exists and has content."""
    if os.path.exists('.env'):
        print("[+] .env file found.")
        # Optional: Check if keys are actually filled
        with open('.env', 'r') as f:
            content = f.read()
            if "API_KEY=" in content and not any(line.strip().endswith("=") for line in content.splitlines() if "API_KEY" in line and not line.startswith("#")):
                 print("    [!] Warning: Some API keys in .env might be empty.")
    else:
        print("[-] .env file NOT found. Please create one using the template.")

def main():
    print("--- FlowTrace: Phase 1 Environment Check ---\n")
    
    # List of (pip_package_name, import_module_name)
    packages = [
        ("openai", "openai"),
        ("aiohttp", "aiohttp"),
        ("websockets", "websockets"),
        ("polygon-api-client", "polygon"),
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("networkx", "networkx"),
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
        ("sec-edgar-downloader", "sec_edgar_downloader"),
        ("plyer", "plyer"),
        ("requests", "requests"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("pandas_datareader", "pandas_datareader"),
        ("yfinance", "yfinance"),
        ("tweepy", "tweepy"),
        ("langgraph", "langgraph"),
        ("langchain", "langchain"),
        ("playwright", "playwright"),
        ("plotly", "plotly"),
        ("python-dotenv", "dotenv"),
        ("langchain-openai", "langchain_openai"),
        ("beautifulsoup4", "bs4")
    ]

    check_env_file()
    print("\nChecking Dependencies:")
    
    missing = [pkg for pkg, imp in packages if not check_package(pkg, imp)]

    print("\n------------------------------------------")
    if not missing:
        print("Success: Environment appears ready for Phase 2.")
        print("Note: Ensure you run 'python -m playwright install' to download browser binaries for agents.")
    else:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()