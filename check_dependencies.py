import importlib.util
import subprocess
import sys

def check_package(package_name, import_name=None):
    """Checks if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[-] Dependency '{package_name}' is NOT installed. Please run: pip install {package_name}")
        return False
    else:
        print(f"[+] Dependency '{package_name}' is installed.")
        return True

def check_playwright_browsers():
    """Checks if Playwright browsers are installed by running a dry-run."""
    print("\n[*] Checking for Playwright browser binaries...")
    try:
        # Using 'install --with-deps --dry-run' is a safe way to check without modifying the system.
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--with-deps", "--dry-run", "chromium"],
            capture_output=True, text=True, check=True, timeout=60
        )
        print("[+] Playwright and its browser dependencies seem to be correctly set up.")
        print("    To be certain, you can manually run: python -m playwright install --with-deps chromium")
        return True
    except FileNotFoundError:
        print("[-] 'playwright' command not found. Is the package installed?")
        return False
    except subprocess.CalledProcessError as e:
        print("[-] Playwright browser check failed. It's likely they are not installed.")
        print(f"    Error: {e.stderr}")
        print("    Please run: python -m playwright install --with-deps chromium")
        return False

def main():
    print("--- Advanced Dependency Check ---")
    all_ok = True
    
    if not check_package("pandas-datareader", "pandas_datareader"):
        all_ok = False
        
    if check_package("playwright"):
        if not check_playwright_browsers():
            all_ok = False
            
    print("\n---------------------------------")
    if all_ok:
        print("✅ All checked dependencies are satisfied.")
    else:
        print("❌ Some dependencies are missing. Please follow the instructions above.")

if __name__ == "__main__":
    main()