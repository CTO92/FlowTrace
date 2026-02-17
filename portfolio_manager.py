import sqlite3
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "portfolio.db")

def init_db():
    """Initialize the portfolio database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Account table (Singleton row for simplicity)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash_balance REAL DEFAULT 100000.0
        )
    ''')
    
    # Positions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            quantity INTEGER,
            avg_price REAL
        )
    ''')
    
    # Trade History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            action TEXT,
            quantity INTEGER,
            price REAL
        )
    ''')
    
    # Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY,
            added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            target_price REAL,
            alert_condition TEXT -- 'ABOVE' or 'BELOW'
        )
    ''')
    
    # Ensure account row exists
    cursor.execute("INSERT OR IGNORE INTO account (id, cash_balance) VALUES (1, 100000.0)")
    
    conn.commit()
    conn.close()

def get_portfolio_summary():
    """Returns total cash and list of positions."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT cash_balance FROM account WHERE id=1")
    row = cursor.fetchone()
    cash = row['cash_balance'] if row else 0.0
    
    cursor.execute("SELECT * FROM positions")
    positions = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return {"cash": cash, "positions": positions}

def update_cash(amount: float):
    """Updates the cash balance."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE account SET cash_balance = cash_balance + ? WHERE id=1", (amount,))
    conn.commit()
    conn.close()

def calculate_exposure(ticker: str):
    """
    Calculates the percentage of portfolio allocated to a specific ticker.
    Returns (exposure_pct, total_portfolio_value).
    """
    summary = get_portfolio_summary()
    cash = summary['cash']
    positions = summary['positions']
    
    # Calculate total value based on cost basis (avg_price)
    # In a production system, we would fetch real-time prices here.
    total_invested = sum(p['quantity'] * p['avg_price'] for p in positions)
    total_value = cash + total_invested
    
    if total_value == 0:
        return 0.0, 0.0
        
    target_pos = next((p for p in positions if p['ticker'] == ticker.upper()), None)
    
    if not target_pos:
        return 0.0, total_value
        
    target_value = target_pos['quantity'] * target_pos['avg_price']
    exposure = target_value / total_value
    
    return exposure, total_value

def add_position(ticker: str, quantity: int, price: float):
    """Adds or updates a position."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    ticker = ticker.upper()
    
    # Check existing
    cursor.execute("SELECT quantity, avg_price FROM positions WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    
    if row:
        old_qty, old_price = row
        new_qty = old_qty + quantity
        if new_qty > 0:
            # Weighted average price only updates on BUY (positive quantity)
            # On SELL, cost basis per share remains the same
            if quantity > 0:
                new_avg = ((old_qty * old_price) + (quantity * price)) / new_qty
            else:
                new_avg = old_price
            cursor.execute("UPDATE positions SET quantity = ?, avg_price = ? WHERE ticker = ?", (new_qty, new_avg, ticker))
        else:
            cursor.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
    else:
        if quantity > 0:
            cursor.execute("INSERT INTO positions (ticker, quantity, avg_price) VALUES (?, ?, ?)", (ticker, quantity, price))
            
    conn.commit()
    conn.close()

def log_trade(ticker, action, quantity, price):
    """Logs a trade to the history table."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO trade_history (ticker, action, quantity, price) VALUES (?, ?, ?, ?)", (ticker, action, quantity, price))
    conn.commit()
    conn.close()

def get_trade_history():
    """Returns the list of executed trades."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trade_history ORDER BY timestamp DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_equity_curve():
    """Reconstructs the equity curve based on trade history (Realized P&L)."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM trade_history ORDER BY timestamp ASC")
    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if not trades:
        return []

    # Initial State
    cash = 100000.0 
    positions = {} 
    history = []
    
    for trade in trades:
        ticker = trade['ticker']
        action = trade['action']
        qty = trade['quantity']
        price = trade['price']
        timestamp = trade['timestamp']
        
        if action == 'BUY':
            cost = qty * price
            cash -= cost
            if ticker not in positions:
                positions[ticker] = {'qty': 0, 'avg_price': 0.0}
            pos = positions[ticker]
            new_qty = pos['qty'] + qty
            if new_qty > 0:
                new_avg = ((pos['qty'] * pos['avg_price']) + (qty * price)) / new_qty
                positions[ticker] = {'qty': new_qty, 'avg_price': new_avg}
            
        elif action == 'SELL':
            proceeds = qty * price
            cash += proceeds
            if ticker in positions:
                pos = positions[ticker]
                new_qty = pos['qty'] - qty
                if new_qty <= 0:
                    del positions[ticker]
                else:
                    positions[ticker]['qty'] = new_qty
        
        invested_value = sum(p['qty'] * p['avg_price'] for p in positions.values())
        total_equity = cash + invested_value
        
        history.append({
            "timestamp": timestamp,
            "equity": total_equity,
            "cash": cash,
            "invested": invested_value
        })
        
    return history

def get_risk_metrics():
    """Calculates VaR, Sharpe Ratio, Volatility, and Max Drawdown."""
    curve = get_equity_curve()
    if not curve:
        return None
    
    df = pd.DataFrame(curve)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Resample to daily to ensure consistent periods
    # We use the last equity value of the day
    daily_equity = df['equity'].resample('D').last().ffill()
    
    # If less than 2 data points, cannot calculate returns
    if len(daily_equity) < 2:
        return None

    # Calculate daily returns
    daily_returns = daily_equity.pct_change().dropna()
    
    if daily_returns.empty:
        return None
        
    # VaR (95% confidence) - Historical Simulation
    var_95 = np.percentile(daily_returns, 5)
    
    # Sharpe Ratio (assuming risk-free rate ~4% annual)
    rf_daily = 0.04 / 252
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    
    if std_return == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_return - rf_daily) / std_return * np.sqrt(252)
        
    # Max Drawdown
    rolling_max = daily_equity.cummax()
    drawdown = (daily_equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        "var_95_pct": var_95,
        "sharpe_ratio": sharpe,
        "volatility_annual": std_return * np.sqrt(252),
        "max_drawdown": max_drawdown
    }

def run_stress_test(stress_ticker: str, stress_pct_change: float):
    """
    Simulates the impact of a price change in a single asset on the total portfolio.
    stress_pct_change should be a float, e.g., -0.20 for a 20% drop.
    """
    summary = get_portfolio_summary()
    cash = summary['cash']
    positions = summary['positions']

    if not positions:
        return None

    tickers = [p['ticker'] for p in positions]
    
    try:
        # Fetch latest available prices
        data = yf.download(tickers, period="2d", progress=False)
        
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']

        # Get the last valid price for each ticker
        if len(tickers) == 1:
            last_prices = pd.Series([prices.iloc[-1]], index=tickers)
        else:
            last_prices = prices.iloc[-1]

        current_market_value = sum(pos['quantity'] * last_prices[pos['ticker']] for pos in positions)
        initial_total_equity = cash + current_market_value
        
        # Calculate impact
        stressed_pos = next((p for p in positions if p['ticker'].upper() == stress_ticker.upper()), None)
        stressed_pos_value = stressed_pos['quantity'] * last_prices[stressed_pos['ticker']]
        pnl_impact = stressed_pos_value * stress_pct_change
        
        stressed_total_equity = initial_total_equity + pnl_impact
        pnl_impact_pct = (pnl_impact / initial_total_equity) if initial_total_equity != 0 else 0

        return {
            "initial_equity": initial_total_equity,
            "stressed_equity": stressed_total_equity,
            "pnl_impact": pnl_impact,
            "pnl_impact_pct": pnl_impact_pct
        }
    except Exception as e:
        print(f"Error running stress test (could not fetch live prices): {e}")
        return None

def run_macro_stress_test(macro_ticker: str, macro_change_pct: float, lookback_days: int = 180):
    """
    Simulates portfolio impact based on a macro factor move using historical beta.
    macro_change_pct: float (e.g., 0.05 for 5% increase in the factor)
    """
    summary = get_portfolio_summary()
    positions = summary['positions']
    cash = summary['cash']
    
    if not positions:
        return None
        
    tickers = [p['ticker'] for p in positions]
    
    try:
        # Fetch history for holdings + macro
        all_tickers = list(set(tickers + [macro_ticker]))
        data = yf.download(all_tickers, period=f"{lookback_days}d", progress=False)
        
        if 'Adj Close' in data:
            prices = data['Adj Close']
        elif 'Close' in data:
            prices = data['Close']
        else:
            prices = data
            
        # Ensure DataFrame format
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=all_tickers[0])
            
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if macro_ticker not in returns.columns:
            return None
            
        # Get current prices for valuation (use last available)
        current_prices = prices.iloc[-1]
        
        results = {}
        total_pnl_impact = 0.0
        current_invested_value = 0.0
        
        for pos in positions:
            ticker = pos['ticker']
            if ticker not in returns.columns:
                continue
                
            # Calculate Beta: Cov(Stock, Macro) / Var(Macro)
            aligned = returns[[ticker, macro_ticker]].dropna()
            if len(aligned) < 30:
                beta = 0.0
            else:
                cov = aligned.cov().iloc[0, 1]
                var_m = aligned[macro_ticker].var()
                beta = cov / var_m if var_m != 0 else 0
            
            # Expected move = Beta * Macro Move
            expected_stock_move = beta * macro_change_pct
            
            # Dollar impact
            # Use market price if available, else cost basis
            market_price = current_prices.get(ticker, pos['avg_price'])
            current_val = pos['quantity'] * market_price
            current_invested_value += current_val
            
            pnl = current_val * expected_stock_move
            total_pnl_impact += pnl
            
            results[ticker] = {
                "beta": beta,
                "expected_move": expected_stock_move,
                "pnl": pnl
            }
            
        current_total_equity = cash + current_invested_value
        stressed_equity = current_total_equity + total_pnl_impact
        
        return {
            "macro_ticker": macro_ticker,
            "macro_change_pct": macro_change_pct,
            "initial_equity": current_total_equity,
            "stressed_equity": stressed_equity,
            "pnl_impact": total_pnl_impact,
            "pnl_impact_pct": (total_pnl_impact / current_total_equity) if current_total_equity else 0,
            "details": results
        }
        
    except Exception as e:
        print(f"Error in macro stress test: {e}")
        return None

def get_daily_sentiment():
    """
    Aggregates daily sentiment scores for portfolio holdings from the signals table.
    Returns a DataFrame with date and average sentiment score.
    """
    summary = get_portfolio_summary()
    positions = summary['positions']
    if not positions:
        return None
        
    tickers = [p['ticker'] for p in positions]
    placeholders = ','.join(['?'] * len(tickers))
    
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    # Query unified_score for held tickers, averaged by day
    query = f"""
        SELECT date(timestamp) as date, AVG(unified_score) as avg_score
        FROM signals 
        WHERE target_ticker IN ({placeholders})
        GROUP BY date(timestamp)
        ORDER BY date ASC
    """
    
    df = pd.read_sql_query(query, conn, params=tickers)
    conn.close()
    
    return df

def get_holdings_correlation(lookback_days=90):
    """
    Calculates the correlation matrix of the current portfolio holdings.
    Fetches historical price data using yfinance.
    """
    summary = get_portfolio_summary()
    positions = summary['positions']
    
    if not positions:
        return None
        
    tickers = [p['ticker'] for p in positions]
    
    # If only one asset, correlation is 1.0
    if len(tickers) < 2:
        return pd.DataFrame(data=[[1.0]], index=tickers, columns=tickers)
        
    try:
        # Fetch data
        data = yf.download(tickers, period=f"{lookback_days}d", progress=False)
        
        # Handle yfinance return structure (MultiIndex if multiple tickers)
        if 'Adj Close' in data:
            prices = data['Adj Close']
        elif 'Close' in data:
            prices = data['Close']
        else:
            prices = data
            
        # Calculate daily returns and correlation
        returns = prices.pct_change().dropna()
        return returns.corr()
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return None

def add_to_watchlist(ticker):
    """Adds a ticker to the watchlist."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker.upper(),))
        conn.commit()
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
    finally:
        conn.close()

def remove_from_watchlist(ticker):
    """Removes a ticker from the watchlist."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
    conn.commit()
    conn.close()

def get_watchlist():
    """Returns the watchlist."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM watchlist ORDER BY added_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_watchlist_with_prices():
    """Returns watchlist with current prices."""
    watchlist = get_watchlist()
    if not watchlist:
        return []
    
    tickers = [w['ticker'] for w in watchlist]
    try:
        # Fetch 5 days to ensure we have previous close for calculation
        data = yf.download(tickers, period="5d", progress=False)
        
        price_col = 'Adj Close' if 'Adj Close' in data else 'Close'
        prices = data[price_col]
        
        # Ensure DataFrame format even for single ticker
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
            
        results = []
        for t in tickers:
            last_price = 0.0
            change_pct = 0.0
            
            if t in prices.columns:
                series = prices[t].dropna()
                if len(series) >= 2:
                    last_price = float(series.iloc[-1])
                    prev_price = float(series.iloc[-2])
                    change_pct = ((last_price - prev_price) / prev_price) * 100
                elif len(series) == 1:
                    last_price = float(series.iloc[-1])
            
            # Find added_at
            w_item = next((item for item in watchlist if item['ticker'] == t), None)
            added_at = w_item['added_at'] if w_item else None
            
            results.append({"Ticker": t, "Price": last_price, "ChangePct": change_pct, "Added": added_at})
            
        return results
    except Exception as e:
        print(f"Error fetching watchlist prices: {e}")
        return [{"Ticker": w['ticker'], "Price": 0.0, "ChangePct": 0.0, "Added": w['added_at']} for w in watchlist]

def get_macro_history(lookback_days=365*2):
    """
    Fetches historical macro data from FRED.
    Returns a DataFrame with columns: FEDFUNDS, DGS10 (10Y Yield), CPIAUCSL (CPI), UNRATE (Unemployment).
    """
    start = datetime.datetime.now() - datetime.timedelta(days=lookback_days)
    end = datetime.datetime.now()
    
    series_ids = {
        'FEDFUNDS': 'Fed Funds Rate',
        'DGS3MO': '3M Treasury Yield',
        'DGS2': '2Y Treasury Yield',
        'DGS5': '5Y Treasury Yield',
        'DGS10': '10Y Treasury Yield',
        'DGS30': '30Y Treasury Yield',
        'CPIAUCSL': 'CPI (Inflation)',
        'UNRATE': 'Unemployment Rate',
        'M2SL': 'M2 Money Supply'
    }
    
    try:
        df = web.DataReader(list(series_ids.keys()), 'fred', start, end)
        df.rename(columns=series_ids, inplace=True)
        
        # Forward fill to handle different reporting frequencies (daily vs monthly)
        df.ffill(inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching macro history: {e}")
        return pd.DataFrame()

def get_fed_watch_data():
    """
    Fetches FedWatch probabilities from CME Group API.
    Returns a list of meetings with probabilities.
    """
    url = "https://www.cmegroup.com/CmeWS/mvc/XS/IW/360/targetRateProbability"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        "Accept": "application/json"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching FedWatch: {e}")
    return None

def get_commodities_history(lookback_days=365):
    """
    Fetches historical data for key commodities using yfinance.
    Returns a DataFrame with Close prices.
    """
    tickers_map = {
        'CL=F': 'Crude Oil',
        'GC=F': 'Gold',
        'HG=F': 'Copper',
        'SI=F': 'Silver',
        'NG=F': 'Natural Gas'
    }
    symbols = list(tickers_map.keys())
    
    try:
        data = yf.download(symbols, period=f"{lookback_days}d", progress=False)
        
        if 'Adj Close' in data:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
        return prices.rename(columns=tickers_map)
    except Exception as e:
        print(f"Error fetching commodities: {e}")
        return pd.DataFrame()

def set_price_alert(ticker, target_price, condition):
    """Sets a price alert for a watchlist item."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Ensure ticker is in watchlist first
        cursor.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker.upper(),))
        
        cursor.execute("""
            UPDATE watchlist 
            SET target_price = ?, alert_condition = ? 
            WHERE ticker = ?
        """, (target_price, condition, ticker.upper()))
        conn.commit()
    except Exception as e:
        print(f"Error setting alert: {e}")
    finally:
        conn.close()

def check_alerts(current_prices):
    """
    Checks if any alerts are triggered based on current prices.
    current_prices: dict {ticker: price}
    Returns list of triggered alerts: [{ticker, price, target, condition}]
    """
    watchlist = get_watchlist()
    triggered = []
    
    for item in watchlist:
        ticker = item['ticker']
        target = item['target_price']
        cond = item['alert_condition']
        curr = current_prices.get(ticker)
        
        if target and cond and curr:
            if cond == 'ABOVE' and curr >= target:
                triggered.append({"ticker": ticker, "price": curr, "target": target, "condition": cond})
            elif cond == 'BELOW' and curr <= target:
                triggered.append({"ticker": ticker, "price": curr, "target": target, "condition": cond})
                
    return triggered

# Seed mock data if empty for demonstration purposes
def _seed_demo_data():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM positions")
    if cursor.fetchone()[0] == 0:
        # Seed some initial positions
        cursor.execute("INSERT INTO positions (ticker, quantity, avg_price) VALUES (?, ?, ?)", ("NVDA", 20, 450.0))
        cursor.execute("INSERT INTO positions (ticker, quantity, avg_price) VALUES (?, ?, ?)", ("MSFT", 50, 350.0))
        # Seed trade history for performance chart consistency
        cursor.execute("INSERT INTO trade_history (ticker, action, quantity, price) VALUES (?, ?, ?, ?)", ("NVDA", "BUY", 20, 450.0))
        cursor.execute("INSERT INTO trade_history (ticker, action, quantity, price) VALUES (?, ?, ?, ?)", ("MSFT", "BUY", 50, 350.0))
        cursor.execute("UPDATE account SET cash_balance = 73500.0 WHERE id=1")
        conn.commit()
    conn.close()

if __name__ == "__main__":
    _seed_demo_data()
else:
    # Auto-seed on import if empty, just so the agent has something to look at immediately
    _seed_demo_data()