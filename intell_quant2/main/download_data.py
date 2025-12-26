
import os
import sys
import time
import pandas as pd
import akshare as ak
import duckdb
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import from the sibling script which also sets up src path
# We only import constants, not the fetching function which has side effects (local cache preference)
from main.run_backtest import MIN_MARKET_CAP

# Configuration
SAVE_DIR = PROJECT_ROOT / "data" / "A"
INTERVAL = 10  # seconds

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

def fetch_market_cap_symbols_net(min_market_cap: float) -> pd.DataFrame:
    """Fetch symbols directly from network, ignoring local cache."""
    print("Requesting real-time market cap data from Akshare...")
    spot = ak.stock_zh_a_spot_em()
    code_col = "代码" if "代码" in spot.columns else None
    name_col = "名称" if "名称" in spot.columns else None
    cap_col = "总市值" if "总市值" in spot.columns else None
    
    if code_col is None or cap_col is None:
        raise ValueError(f"Missing expected columns in akshare spot data: {spot.columns}")

    cap_vals = _to_numeric(spot[cap_col])
    keep_cols = [code_col, cap_col]
    if name_col is not None:
        keep_cols.insert(1, name_col)
        
    filtered = spot.loc[cap_vals >= min_market_cap, keep_cols].copy()
    filtered["market_cap"] = _to_numeric(filtered[cap_col])
    filtered = filtered.rename(columns={code_col: "symbol", name_col: "name"})
    
    if "name" not in filtered.columns:
        filtered["name"] = ""
        
    filtered = filtered.sort_values("market_cap", ascending=False).reset_index(drop=True)
    return filtered[["symbol", "name", "market_cap"]]


def get_stock_prefix(symbol):
    if symbol.startswith("6"):
        return "sh"
    elif symbol.startswith("0") or symbol.startswith("3"):
        return "sz"
    elif symbol.startswith("8") or symbol.startswith("4"):
        return "bj"
    else:
        return "sz" # Default fallback, verify if needed

def save_to_duckdb(symbol, df, db_path):
    if df.empty:
        return
        
    # Ensure correct columns and types
    # Akshare columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    # We need: date (ts), open, high, low, close, volume, turnover (rate)
    
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "换手率": "turnover"
    }
    
    # Check if '换手率' exists, sometimes it might be missing
    if "换手率" not in df.columns:
        # Try to infer or skip? For hfq, turnover might not be accurate or present?
        # Actually hfq modifies prices. Turnover rate usually remains based on real physical shares?
        # Let's keep it if available.
        df["turnover"] = 0.0
    else:
        df = df.rename(columns={"换手率": "turnover"})
        
    df = df.rename(columns=rename_map)
    
    keep_cols = ["date", "open", "high", "low", "close", "volume", "turnover"]
    # Filter columns that exist
    final_cols = [c for c in keep_cols if c in df.columns]
    data_to_save = df[final_cols].copy()
    
    # Add stock_code column for compatibility with query: "WHERE stock_code = ?"
    data_to_save["stock_code"] = symbol

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE IF NOT EXISTS daily_data AS SELECT * FROM data_to_save")
        # If table exists, replace it (full update strategy)
        con.execute("DROP TABLE IF EXISTS daily_data")
        con.execute("CREATE TABLE daily_data AS SELECT * FROM data_to_save")
    finally:
        con.close()

def download_worker():
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)
        
    print(f"Fetching symbol list (Market Cap >= {MIN_MARKET_CAP})...")
    try:
        # Use local network-only fetcher
        symbols_df = fetch_market_cap_symbols_net(MIN_MARKET_CAP)
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return

    total = len(symbols_df)
    print(f"Found {total} symbols to download.")
    
    for idx, row in symbols_df.iterrows():
        symbol = str(row["symbol"])
        prefix = get_stock_prefix(symbol)
        full_code = f"{prefix}{symbol}"
        filename = f"{full_code}.duckdb"
        file_path = SAVE_DIR / filename
        
        if file_path.exists():
            print(f"[{idx+1}/{total}] {full_code} already exists, skipping.")
            continue
            
        print(f"[{idx+1}/{total}] Downloading {full_code} ...")
        
        try:
            # Use HFQ as decided
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="hfq")
            
            if df is not None and not df.empty:
                save_to_duckdb(symbol, df, file_path)
                print(f"    Saved to {file_path}")
            else:
                print(f"    No data found for {symbol}")
                
        except Exception as e:
            print(f"    Error downloading {symbol}: {e}")
            
        time.sleep(INTERVAL)

if __name__ == "__main__":
    download_worker()
