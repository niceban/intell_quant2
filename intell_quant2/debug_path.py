
import sys
import pandas as pd
from pathlib import Path
import duckdb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import directly from the script to test its logic
from main.run_backtest import _list_local_symbols, _calc_market_cap_from_local, LOCAL_DUCKDB_DIR, MIN_MARKET_CAP

print(f"DEBUG: LOCAL_DUCKDB_DIR in script is: {LOCAL_DUCKDB_DIR}")
print(f"DEBUG: Checking if it exists: {LOCAL_DUCKDB_DIR.exists()}")

if not LOCAL_DUCKDB_DIR.exists():
    print("ERROR: Directory does not exist!")
    sys.exit(1)

# 1. Test Listing
symbols = _list_local_symbols(LOCAL_DUCKDB_DIR)
print(f"DEBUG: Found {len(symbols)} symbols in directory.")
if len(symbols) < 10:
    print(f"Sample symbols: {symbols}")
else:
    print(f"Sample symbols: {symbols[:5]}")

# 2. Test Market Cap Calc for a few
valid_count = 0
high_cap_count = 0

print("\nDEBUG: Testing Market Cap Calculation...")
for sym in symbols[:20]: # Check first 20
    cap = _calc_market_cap_from_local(sym)
    if cap is None:
        print(f"  [FAIL] {sym}: Cap is None (Turnover missing or 0?)")
    else:
        valid_count += 1
        is_high = cap >= MIN_MARKET_CAP
        if is_high:
            high_cap_count += 1
        print(f"  [OK]   {sym}: Cap = {cap/1e9:.2f}B (>=20B: {is_high})")

print(f"\nSummary:")
print(f"Total Checked: {min(len(symbols), 20)}")
print(f"Valid Cap: {valid_count}")
print(f"High Cap (>=20B): {high_cap_count}")

