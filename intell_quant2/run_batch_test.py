
import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# Import the logic directly from the file we just fixed
sys.path.append(".")
from preprocessing.visualize_rules import apply_rules_and_sim

DATA_DIR = Path("data/A")

targets = ["sh600519", "sh600036", "sh601857", "sz000002", "sh600060"]

print(f"{'Symbol':<10} | {'Equity':<8} | {'RewSum':<8} | {'Trades':<6} | {'WinRate':<8} | {'MustSell':<8}")
print("-" * 70)

for sym in targets:
    try:
        path = list(DATA_DIR.glob(f"*{sym}*.duckdb"))[0]
        con = duckdb.connect(str(path))
        df = con.execute("SELECT * FROM features ORDER BY ts").df()
        con.close()
        
        df = df.dropna().reset_index(drop=True)
        if len(df) > 25: df = df.iloc[:-25]
        
        # Run Sim
        _, _, _, actions, rewards, equity = apply_rules_and_sim(df)
        
        final_eq = equity[-1]
        rew_sum = rewards.sum()
        
        # Trades: Buy Actions (1)
        num_trades = (actions == 1).sum()
        
        # Win Rate calculation is tricky without full trade tracking, 
        # but we can look at Equity growth periods or simplify.
        # Let's count Must Sell triggers (logic: action=2 but reward<0 ?)
        # Actually action=2 is mixed (random + must). 
        # Let's just track equity direction.
        
        print(f"{sym:<10} | {final_eq:.4f}   | {rew_sum:.2f}     | {num_trades:<6} | {'N/A':<8} | {'N/A':<8}")
        
    except Exception as e:
        print(f"{sym:<10} | Error: {e}")
