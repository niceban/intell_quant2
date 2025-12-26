
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Simulate the logic from feature_generator.py and env.py

def analyze_stats(file_path):
    print(f"Analyzing {file_path}...")
    con = duckdb.connect(str(file_path), read_only=True)
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    
    if "week_skdj_d_state" not in df.columns:
        print("Missing required columns.")
        return

    # 1. State Distribution Logic
    w_d = df["week_skdj_d_state"]
    # Month logic (simplified proxy based on logic in env.py)
    # m_k>0 & m_d>0 & m_bar>0 & m_dif>0 & m_dea>0 & m_dma>0
    month_good_mask = (
        (df["month_skdj_k_state"] > 0) & 
        (df["month_skdj_d_state"] > 0) & 
        (df["month_macd_bar_state"] > 0) & 
        (df["month_macd_dif_state"] > 0) & 
        (df["month_macd_dea_state"] > 0) & 
        (df["month_dma_state"] > 0)
    )

    buy_opp = month_good_mask & (w_d == 2)
    sell_opp_strict = (w_d == -2)
    sell_opp_relaxed = (w_d <= -1) # -1 or -2
    
    total_steps = len(df)
    buy_opp_count = buy_opp.sum()
    sell_strict_count = sell_opp_strict.sum()
    sell_relaxed_count = sell_opp_relaxed.sum()
    
    print(f"Total Steps: {total_steps}")
    print(f"Buy Opportunities (Strict): {buy_opp_count} ({buy_opp_count/total_steps:.2%})")
    print(f"Sell Opportunities (Strict w_d==-2): {sell_strict_count} ({sell_strict_count/total_steps:.2%})")
    print(f"Sell Opportunities (Relaxed w_d<=-1): {sell_relaxed_count} ({sell_relaxed_count/total_steps:.2%})")
    
    # 2. Reward Inflation Simulation
    print("\n--- Reward Inflation Simulation ---")
    # Simulate a trade: Buy at step T, hold for 20 steps
    # Assume price grows 1% per step (unrealistic but good for math check)
    h_days = np.arange(1, 101) # 100 days
    entry_price = 100.0
    prices = entry_price * (1.001 ** h_days) # Slow steady growth
    
    cur_rets = (prices / entry_price) - 1.0
    avg_rets = cur_rets / np.maximum(h_days, 5.0)
    avg_ret_threshold = 0.0 # Early training
    
    # Formula: cur_ret + h_days * (avg_ret - threshold)
    rewards_linear = cur_rets + h_days * (avg_rets - avg_ret_threshold)
    rewards_sqrt = cur_rets + np.sqrt(h_days) * (avg_rets - avg_ret_threshold)
    
    print(f"--- Linear (Current) ---")
    print(f"Day 1: {rewards_linear[0]:.4f}")
    print(f"Day 100: {rewards_linear[99]:.4f} (Ratio: {rewards_linear[99]/rewards_linear[0]:.1f}x)")

    print(f"\n--- Sqrt (Proposed) ---")
    print(f"Day 1: {rewards_sqrt[0]:.4f}")
    print(f"Day 100: {rewards_sqrt[99]:.4f} (Ratio: {rewards_sqrt[99]/rewards_sqrt[0]:.1f}x)")
    
    if rewards_sqrt[99] > rewards_sqrt[0] * 10:
        print("!! ALERT: Sqrt still scales, but less aggressively !!")

if __name__ == "__main__":
    import sys
    # Find a sample file
    data_dir = Path("data/A")
    files = list(data_dir.glob("*.duckdb"))
    if files:
        analyze_stats(files[0])
    else:
        print("No duckdb files found.")
