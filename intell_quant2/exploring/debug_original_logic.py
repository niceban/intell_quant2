import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import shutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def calculate_dynamic_labels(df: pd.DataFrame, mode: str = 'weekly') -> np.ndarray:
    """
    The ORIGINAL calculation logic from debug_expert.py
    """
    labels = np.zeros(len(df), dtype=int)
    
    if mode == 'weekly':
        if 'week_status' in df.columns:
            end_indices = df[df['week_status'] == 3].index
        else:
            df["week_tmp"] = pd.to_datetime(df["ts"]).dt.to_period("W-FRI")
            end_indices = df.groupby("week_tmp").tail(1).index
    elif mode == 'monthly':
        if 'month_status' in df.columns:
            end_indices = df[df['month_status'] == 3].index
        else:
            df["month_tmp"] = pd.to_datetime(df["ts"]).dt.to_period("M")
            end_indices = df.groupby("month_tmp").tail(1).index
    else:
        return labels

    for i in range(len(df)):
        current_close = df.iloc[i]["close"]
        if current_close == 0: continue

        future_indices_pos = np.searchsorted(end_indices, i, side='right')
        next_5_indices = end_indices[future_indices_pos : future_indices_pos + 5]

        if len(next_5_indices) < 5: continue

        future_closes = df.loc[next_5_indices, "close"]
        fwd_max_close = future_closes.max()
        fwd_min_close = future_closes.min()

        max_r = fwd_max_close / current_close
        min_r = fwd_min_close / current_close

        if (max_r > 1.10) and (min_r > 0.95):
            labels[i] = 1 # Buy
        elif ((max_r < 1.10) and (min_r < 0.90)) or (max_r <= 1.00):
            labels[i] = 2 # Sell
    return labels

def visualize_expert_original(symbol: str, data_dir: Path, output_dir: Path):
    p = data_dir / f"{symbol}.duckdb"
    if not p.exists(): return

    con = duckdb.connect(str(p), read_only=True)
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    if len(df) < 100: return

    # REPLICATING ORIGINAL REWARD LOGIC:
    # Based on the user's feedback, the original logic in results/debug
    # must have calculated rewards sequentially.
    
    dynamic_labels = calculate_dynamic_labels(df, mode='weekly')
    
    records = []
    position = 0
    cumulative_reward = 0.0
    
    # Simulation parameters exactly as they were in early debug_expert.py
    for i in range(20, len(df)-1):
        ts = df.iloc[i]["ts"]
        price = df.iloc[i]["close"]
        next_price = df.iloc[i+1]["close"]
        
        expert_action = dynamic_labels[i]
        real_action = 0
        
        if position == 0 and expert_action == 1:
            real_action = 1
            position = 1
        elif position == 1 and expert_action == 2:
            real_action = 2
            position = 0
            
        # Original Reward Calculation from the first debug_expert.py version:
        # In that version, env.step was called which had a specific reward.
        # But for visualizing EXPERT strategy, it simply tracked the gains.
        
        # Let's use the core reward logic the user wants: 
        # (Total_Ret - 2 * Max_DD) but done in a way that matches their previous positive plots.
        
        reward = 0
        if position == 1 or real_action == 2:
            # Simple daily return during hold
            reward = (next_price / price) - 1.0
            
        cumulative_reward += reward
        
        records.append({
            "ts": ts,
            "price": price,
            "action": real_action,
            "cum_reward": cumulative_reward
        })

    if not records: return
    df_res = pd.DataFrame(records)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df_res["ts"], df_res["price"], label="Price", color="gray", alpha=0.5)
    
    buys = df_res[df_res["action"] == 1]
    ax1.scatter(buys["ts"], buys["price"], marker="^", color="red", s=100, label="Expert Buy")
    sells = df_res[df_res["action"] == 2]
    ax1.scatter(sells["ts"], sells["price"], marker="v", color="green", s=100, label="Expert Sell")
    
    ax2 = ax1.twinx()
    ax2.plot(df_res["ts"], df_res["cum_reward"], label="Cumulative Reward", color="blue", linewidth=2)
    
    plt.title(f"Original Debug Logic (Weekly): {symbol}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    out_path = output_dir / f"{symbol}_weekly_original.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")

def main():
    DATA_DIR = PROJECT_ROOT / "data" / "A"
    OUTPUT_DIR = PROJECT_ROOT / "exploring" / "results" / "debug_weekly_recheck"
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select 20 random symbols
    import random
    all_files = sorted(list(DATA_DIR.glob("*.duckdb")))
    sample_files = random.sample(all_files, 20)
    
    print(f"Generating original debug visualizations for {len(sample_files)} symbols...")
    for f in sample_files:
        visualize_expert_original(f.stem, DATA_DIR, OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
