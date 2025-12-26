import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def calculate_weekly_expert_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(df), dtype=int)
    
    # Identify week ends
    if 'week_status' in df.columns:
        end_indices = df[df['week_status'] == 3].index
    else:
        df["week"] = pd.to_datetime(df["ts"]).dt.to_period("W-FRI")
        end_indices = df.groupby("week").tail(1).index
            
    for i in range(len(df)):
        current_close = df.iloc[i]["close"]
        if current_close == 0: continue

        # Look at next 5 week-end nodes
        future_indices_pos = np.searchsorted(end_indices, i, side='right')
        next_5_indices = end_indices[future_indices_pos : future_indices_pos + 5]

        if len(next_5_indices) < 5: continue

        future_closes = df.loc[next_5_indices, "close"]
        max_r = future_closes.max() / current_close
        min_r = future_closes.min() / current_close

        if (max_r > 1.10) and (min_r > 0.95):
            labels[i] = 1 # Buy
        elif ((max_r < 1.10) and (min_r < 0.90)) or (max_r <= 1.00):
            labels[i] = 2 # Sell
    return labels

def visualize_debug_logic(symbol: str, data_dir: Path, output_dir: Path):
    print(f"Generating debug visualization for {symbol}...")
    p = data_dir / f"{symbol}.duckdb"
    if not p.exists(): return

    con = duckdb.connect(str(p), read_only=True)
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    
    if len(df) < 100: return

    labels = calculate_weekly_expert_labels(df)
    
    # Simple simulation for visualization
    records = []
    pos = 0
    cum_reward = 0.0
    entry_price = 0.0
    
    for i in range(20, len(df)-1):
        ts = df.iloc[i]["ts"]
        price = df.iloc[i]["close"]
        next_price = df.iloc[i+1]["close"]
        action = 0
        step_reward = 0
        
        if pos == 0:
            if labels[i] == 1:
                pos = 1
                entry_price = price
                action = 1
        elif pos == 1:
            # While holding, reward is the price change
            step_reward = (next_price / price) - 1.0
            if labels[i] == 2:
                pos = 0
                action = 2
                # In the original debug viz, the drawdown penalty might not have been day-by-day
                # Let's keep it simple: just the cumulative return for now to see if it matches "results/debug"
        
        cum_reward += step_reward
        records.append({
            "ts": ts,
            "price": price,
            "action": action,
            "cum_reward": cum_reward
        })

    df_res = pd.DataFrame(records)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df_res["ts"], df_res["price"], color="gray", alpha=0.5, label="Price")
    
    buys = df_res[df_res["action"] == 1]
    ax1.scatter(buys["ts"], buys["price"], marker="^", color="red", s=80, label="Buy")
    sells = df_res[df_res["action"] == 2]
    ax1.scatter(sells["ts"], sells["price"], marker="v", color="green", s=80, label="Sell")
    
    ax2 = ax1.twinx()
    ax2.plot(df_res["ts"], df_res["cum_reward"], color="blue", label="Cumulative Reward")
    
    plt.title(f"Weekly Debug Logic Check: {symbol}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    out_path = output_dir / f"debug_{symbol}_weekly.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved to {out_path}")

def main():
    DATA_DIR = PROJECT_ROOT / "data" / "A"
    OUTPUT_DIR = PROJECT_ROOT / "exploring" / "results" / "debug_recheck"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    import random
    all_files = sorted(list(DATA_DIR.glob("*.duckdb")))
    if not all_files:
        print("No data files found!")
        return
        
    sample_files = random.sample(all_files, min(20, len(all_files)))
    test_symbols = [f.stem for f in sample_files]
    
    print(f"Generating debug visualizations for {len(test_symbols)} symbols...")
    for sym in test_symbols:
        visualize_debug_logic(sym, DATA_DIR, OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
