
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

def calculate_env_style_data(df: pd.DataFrame):
    """
    Exactly replicate the calculation logic in env.py _load_symbol_data
    """
    labels = np.zeros(len(df), dtype=int)
    flat_rewards = np.zeros(len(df), dtype=np.float32)
    
    if 'week_status' in df.columns:
        week_ends = df[df['week_status'] == 3].index.to_numpy()
    else:
        df['ts_dt'] = pd.to_datetime(df['ts'])
        week_ends = df.groupby(df['ts_dt'].dt.to_period('W-FRI')).tail(1).index.to_numpy()
            
    for i in range(len(df)):
        current_close = df.iloc[i]["close"]
        if current_close == 0: continue
        
        future_pos = np.searchsorted(week_ends, i, side='right')
        next_5_indices = week_ends[future_pos : future_pos + 5]
        
        if len(next_5_indices) < 5: continue
        
        future_closes = df.loc[next_5_indices, "close"].values
        max_r = future_closes.max() / current_close
        min_r = future_closes.min() / current_close
        
        # Label logic
        if (max_r > 1.10) and (min_r > 0.95):
            labels[i] = 1 # Buy
        elif ((max_r < 1.10) and (min_r < 0.90)) or (max_r <= 1.00):
            labels[i] = 2 # Sell
            
        # Flat reward logic
        flat_rewards[i] = (min_r - 1.0) * -0.05
        
    df['expert_label'] = labels
    df['flat_reward'] = flat_rewards
    return df

def visualize_env_aligned(symbol: str, data_dir: Path, output_dir: Path):
    p = data_dir / f"{symbol}.duckdb"
    if not p.exists(): return
    con = duckdb.connect(str(p), read_only=True)
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    if len(df) < 100: return

    df = calculate_env_style_data(df)
    
    records = []
    position = 0
    cum_reward = 0.0
    
    # Env state variables
    entry_price = 0.0
    trade_max_price = 0.0
    trade_max_dd = 0.0
    prev_trade_metric = 0.0
    
    # Simulation using EXACT env.step logic
    for i in range(20, len(df)-1):
        ts = df.iloc[i]['ts']
        price = df.iloc[i]['close']
        next_price = df.iloc[i+1]['close'] # What matters for reward
        
        expert_label = df.iloc[i]['expert_label']
        action = 0
        step_reward = 0.0
        
        if position == 0:
            if expert_label == 1:
                # Buy action triggered
                action = 1
                position = 1
                entry_price = price
                trade_max_price = price
                trade_max_dd = 0.0
                prev_trade_metric = 0.0
                step_reward = 0.0
            else:
                # Hold Flat
                step_reward = df.iloc[i]['flat_reward']
        else: # position == 1
            # Real-time Update (Delta Reward)
            current_ret = (next_price / entry_price) - 1.0
            trade_max_price = max(trade_max_price, next_price)
            current_dd = abs(next_price / trade_max_price - 1.0)
            trade_max_dd = max(trade_max_dd, current_dd)
            
            current_metric = current_ret - (trade_max_dd * 2.0)
            step_reward = current_metric - prev_trade_metric
            prev_trade_metric = current_metric
            
            if expert_label == 2:
                # Sell action triggered
                action = 2
                position = 0
        
        cum_reward += step_reward
        records.append({
            "ts": ts,
            "price": price,
            "action": action,
            "reward": step_reward,
            "cum_reward": cum_reward,
            "metric": prev_trade_metric if position == 1 or action == 2 else 0
        })

    if not records: return
    df_res = pd.DataFrame(records)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top plot: Price and Actions
    ax1.plot(df_res['ts'], df_res['price'], color='gray', alpha=0.4, label='Price')
    buys = df_res[df_res["action"] == 1]
    ax1.scatter(buys['ts'], buys['price'], marker='^', color='red', s=100, label='Expert Buy', zorder=5)
    sells = df_res[df_res["action"] == 2]
    ax1.scatter(sells['ts'], sells['price'], marker='v', color='green', s=100, label='Expert Sell', zorder=5)
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    ax1.set_title(f"ENV ALIGNED DEBUG: {symbol} (Full Logic Simulation)")
    
    # Bottom plot: Cumulative Reward
    ax2.plot(df_res['ts'], df_res['cum_reward'], color='blue', linewidth=2, label='Cumulative Reward (Env Aligned)')
    ax2.axhline(0, color='black', alpha=0.3, linestyle='--')
    ax2.set_ylabel("Reward")
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    out_path = output_dir / f"{symbol}_env_aligned.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Generated {out_path}")

def main():
    DATA_DIR = PROJECT_ROOT / "data" / "A"
    OUTPUT_DIR = PROJECT_ROOT / "exploring" / "results" / "debug_env_aligned"
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    import random
    all_files = sorted(list(DATA_DIR.glob("*.duckdb")))
    samples = random.sample(all_files, 20)
    
    print(f"Generating ENV-ALIGNED debug visualizations for {len(samples)} symbols...")
    for f in samples:
        visualize_env_aligned(f.stem, DATA_DIR, OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
