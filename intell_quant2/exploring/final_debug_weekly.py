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

def calculate_weekly_labels(df: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(df), dtype=int)
    # Identify Fridays or last trading days of the week
    df['ts_dt'] = pd.to_datetime(df['ts'])
    df['week_key'] = df['ts_dt'].dt.to_period('W-FRI')
    end_indices = df.groupby('week_key').tail(1).index.to_numpy()
    
    for i in range(len(df)):
        curr_price = df.iloc[i]['close']
        if curr_price <= 0: continue
        
        # Look for next 5 week-end markers
        future_pos = np.searchsorted(end_indices, i, side='right')
        next_5 = end_indices[future_pos : future_pos + 5]
        
        if len(next_5) < 5: continue
        
        fwd_closes = df.loc[next_5, 'close'].values
        max_r = fwd_closes.max() / curr_price
        min_r = fwd_closes.min() / curr_price
        
        if max_r > 1.10 and min_r > 0.95:
            labels[i] = 1 # Buy
        elif (max_r < 1.10 and min_r < 0.90) or max_r <= 1.00:
            labels[i] = 2 # Sell
    return labels

def run_viz(symbol: str, data_dir: Path, output_dir: Path):
    p = data_dir / f"{symbol}.duckdb"
    if not p.exists(): return
    con = duckdb.connect(str(p), read_only=True)
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    if len(df) < 100: return

    labels = calculate_weekly_labels(df)
    
    records = []
    pos = 0
    cum_reward = 0.0
    
    # Track metrics for the final "Metric" calculation
    entry_price = 0.0
    trade_max_p = 0.0
    trade_max_dd = 0.0
    
    for i in range(20, len(df)-1):
        price = df.iloc[i]['close']
        next_price = df.iloc[i+1]['close']
        action = 0
        step_reward = 0
        
        if pos == 0:
            if labels[i] == 1:
                pos = 1
                entry_price = price
                trade_max_p = price
                trade_max_dd = 0.0
                action = 1
            else:
                # FOMO Reward logic aligned with user request
                # (min_fwd_5w / curr - 1) * -0.05
                # Using the same logic as label calculation for fwd min
                end_indices = df.groupby(pd.to_datetime(df['ts']).dt.to_period('W-FRI')).tail(1).index.to_numpy()
                future_pos = np.searchsorted(end_indices, i, side='right')
                next_5 = end_indices[future_pos : future_pos + 5]
                if len(next_5) == 5:
                    min_fwd = df.loc[next_5, 'close'].min()
                    step_reward = (min_fwd / price - 1.0) * -0.05
        
        elif pos == 1:
            # Holding reward: simple price change to match "positive" viz
            step_reward = (next_price / price) - 1.0
            
            # Track DD for the final penalty at exit
            trade_max_p = max(trade_max_p, price)
            curr_dd = abs(price / trade_max_p - 1.0)
            trade_max_dd = max(trade_max_dd, curr_dd)
            
            if labels[i] == 2:
                pos = 0
                action = 2
                # Apply the 2*MaxDD penalty ONCE at the exit point 
                # This keeps the curve rising during the trade but reflects the cost at the end
                step_reward -= (trade_max_dd * 2.0)
        
        cum_reward += step_reward
        records.append({
            "ts": df.iloc[i]['ts'],
            "price": price,
            "action": action,
            "cum_reward": cum_reward
        })

    if not records: return
    df_res = pd.DataFrame(records)
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.plot(df_res['ts'], df_res['price'], color='gray', alpha=0.4, label='Price')
    
    buys = df_res[df_res['action'] == 1]
    ax1.scatter(buys['ts'], buys['price'], marker='^', color='red', s=100, label='Buy')
    sells = df_res[df_res['action'] == 2]
    ax1.scatter(sells['ts'], sells['price'], marker='v', color='green', s=100, label='Sell')
    
    ax2 = ax1.twinx()
    ax2.plot(df_res['ts'], df_res['cum_reward'], color='blue', linewidth=2, label='Cumulative Reward')
    ax2.axhline(0, color='black', alpha=0.2)
    
    plt.title(f"Weekly Original Logic (Corrected Reward): {symbol}")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    out_path = output_dir / f"{symbol}_weekly_debug.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Generated {out_path}")

def main():
    DATA_DIR = PROJECT_ROOT / "data" / "A"
    OUTPUT_DIR = PROJECT_ROOT / "exploring" / "results" / "debug_final"
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    import random
    all_files = sorted(list(DATA_DIR.glob("*.duckdb")))
    samples = random.sample(all_files, 20)
    
    for f in samples:
        run_viz(f.stem, DATA_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
