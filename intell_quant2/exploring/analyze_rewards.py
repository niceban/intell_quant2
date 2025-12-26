
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import random

def analyze_real_data(data_dir, num_samples=50):
    all_files = list(Path(data_dir).glob("*.duckdb"))
    samples = random.sample(all_files, min(num_samples, len(all_files)))
    
    all_trade_metrics = []
    daily_rewards = []
    
    for p in samples:
        con = duckdb.connect(str(p), read_only=True)
        # We need close, ts and the features to identify week ends
        df = con.execute("SELECT ts, close FROM features ORDER BY ts").df()
        # Mocking the week_end indices logic from env.py
        df['ts_dt'] = pd.to_datetime(df['ts'])
        week_ends = df.groupby(df['ts_dt'].dt.to_period('W-FRI')).tail(1).index.to_numpy()
        con.close()
        
        if len(df) < 100: continue
        
        # 1. Calculate Expert Labels (Strict Weekly)
        labels = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            curr_price = df.iloc[i]['close']
            if curr_price <= 0: continue
            future_pos = np.searchsorted(week_ends, i, side='right')
            next_5 = week_ends[future_pos : future_pos + 5]
            if len(next_5) < 5: continue
            fwd_closes = df.loc[next_5, 'close'].values
            max_r = fwd_closes.max() / curr_price
            min_r = fwd_closes.min() / curr_price
            if (max_r > 1.10) and (min_r > 0.95): labels[i] = 1
            elif ((max_r < 1.10) and (min_r < 0.90)) or (max_r <= 1.00): labels[i] = 2

        # 2. Simulate Trades and Collect Stats
        pos = 0
        entry_price = 0.0
        hold_days = 0
        trade_max_p = 0.0
        trade_max_dd = 0.0
        
        for i in range(len(df)):
            price = df.iloc[i]['close']
            if pos == 0:
                if labels[i] == 1:
                    pos = 1
                    entry_price = price
                    hold_days = 0
                    trade_max_p = price
                    trade_max_dd = 0.0
            else:
                hold_days += 1
                trade_max_p = max(trade_max_p, price)
                curr_dd = abs(price / trade_max_p - 1.0)
                trade_max_dd = max(trade_max_dd, curr_dd)
                
                cum_ret = (price / entry_price) - 1.0
                avg_ret = cum_ret / max(5, hold_days) # Aligned with smoothed logic
                
                # Reward formula
                reward = (cum_ret + 3.0 * avg_ret - trade_max_dd) * 10.0
                daily_rewards.append(reward)
                
                if labels[i] == 2 or i == len(df)-1:
                    all_trade_metrics.append({
                        'symbol': p.stem,
                        'ret': cum_ret,
                        'days': hold_days,
                        'max_dd': trade_max_dd,
                        'total_reward': reward # This is just the last step reward for context
                    })
                    pos = 0

    if not daily_rewards:
        print("No trades found in samples.")
        return

    # 3. Report Stats
    print(f"--- Real Data Analysis Results ({len(samples)} symbols) ---")
    print(f"Total Completed Trades: {len(all_trade_metrics)}")
    
    df_trades = pd.DataFrame(all_trade_metrics)
    print("\nTrade Statistics:")
    print(df_trades[['ret', 'days', 'max_dd']].describe())
    
    print("\nDaily Reward (Hold Phase) Statistics:")
    print(pd.Series(daily_rewards).describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

if __name__ == "__main__":
    analyze_real_data("data/A")
