import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploring.rl.env import TradingEnv

def run_comparison(symbol="sh600000"):
    DATA_DIR = PROJECT_ROOT / "data" / "A"
    
    # 1. Initialize REAL Env
    env = TradingEnv(DATA_DIR, load_symbols=[symbol])
    env._load_symbol_data(symbol)
    
    # Get the FULL pre-processed dataframe from cache
    full_df = env.cache[symbol].copy()
    full_df['ts'] = pd.to_datetime(full_df['ts'])
    
    # Filter for 2010-2020 while keeping ALL columns (including flat_reward)
    mask = (full_df['ts'] >= '2010-01-01') & (full_df['ts'] <= '2020-12-31')
    compare_df = full_df[mask].reset_index(drop=True)
    
    # Inject filtered df back to env
    env.current_df = compare_df
    env.current_symbol = symbol
    env.position = 0
    env.current_step = 0
    
    print(f"Comparing Rewards for {symbol} (2010-2020), Data Length: {len(compare_df)}")
    
    env_rewards = []
    viz_rewards = []
    
    # Viz state variables
    viz_pos = 0
    viz_entry_p = 0.0
    viz_max_p = 0.0
    viz_max_dd = 0.0
    viz_prev_metric = 0.0
    
    labels = env.labels_cache[symbol] # Global index labels
    # We need to map global labels back to our filtered local index
    # Labels are computed on original indices. Let's just find them by TS.
    local_labels = compare_df.merge(pd.DataFrame({'ts': full_df['ts'], 'label': labels}), on='ts')['label'].values
    
    diffs = []
    
    for i in range(len(compare_df) - 10):
        # A. ENV Step
        expert_action = local_labels[i]
        action = 0
        if env.position == 0 and expert_action == 1: action = 1
        elif env.position == 1 and expert_action == 2: action = 2
        
        _, e_rew, _, _ = env.step(action)
        env_rewards.append(e_rew)
        
        # B. VIZ Manual (Mirroring debug_original_final.py)
        price = compare_df.iloc[i]['close']
        next_price = compare_df.iloc[i+1]['close']
        v_rew = 0.0
        
        if viz_pos == 0:
            if expert_action == 1:
                viz_pos = 1
                viz_entry_p = price
                viz_max_p = price
                viz_max_dd = 0.0
                viz_prev_metric = 0.0
                v_rew = 0.0
            else:
                v_rew = float(compare_df.iloc[i]['flat_reward'])
        else:
            # Holding calculation
            cur_ret = (next_price / viz_entry_p) - 1.0
            viz_max_p = max(viz_max_p, next_price)
            cur_dd = abs(next_price / viz_max_p - 1.0)
            viz_max_dd = max(viz_max_dd, cur_dd)
            
            cur_metric = cur_ret - (viz_max_dd * 2.0)
            v_rew = cur_metric - viz_prev_metric
            viz_prev_metric = cur_metric
            
            if expert_action == 2:
                viz_pos = 0
        
        viz_rewards.append(v_rew)
        
        if abs(e_rew - v_rew) > 1e-7:
            diffs.append((i, compare_df.iloc[i]['ts'], e_rew, v_rew))

    print(f"Total steps compared: {len(env_rewards)}")
    print(f"Number of steps with mismatch: {len(diffs)}")
    
    if diffs:
        print("\nFirst 5 mismatches:")
        for d in diffs[:5]:
            print(f"Step {d[0]} ({d[1]}): Env={d[2]:.6f}, Viz={d[3]:.6f}, Diff={d[2]-d[3]:.6f}")
    
    print(f"\nTotal Env Reward Sum: {sum(env_rewards):.4f}")
    print(f"Total Viz Reward Sum: {sum(viz_rewards):.4f}")

if __name__ == "__main__":
    run_comparison()