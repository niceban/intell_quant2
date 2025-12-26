import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import random

DATA_DIR = Path("data/A")

def simulate_full_features(df):
    n = len(df)
    
    # Precompute week_id for immediate cancel logic
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'])
    
    # Use ISO calendar year * 100 + week to uniquely identify weeks
    # This matches the 'week_id' logic in the RL environment
    df['week_id'] = df['ts'].dt.isocalendar().year * 100 + df['ts'].dt.isocalendar().week
    week_ids = df['week_id'].values

    # 1. Watch Period
    m_strict = (
        (df['month_skdj_k_state'] > 0) & (df['month_skdj_d_state'] > 0) & 
        (df['month_macd_bar_state'] > 0) & (df['month_macd_dif_state'] > 0) & 
        (df['month_macd_dea_state'] > 0) & (df['month_dma_state'] > 0)
    )
    m_loose = (df['month_macd_dea_state'] > 0) & (df['month_ama_state'] > 0)
    watch_period = m_strict | m_loose
    
    # 2. Opportunities
    buy_opp = watch_period & (df['week_skdj_d_state'] == 2)
    buy_opp_vals = buy_opp.values
    
    # Sell Opp Expansion (Sniper Logic)
    s_d = (df['week_skdj_d_state'] == -2)
    s_dea = (df['week_macd_dea_state'] == -2)
    s_water = (df['week_macd_dea_state'] < 0) & (df['week_macd_dif_state'] < 0) & (df['week_macd_bar_state'] < 0)
    sell_opp = s_d | s_dea | s_water
    sell_opp_vals = sell_opp.values
    
    # Force Sell (6-line negative)
    # Note: force_sell is just a mandatory sell, fits into Sell Exec category
    force_sell_cond = (
        (df['week_skdj_d_state'] < 0) & (df['week_skdj_k_state'] < 0) & 
        (df['week_macd_dif_state'] < 0) & (df['week_macd_dea_state'] < 0) & (df['week_macd_bar_state'] < 0) & 
        (df['week_ama_state'] < 0)
    )
    force_sell_vals = force_sell_cond.values

    prices = df['close'].values
    max_5w = pd.Series(prices).rolling(window=25).max().shift(-25).values
    min_5w = pd.Series(prices).rolling(window=25).min().shift(-25).values
    
    holding = False
    entry_price = 0.0
    entry_week_id = -1
    hold_days = 0
    trade_max_p = 0.0
    trade_max_dd_val = 0.0
    
    buy_exec_list = []
    buy_miss_list = []
    
    # holding_list will track the raw_holding_rew (State Feature), though RL reward is 0
    state_holding_list = [] 
    
    sell_exec_list = []
    sell_miss_list = []
    
    d1_list = []
    d2_list = []
    
    eps = 1e-8
    
    prev_rew_scaled = 0.0
    prev_d1_unscaled = 0.0

    for i in range(n - 1):
        p_t = prices[i]
        p_next = prices[i+1]
        if p_t < eps: continue
        
        term_val = 0
        if np.isfinite(max_5w[i]) and np.isfinite(min_5w[i]):
            term_val = ((max_5w[i] - p_t) - (p_t - min_5w[i])) / p_t
        
        # Env Clamping (-10, 10)
        term_val = max(-10.0, min(10.0, term_val))
        term_reward = term_val 

        current_rew = 0
        
        if not holding:
            prev_rew_scaled = 0.0
            prev_d1_unscaled = 0.0
            
            if buy_opp_vals[i]:
                # Random Mode: 0.8 Buy Prob
                if random.random() < 0.8:
                    holding = True
                    # Buy Cost: Entry Price * 1.005
                    entry_price = p_t * 1.005
                    entry_week_id = week_ids[i]
                    hold_days = 0
                    trade_max_p = p_t
                    trade_max_dd_val = 0.0
                    
                    # Buy Exec: Reward = Term
                    current_rew = term_reward
                    buy_exec_list.append(current_rew * 5.0) # Scale x5
                else:
                    # Buy Miss: Reward = Term * -0.5
                    current_rew = -term_reward * 0.5
                    buy_miss_list.append(current_rew * 5.0) # Scale x5
            else:
                current_rew = 0
        else:
            hold_days += 1
            trade_max_p = max(trade_max_p, p_next)
            
            # Net Value Logic
            liquid_price = p_t * 0.995
            cur_ret = (liquid_price / (entry_price + eps)) - 1.0
            
            avg_ret = cur_ret / max(5.0, hold_days)
            dd = (trade_max_p - p_t) / (trade_max_p + eps)
            trade_max_dd_val = max(trade_max_dd_val, dd)
            
            # Raw Holding Reward (For State & Diffs)
            if cur_ret > 0:
                raw_holding_rew = (1 + cur_ret) ** 0.5 - 1 + (hold_days**0.33) * avg_ret - trade_max_dd_val
            else:
                raw_holding_rew = -(1 - cur_ret) ** 2 + 1 + (hold_days**0.33) * avg_ret - trade_max_dd_val
            
            state_holding_list.append(raw_holding_rew)
            
            # --- Feature Diff Logic (Matching Env) ---
            # Env stores LastReward as Raw * 5.0
            curr_rew_scaled = raw_holding_rew * 5.0
            
            # d1_unscaled = LastRew_t - LastRew_{t-1} (Both are x5 scaled)
            # So d1_unscaled is RawDiff * 5.0
            d1_unscaled = curr_rew_scaled - prev_rew_scaled
            
            # d2_unscaled = d1_unscaled_{t-1}
            # This is "m_1 - m_2" in Env
            d2_unscaled = prev_d1_unscaled
            
            # Final Feature: d1_scaled = d1_unscaled * 5.0 (Total x25)
            d1_final = d1_unscaled * 5.0
            
            # Final Feature: d2_scaled = d2_unscaled * 5.0 (Total x25)
            d2_final = d2_unscaled * 5.0
            
            if hold_days > 1: d1_list.append(d1_final)
            if hold_days > 2: d2_list.append(d2_final)
            
            prev_rew_scaled = curr_rew_scaled
            prev_d1_unscaled = d1_unscaled
            
            should_sell = False
            
            # Immediate Cancel Check (Same Week & No Buy Signal)
            curr_week = week_ids[i]
            is_same_week = (curr_week == entry_week_id)
            # If in the same week, and the buy signal is gone, we must sell
            immediate_cancel = is_same_week and (not buy_opp_vals[i])

            if force_sell_vals[i] or immediate_cancel:
                should_sell = True
                # Force Sell is an Executed Sell
                # Reward = Term + Raw Holding
                current_rew = term_reward + raw_holding_rew
                sell_exec_list.append(current_rew * 5.0) # Scale x5
                
            elif sell_opp_vals[i]:
                # Random Mode: 0.2 Sell Prob
                if random.random() < 0.2:
                    should_sell = True
                    current_rew = term_reward + raw_holding_rew
                    sell_exec_list.append(current_rew * 5.0) # Scale x5
                else:
                    # Sell Miss: Reward = Term * 2.0
                    current_rew = term_reward * 2.0
                    sell_miss_list.append(current_rew * 5.0) # Scale x5
                
            if should_sell:
                holding = False
                hold_days = 0
                trade_max_p = 0.0
                trade_max_dd_val = 0.0
                entry_week_id = -1
            
            # Note: If holding and not selling, RL Reward is 0.0 (Implicit)
            
    return {
        "buy_exec": buy_exec_list,
        "buy_miss": buy_miss_list,
        "state_holding": state_holding_list, # Not scaled, just raw values for ref
        "sell_exec": sell_exec_list,
        "sell_miss": sell_miss_list,
        "d1_rew": d1_list,
        "d2_rew": d2_list
    }

def main():
    all_files = list(DATA_DIR.glob("*.duckdb"))
    if not all_files:
        print("No Data Found.")
        return
        
    samples = random.sample(all_files, min(100, len(all_files)))
    
    print(f"Analyzing REWARDS (x5) & DIFFS (x25) across {len(samples)} files (Sniper Logic)...")
    
    aggregated = {
        "buy_exec": [], "buy_miss": [], "state_holding": [], "sell_exec": [], "sell_miss": [],
        "d1_rew": [], "d2_rew": []
    }
    
    for p in samples:
        try:
            con = duckdb.connect(str(p))
            df = con.execute("SELECT * FROM features ORDER BY ts").df()
            con.close()
            df = df.dropna().reset_index(drop=True)
            if len(df) > 25: df = df.iloc[:-25].reset_index(drop=True)
            
            res = simulate_full_features(df)
            for k in aggregated:
                aggregated[k].extend(res[k])
        except Exception as e: pass
            
    print("-" * 100)
    print(f"{ 'Type':<12} | { 'Count':<8} | { 'AM':<8} | Deciles (0% - 100%)")
    print("-" * 100)
    
    for k, v in aggregated.items():
        if not v:
            print(f"{k:<12} | 0        | N/A")
            continue
            
        arr = np.array(v)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0: continue
        
        abs_mean = np.mean(np.abs(arr))
        deciles = np.percentile(arr, np.arange(0, 101, 10))
        decile_str = " | ".join([f"{val:.3f}" for val in deciles])
        
        print(f"{k:<12} | {len(arr):<8} | {abs_mean:.4f}   | {decile_str}")

if __name__ == "__main__":
    main()