
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import random

DATA_DIR = Path("data/A")

def debug_gap(symbol):
    try:
        path = list(DATA_DIR.glob(f"*{symbol}*.duckdb"))[0]
    except IndexError:
        print(f"File not found for {symbol}")
        return

    print(f"DEBUGGING {symbol} ({path.name})")
    con = duckdb.connect(str(path))
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    
    # Cleaning
    df = df.dropna().reset_index(drop=True)
    if len(df) > 25: df = df.iloc[:-25]
    
    # Re-implement Logic from 08_visualize_rules.py (Exact Copy)
    m_strict = (df['month_skdj_k_state'] > 0) & (df['month_skdj_d_state'] > 0) & \
               (df['month_macd_bar_state'] > 0) & (df['month_macd_dif_state'] > 0) & \
               (df['month_macd_dea_state'] > 0) & (df['month_dma_state'] > 0)
    m_loose = (df['month_macd_dea_state'] > 0) & (df['month_ama_state'] > 0)
    watch_period = m_strict | m_loose
    
    buy_opp = watch_period & (df['week_skdj_d_state'] == 2)
    sell_opp = (df['week_skdj_d_state'] == -2)
    
    force_sell_cond = (df['week_skdj_d_state'] < 0) & \
                      (df['week_macd_dif_state'] < 0) & \
                      (df['week_macd_dea_state'] < 0) & \
                      (df['week_ama_state'] < 0)

    prices = df['close'].values
    max_5w = pd.Series(prices).rolling(window=25).max().shift(-25).values
    min_5w = pd.Series(prices).rolling(window=25).min().shift(-25).values
    
    holding = False
    entry_price = 0.0
    hold_days = 0
    trade_max_p = 0.0
    trade_max_dd = 0.0
    
    # Tracking for Gap Analysis
    current_trade_rew = 0.0
    current_trade_equity_change = 0.0
    trade_start_idx = 0
    
    print(f"{'Start':<12} | {'End':<12} | {'Days':<4} | {'Real PnL':<10} | {'Total Rew':<10} | {'Gap':<10}")
    print("----------------------------------------------------------------------")
    
    total_equity_pnl = 0.0
    total_reward = 0.0

    eps = 1e-8

    for i in range(len(df) - 1):
        p_t = prices[i]
        p_next = prices[i+1]
        ts = df['ts'][i].strftime('%Y-%m-%d')
        
        # Reward Components
        term_val = 0
        if np.isfinite(max_5w[i]) and np.isfinite(min_5w[i]):
            term_val = ((max_5w[i] - p_t) - (p_t - min_5w[i])) / p_t
        term_reward = term_val * 3.0 

        step_reward = 0.0
        
        if not holding:
            if buy_opp[i]:
                # Force Buy for debug (assume 100% execution to check logic)
                # Or mimic random? Let's mimic random 0.8 to match visualization
                if random.random() < 0.8: 
                    holding = True
                    entry_price = p_t
                    hold_days = 0
                    trade_max_p = p_t
                    trade_max_dd = 0.0
                    
                    step_reward = term_reward
                    
                    # Start Trade Tracking
                    current_trade_rew = step_reward
                    trade_start_idx = i
                else:
                    step_reward = -term_reward
            else:
                step_reward = 0
        else: # Holding
            hold_days += 1
            trade_max_p = max(trade_max_p, p_next)
            
            cur_ret = (p_t / (entry_price + eps)) - 1.0
            avg_ret = cur_ret / max(5.0, hold_days)
            dd = (trade_max_p - p_t) / (trade_max_p + eps)
            trade_max_dd = max(trade_max_dd, dd)
            
            # User Specified Non-Linear Reward Formula
            if cur_ret > 0:
                holding_reward = (1 + cur_ret) ** 0.5 - 1 + (hold_days**0.33) * avg_ret - trade_max_dd
            else:
                # Assuming cur_ret is negative
                holding_reward = -(1 - cur_ret) ** 2 + 1 + (hold_days**0.33) * avg_ret - trade_max_dd
            
            should_sell = False
            
            if force_sell_cond[i]:
                should_sell = True
                step_reward = holding_reward
            elif sell_opp[i]:
                must_sell = (cur_ret > 0) and (holding_reward < 0)
                if must_sell or (random.random() < 0.2):
                    should_sell = True
                    step_reward = holding_reward
                else:
                    step_reward = term_reward # Missed
            else:
                step_reward = holding_reward
                
            current_trade_rew += step_reward
            
            if should_sell:
                holding = False
                exit_price = p_t
                real_pnl = (exit_price - entry_price) / entry_price
                
                total_equity_pnl += real_pnl
                total_reward += current_trade_rew
                
                print(f"{df['ts'][trade_start_idx].strftime('%Y-%m-%d'):<12} | {ts:<12} | {hold_days:<4} | {real_pnl:>9.2%} | {current_trade_rew:>9.2f} | {current_trade_rew - real_pnl:>9.2f}")
                
                current_trade_rew = 0.0
                
        # Non-holding rewards also contribute to total_reward
        if not holding and step_reward != 0:
             # These are missed opportunity costs, they add to Reward Sum but have 0 Equity impact
             # This is a major source of divergence!
             pass

if __name__ == "__main__":
    debug_gap("sh601857")
