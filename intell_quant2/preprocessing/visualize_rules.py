import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys

# 配置
DATA_DIR = Path("data/A")
OUTPUT_DIR = Path("outputs/visualize_rules")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_rules_and_sim(df):
    n = len(df)
    # 1. Define Watch Period
    m_strict = (df['month_skdj_k_state'] > 0) & (df['month_skdj_d_state'] > 0) & \
               (df['month_macd_bar_state'] > 0) & (df['month_macd_dif_state'] > 0) & \
               (df['month_macd_dea_state'] > 0) & (df['month_dma_state'] > 0)
    m_loose = (df['month_macd_dea_state'] > 0) & (df['month_ama_state'] > 0)
    watch_period = m_strict | m_loose
    
    # 2. Opportunities
    buy_opp = watch_period & (df['week_skdj_d_state'] == 2)
    
    # Sell Opp Expansion
    s_d = (df['week_skdj_d_state'] == -2)
    s_dea = (df['week_macd_dea_state'] == -2)
    s_water = (df['week_macd_dea_state'] < 0) & (df['week_macd_dif_state'] < 0) & (df['week_macd_bar_state'] < 0)
    sell_opp = s_d | s_dea | s_water
    
    # Force Sell Condition
    # 6-line negative: skdj_d, skdj_k, macd_dif, macd_dea, macd_bar, ama
    force_sell_cond = (df['week_skdj_d_state'] < 0) & (df['week_skdj_k_state'] < 0) & \
                      (df['week_macd_dif_state'] < 0) & (df['week_macd_dea_state'] < 0) & (df['week_macd_bar_state'] < 0) & \
                      (df['week_ama_state'] < 0)

    # 3. Future Extremes
    prices = df['close'].values
    max_5w = pd.Series(prices).rolling(window=25).max().shift(-25).values
    min_5w = pd.Series(prices).rolling(window=25).min().shift(-25).values
    
    # 4. Simulation
    holding = False
    entry_price = 0.0
    hold_days = 0
    trade_max_p = 0.0
    trade_max_dd = 0.0
    
    rewards = np.zeros(n)
    actions = np.zeros(n) # 0:None, 1:Buy, 2:Sell, 3:Force
    equity_curve = np.zeros(n)
    current_capital = 1.0
    equity_curve[0] = current_capital
    
    eps = 1e-8

    for i in range(n - 1):
        p_t = prices[i]
        p_next = prices[i+1]
        
        # Equity Update (Continuous)
        if holding:
            change = (p_next - p_t) / (p_t + eps)
            current_capital *= (1.0 + change)
        
        equity_curve[i+1] = current_capital
        
        if p_t < eps: continue
        
        # Reward Components
        term_val = 0
        if np.isfinite(max_5w[i]) and np.isfinite(min_5w[i]):
            term_val = ((max_5w[i] - p_t) - (p_t - min_5w[i])) / p_t
            
        term_reward = term_val 

        if not holding:
            if buy_opp[i]:
                if random.random() < 0.8: # 80% Buy
                    holding = True
                    # Buy Cost: Entry Price is 0.5% higher
                    entry_price = p_t * 1.005
                    # Equity Impact: Immediate 0.5% loss on buy
                    current_capital *= 0.995 
                    equity_curve[i] = current_capital # Update current step equity
                    
                    hold_days = 0
                    trade_max_p = p_t
                    trade_max_dd = 0.0
                    
                    actions[i] = 1
                    rewards[i] = term_reward # Buy Exec = Term
                else:
                    rewards[i] = -term_reward * 0.5 # Buy Miss = Term * -0.5
            else:
                rewards[i] = 0
        else: # Holding
            hold_days += 1
            trade_max_p = max(trade_max_p, p_next)
            
            # Net Value Logic
            liquid_price = p_t * 0.995
            cur_ret = (liquid_price / (entry_price + eps)) - 1.0
            
            avg_ret = cur_ret / max(5.0, hold_days)
            dd = (trade_max_p - p_t) / (trade_max_p + eps)
            trade_max_dd = max(trade_max_dd, dd)
            
            # User Specified Non-Linear Reward Formula (Raw PnL Score)
            if cur_ret > 0:
                holding_reward = (1 + cur_ret) ** 0.5 - 1 + (hold_days**0.33) * avg_ret - trade_max_dd
            else:
                holding_reward = -(1 - cur_ret) ** 2 + 1 + (hold_days**0.33) * avg_ret - trade_max_dd
            
            should_sell = False
            
            if force_sell_cond[i]:
                should_sell = True
                actions[i] = 3
            elif sell_opp[i]:
                # Random Sell (0.2 Prob)
                if random.random() < 0.2:
                    should_sell = True
                    actions[i] = 2
                else:
                    rewards[i] = term_reward * 2.0 # Sell Miss = Term * 2.0
            
            if should_sell:
                holding = False
                # Equity Impact: Immediate 0.5% loss on sell
                current_capital *= 0.995
                equity_curve[i] = current_capital # Update current step equity (after fee)
                
                # Sell Exec = Term + Raw Holding
                rewards[i] = term_reward + holding_reward
            else:
                # Sparse Reward: Holding = 0
                rewards[i] = 0.0

    return watch_period, buy_opp, sell_opp, actions, rewards, equity_curve

def plot_symbol(symbol):
    try:
        path = list(DATA_DIR.glob(f"*{symbol}*.duckdb"))[0]
    except IndexError:
        print(f"Error: DuckDB file for symbol '{symbol}' not found.")
        return
        
    con = duckdb.connect(str(path))
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    
    # Cleaning
    df = df.dropna().reset_index(drop=True)
    if len(df) > 25: df = df.iloc[:-25]
    df['ts'] = pd.to_datetime(df['ts'])
    
    watch, buy_opp, sell_opp, actions, rewards, equity = apply_rules_and_sim(df)
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    ax1.plot(df['ts'], df['close'], color='gray', alpha=0.3, label='Close Price')
    
    for i in range(len(watch)):
        if watch[i]:
            ax1.axvspan(df['ts'][i], df['ts'][min(i+1, len(df)-1)], color='yellow', alpha=0.1)

    idx_buy_opp = np.where(buy_opp)[0]
    idx_sell_opp = np.where(sell_opp)[0]
    
    # Plot Potential Opportunities (Hollow)
    ax1.scatter(df['ts'].iloc[idx_buy_opp], df['close'].iloc[idx_buy_opp], 
                edgecolors='red', facecolors='none', marker='^', s=80, label='Potential Buy', zorder=2)
    ax1.scatter(df['ts'].iloc[idx_sell_opp], df['close'].iloc[idx_sell_opp], 
                edgecolors='green', facecolors='none', marker='v', s=80, label='Potential Sell', zorder=2)

    buys = df[actions == 1]
    sells = df[actions == 2]
    forces = df[actions == 3]
    
    ax1.scatter(buys['ts'], buys['close'], color='red', marker='^', s=100, label='Executed Buy', zorder=3)
    ax1.scatter(sells['ts'], sells['close'], color='green', marker='v', s=100, label='Executed Sell', zorder=3)
    ax1.scatter(forces['ts'], forces['close'], color='black', marker='x', s=100, label='Force Sell', zorder=3)
    
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    # Plot Reward
    ax2.plot(df['ts'], rewards.cumsum(), color='blue', linestyle='--', alpha=0.6, label='Cum Reward (Model)')
    # Plot Equity
    final_eq = equity[-1]
    ax2.plot(df['ts'], equity, color='purple', linewidth=2, label=f'Equity (Final: {final_eq:.2f})')
    
    ax2.axhline(0, color='blue', linestyle=':', alpha=0.3)
    ax2.axhline(1.0, color='purple', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Metrics')
    ax2.legend(loc='upper right')
    
    plt.title(f"Sim Check: {symbol} | Final Equity: {final_eq:.2f}")
    plt.savefig(OUTPUT_DIR / f"{symbol}_check.png")
    plt.close()
    print(f"Saved visualization for {symbol}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target = sys.argv[1]
        print(f"Generating for single symbol: {target}")
        plot_symbol(target)
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        all_files = [p.stem.split('.')[0] for p in DATA_DIR.glob("*.duckdb")]
        total = len(all_files)
        print(f"Processing ALL {total} symbols...")
        for i, s in enumerate(all_files):
            try:
                plot_symbol(s)
                if (i + 1) % 50 == 0:
                    print(f"Progress: {i+1}/{total}")
            except Exception as e:
                print(f"Error plotting {s}: {e}")
        print("Done.")
    else:
        print("Usage:")
        print("  python preprocessing/08_visualize_rules.py <symbol>   # Generate for one symbol")
        print("  python preprocessing/08_visualize_rules.py --all      # Generate for ALL symbols")