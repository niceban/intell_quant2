import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import random

DATA_DIR = Path("data/A")
SYMBOL = "sh601288"

def debug_symbol(symbol):
    path = list(DATA_DIR.glob(f"*{symbol}*.duckdb"))[0]
    con = duckdb.connect(str(path))
    df = con.execute("SELECT * FROM features ORDER BY ts").df()
    con.close()
    
    # 模拟规则
    df['ts'] = pd.to_datetime(df['ts'])
    # 截取最后 50 天（出现下降的那段）
    df = df.iloc[-50:].reset_index(drop=True)
    n = len(df)
    
    prices = df['close'].values
    holding = True # 强制假设持仓，复现 Holding Reward 逻辑
    entry_price = prices[0] # 假设在第0天买入
    hold_days = 0
    trade_max_p = prices[0]
    trade_max_dd = 0.0
    
    print(f"Debug {symbol} Last 50 Days (Forced Holding Simulation)")
    print(f"{'Date':<12} | {'Close':<6} | {'Days':<4} | {'CurRet':<7} | {'AvgRet':<7} | {'TimeTerm':<8} | {'MaxDD':<7} | {'REWARD':<8}")
    print("-" * 100)
    
    for i in range(n - 1):
        p_t = prices[i]
        ts = df['ts'][i].strftime('%Y-%m-%d')
        
        hold_days += 1
        trade_max_p = max(trade_max_p, p_t)
        dd = (trade_max_p - p_t) / (trade_max_p + 1e-8)
        trade_max_dd = max(trade_max_dd, dd)
        
        cur_ret = (p_t / entry_price) - 1.0
        avg_ret = cur_ret / max(5.0, hold_days)
        
        # 你的公式：0.01 阈值
        time_term = hold_days * (avg_ret - 0.01)
        reward = cur_ret + time_term - trade_max_dd * 2.0
        
        print(f"{ts:<12} | {p_t:<6.2f} | {hold_days:<4} | {cur_ret:<7.2%} | {avg_ret:<7.2%} | {time_term:<8.4f} | {trade_max_dd:<7.2%} | {reward:<8.4f}")

if __name__ == "__main__":
    debug_symbol(SYMBOL)
