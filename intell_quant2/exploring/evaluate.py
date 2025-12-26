
import numpy as np
import pandas as pd
import torch
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploring.rl.env import TradingEnv
from exploring.rl.agent import DQNAgent

def evaluate():
    ROOT = PROJECT_ROOT
    FEATURE_DIR = ROOT / "data" / "A"
    MODEL_DIR = ROOT / "exploring" / "results" / "models"
    MODEL_PATH = MODEL_DIR / "final_model.pth"
    SYMBOLS_PATH = MODEL_DIR / "active_symbols.json"
    OUTPUT_DIR = ROOT / "exploring" / "results" / "evaluation"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = "cpu" # Eval usually on CPU is fine and simpler for batch=1
    
    if not MODEL_PATH.exists():
        print("Model not found!")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    
    env = TradingEnv(FEATURE_DIR, lookback_window=20)
    
    # Filter symbols if pruning record exists
    if SYMBOLS_PATH.exists():
        with open(SYMBOLS_PATH, "r") as f:
            active_symbols = json.load(f)
        print(f"Loaded active symbols list: {len(active_symbols)} symbols.")
        # Filter env symbols
        env.symbols = [s for s in env.symbols if s in active_symbols]
        if not env.symbols:
            print("Error: No active symbols match loaded data!")
            return
    else:
        print("No active_symbols.json found. Evaluating on ALL symbols.")
    
    agent = DQNAgent(input_dim=env.state_dim, action_dim=3, device=device)
    # Load model weights only (since we don't save optimizer in simple save)
    # Our agent.save() saved policy_net state dict directly.
    agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.policy_net.eval()
    
    trades = []
    
    # Evaluate on ALL symbols in Test Mode
    print("Starting evaluation on Test Set (Last 2 Years)...")
    
    for symbol in env.symbols:
        # Reset env specifically for this symbol in test mode
        # The env.reset() is random. We need a way to force a symbol and iterate its test period.
        # But our Env is designed for random sampling.
        # Let's iterate manually using env internals.
        
        env.current_symbol = symbol
        env.current_df = env.cache[symbol]
        split_idx = env.split_indices[symbol]
        
        # Test range: split_idx to end
        start_idx = split_idx
        end_idx = len(env.current_df) - 10
        
        if start_idx >= end_idx:
            continue
            
        # We simulate a continuous run from start_idx
        env.current_step = start_idx
        env.position = 0
        env.entry_price = 0.0
        env.max_price_during_hold = 0.0
        
        state = env._get_state()
        done = False
        
        # Track trades
        current_trade = None
        
        while env.current_step < end_idx:
            # Action
            action = agent.select_action(state, eval_mode=True)
            
            # Record Trade
            ts = env.current_df.iloc[env.current_step]["ts"]
            price = env.current_df.iloc[env.current_step]["close"]
            
            if env.position == 0 and action == 1:
                # Open Long
                current_trade = {
                    "symbol": symbol,
                    "entry_ts": ts,
                    "entry_price": price
                }
            elif env.position == 1 and action == 2:
                # Close Long
                if current_trade:
                    current_trade["exit_ts"] = ts
                    current_trade["exit_price"] = price
                    current_trade["return"] = (price / current_trade["entry_price"]) - 1.0
                    trades.append(current_trade)
                    current_trade = None
            
            next_state, reward, d, _ = env.step(action)
            state = next_state
            
            if d: # Forced close by env logic (end of data)
                if current_trade: # Force close logic in env doesn't return info, we do it here
                    current_trade["exit_ts"] = ts
                    current_trade["exit_price"] = price
                    current_trade["return"] = (price / current_trade["entry_price"]) - 1.0
                    trades.append(current_trade)
                    current_trade = None
                break
                
    # Save Trades
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.to_csv(OUTPUT_DIR / "test_trades.csv", index=False)
        print(f"Evaluation complete. {len(trades)} trades generated.")
        
        # Metrics
        win_rate = (trades_df["return"] > 0).mean()
        avg_ret = trades_df["return"].mean()
        total_ret = trades_df["return"].sum() # Simple sum of per-trade returns
        
        print("-" * 30)
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Return: {avg_ret:.2%}")
        print(f"Total Cumulative: {total_ret:.2%}")
        print("-" * 30)
        print(f"Detailed report saved to {OUTPUT_DIR}/test_trades.csv")
    else:
        print("No trades generated during evaluation.")

if __name__ == "__main__":
    evaluate()
