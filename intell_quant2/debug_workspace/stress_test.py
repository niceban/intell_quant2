import sys
from pathlib import Path
import torch
import random
import numpy as np
import duckdb

# Setup Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_debug import VectorizedTradingEnvGPU
from agent_debug import DQNAgent

def get_bundle(file_paths):
    results = []
    for fp in file_paths:
        con = duckdb.connect(str(fp), read_only=True)
        df = con.execute("SELECT * FROM features ORDER BY ts").df(); con.close()
        df = df.dropna().reset_index(drop=True)
        if len(df) < 600: continue
        m_cols = [c for c in df.columns if c.endswith("_state")]
        market = df[m_cols].values.astype(np.float32)
        prices = df["close"].values.astype(np.float32)
        iso_cal = df['ts'].dt.isocalendar()
        week_ids = (iso_cal.year * 100 + iso_cal.week).values.astype(np.int64)
        results.append({'market': market, 'prices': prices, 'week_ids': week_ids, 'len': len(df)})
    
    if not results: return None
    max_l = max(r['len'] for r in results)
    n = len(results)
    b = {
        'names': [f"s_{i}" for i in range(n)],
        'market_tensors': torch.zeros((n, max_l, 20)),
        'price_tensors': torch.zeros((n, max_l)),
        'state_indices_tensors': torch.zeros((n, max_l, 20), dtype=torch.long),
        'week_id_tensors': torch.zeros((n, max_l), dtype=torch.long),
        'lengths': torch.tensor([r['len'] for r in results])
    }
    for i, r in enumerate(results):
        l = r['len']
        b['market_tensors'][i, :l] = torch.from_numpy(r['market'])
        b['price_tensors'][i, :l] = torch.from_numpy(r['prices'])
        b['week_id_tensors'][i, :l] = torch.from_numpy(r['week_ids'])
        # Simplified state indices for debug
        idx = np.arange(l)
        b['state_indices_tensors'][i, :l, -1] = torch.from_numpy(idx)
    return b

def run_test(gamma_value, name):
    print(f"\n\n{'='*20} TESTING: {name} (Gamma={gamma_value}) {'='*20}")
    data_files = sorted(list(Path("data/A").glob("*.duckdb")))[:10]
    bundle = get_bundle(data_files)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env = VectorizedTradingEnvGPU(bundle, device=device, num_envs=512)
    # Using 128 head_hidden_dim like the log that performed better
    agent = DQNAgent(input_dim=env.state_dim, action_dim=3, gamma=gamma_value, device=device, head_hidden_dim=128)
    
    states = env.reset()
    for i in range(2000): # Run for 2000 blocks
        # Env step
        actions = agent.select_action(states)
        prev_pos = env.env_pos.clone()
        next_states, rewards, dones, final_acts = env.step(actions)
        
        # Only push interesting samples to speed up
        mask = (final_acts != 0) | (rewards != 0)
        if mask.any():
            agent.memory.push_batch(states[mask], final_acts[mask], rewards[mask], next_states[mask], dones[mask], prev_pos.long()[mask])
        
        states = next_states
        
        # Train
        if len(agent.memory) > 1024:
            for _ in range(5): # 5 updates per step
                agent.update()
        
        if i % 200 == 0:
            print(f"Block {i} completed.")

if __name__ == "__main__":
    # Test 1: Your design (Gamma=0)
    run_test(0.0, "GAMMA_ZERO")
    # Test 2: Potential Leak (Gamma=0.99)
    run_test(0.99, "GAMMA_HIGH")
