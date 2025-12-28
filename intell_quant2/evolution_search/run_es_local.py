import sys
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time

# Setup Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from es_engine import BayesianES
from env_es import VectorizedTradingEnvGPU
from model_es import LSTM_DDDQN

def load_and_prep_symbol(file_path):
    import duckdb
    try:
        con = duckdb.connect(str(file_path), read_only=True)
        df = con.execute("SELECT * FROM features ORDER BY ts").df(); con.close()
        df = df.dropna().reset_index(drop=True)
        if len(df) > 25: df = df.iloc[:-25]
        if len(df) < 600: return None
        m_cols = [c for c in df.columns if c.endswith("_state")]
        market_data = df[m_cols].values.astype(np.float32)
        prices = df["close"].values.astype(np.float32)
        n = len(df)
        iso_cal = df['ts'].dt.isocalendar()
        week_ids = (iso_cal.year * 100 + iso_cal.week).values.astype(np.int64)
        week_end_idxs = df[df["week_status"] == 3].index.to_numpy() if "week_status" in df.columns else df.groupby(pd.to_datetime(df["ts"]).dt.to_period("W-FRI")).tail(1).index.to_numpy()
        state_indices = np.zeros((n, 20), dtype=np.int64)
        state_indices[:, -1] = np.arange(n)
        positions = np.searchsorted(week_end_idxs, np.arange(n), side='left')
        indexer = np.clip(positions[:, None] + np.arange(-19, 0)[None, :], 0, len(week_end_idxs) - 1)
        if len(week_end_idxs) > 0:
            hist = week_end_idxs[indexer]
            hist[hist >= state_indices[:, -1][:, None]] = 0
            state_indices[:, :-1] = hist
        return {'name': file_path.stem, 'market': market_data, 'prices': prices, 'state_indices': state_indices, 'week_ids': week_ids, 'len': len(df)}
    except: return None

def load_data(limit=200):
    data_dir = Path("data/A")
    all_files = sorted(list(data_dir.glob("*.duckdb")))
    print(f"[DATA] Loading {limit} unique symbols...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_and_prep_symbol, all_files[:limit]))
    results = [r for r in results if r is not None]
    m_len = max(r['len'] for r in results)
    n = len(results)
    b = {
        'market_tensors': torch.zeros((n, m_len, 20)),
        'price_tensors': torch.zeros((n, m_len)),
        'state_indices_tensors': torch.zeros((n, m_len, 20), dtype=torch.long),
        'week_id_tensors': torch.zeros((n, m_len), dtype=torch.long),
        'lengths': torch.zeros(n, dtype=torch.long),
        'names': [r['name'] for r in results]
    }
    for i, r in enumerate(results):
        l = r['len']
        b['market_tensors'][i, :l] = torch.from_numpy(r['market'])
        b['price_tensors'][i, :l] = torch.from_numpy(r['prices'])
        b['state_indices_tensors'][i, :l] = torch.from_numpy(r['state_indices'])
        b['week_id_tensors'][i, :l] = torch.from_numpy(r['week_ids'])
        b['lengths'][i] = l
    return b

def run_es():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 90%+ UTILIZATION CONFIG: 32 agents * 1024 envs = 32,768 Parallel Environments
    # Note: Using Shared Memory mode in env_es.py
    pop_size = 32
    envs_per_agent = 1024
    num_envs = pop_size * envs_per_agent 
    
    bundle = load_data(limit=300) # Load 300 symbols once
    env = VectorizedTradingEnvGPU(bundle, device=device, num_envs=num_envs)
    es = BayesianES(34, 3, pop_size=pop_size, sigma=0.05, device=device)
    model = LSTM_DDDQN(34, 3, head_hidden_dim=128).to(device)
    
    print(f"\n[EXEC] Config: Pop={pop_size}, Envs/Agent={envs_per_agent}, Total={num_envs}")
    print(f"[EXEC] Mode: NeuroEvolution | Shared Memory | Non-Linear Quantized Reward\n")

    for gen in range(1000):
        gen_start = time.time()
        agents = es.ask()
        obs = env.reset()
        total_rewards = torch.zeros(num_envs, device=device)
        
        sim_steps = 300
        for s in range(sim_steps):
            full_actions = torch.zeros(num_envs, dtype=torch.long, device=device)
            full_rule_mask = torch.zeros((num_envs, 9), dtype=torch.bool, device=device)
            
            with torch.no_grad():
                for i in range(pop_size):
                    start, end = i * envs_per_agent, (i + 1) * envs_per_agent
                    model.load_state_dict(agents[i].weights_dict, strict=False)
                    q_entry, q_exit = model(obs[start:end])
                    pos = obs[start:end, -1, 20]
                    sub_acts = torch.zeros(envs_per_agent, dtype=torch.long, device=device)
                    sub_acts[pos < 0.5] = q_entry[pos < 0.5].argmax(dim=1)
                    sub_acts[pos >= 0.5] = torch.where(q_exit[pos >= 0.5].argmax(dim=1) == 1, torch.tensor(2, device=device), torch.tensor(0, device=device))
                    full_actions[start:end] = sub_acts
                    full_rule_mask[start:end] = agents[i].rule_mask
            
            obs, rewards, _, _ = env.step(full_actions, rule_mask=full_rule_mask)
            total_rewards += rewards

        for i in range(pop_size):
            agents[i].fitness = total_rewards[i*envs_per_agent:(i+1)*envs_per_agent].mean().item()
            
        best_f, mean_f = es.tell(agents)
        dur = time.time() - gen_start
        trd = env.env_trade_count.float().mean().item()
        print(f"Gen {gen+1:3d} | FitBest: {best_f:6.2f} | FitMean: {mean_f:6.2f} | Trd: {trd:4.1f} | Speed: {num_envs*sim_steps/dur:,.0f} s/s | {dur:4.1f}s")
        
        if (gen+1) % 20 == 0:
            print("\n" + es.distill_rules() + "\n")
            torch.save({'weights': es.get_master_state_dict(), 'gen': gen+1}, f"evolution_search/best_checkpoint.pth")

if __name__ == "__main__":
    run_es()
