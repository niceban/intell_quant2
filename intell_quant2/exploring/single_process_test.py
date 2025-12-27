import numpy as np
import random
import torch
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
from datetime import datetime
import psutil
from concurrent.futures import ProcessPoolExecutor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploring.rl.env import VectorizedTradingEnvGPU
from exploring.rl.agent import DQNAgent

def get_args():
    parser = argparse.ArgumentParser(description="RL Training Loop - v-Final Spec")
    parser.add_argument("--exp_name", type=str, default="v3_hybrid_reward_final")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "A"))
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=1000000)
    parser.add_argument("--warm_start_steps", type=int, default=100)
    parser.add_argument("--collect_steps_per_block", type=int, default=10)
    parser.add_argument("--train_updates_per_block", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=200)
    
    # Random Search Support
    parser.add_argument("--curriculum_step", type=int, default=1000, help="Blocks per 0.001 threshold increase")
    parser.add_argument("--max_threshold", type=float, default=0.01)
    
    # Deprecated but kept
    parser.add_argument("--expert_decay_episodes", type=int, default=1000)
    parser.add_argument("--min_expert_weight", type=float, default=0.05)
    return parser.parse_args()

def load_and_prep_symbol(file_path):
    import duckdb; import numpy as np
    import pandas as pd
    try:
        con = duckdb.connect(str(file_path), read_only=True)
        df = con.execute("SELECT * FROM features ORDER BY ts").df(); con.close()
        
        # 1. Data Cleaning (v-Final)
        df = df.dropna().reset_index(drop=True)
        if len(df) > 25:
            df = df.iloc[:-25] # Cut last 5 weeks
        
        if len(df) < 600: return None
        
        # 2. Extract Data
        m_cols = [c for c in df.columns if c.endswith("_state")]
        market_data = df[m_cols].values.astype(np.float32)
        prices = df["close"].values.astype(np.float32)
        n = len(df)
        
        # Calculate Week IDs for Immediate Cancel Logic
        # Format: YYYYWW (Year * 100 + Week)
        # Using isocalendar() to be precise
        iso_cal = df['ts'].dt.isocalendar()
        week_ids = (iso_cal.year * 100 + iso_cal.week).values.astype(np.int64)

        # 3. PRE-COMPUTE LOOKBACK INDICES (1 Daily + 19 Weekly)
        if "week_status" in df.columns: 
            week_ends = df[df["week_status"] == 3].index.to_numpy()
        else: 
            week_ends = df.groupby(pd.to_datetime(df["ts"]).dt.to_period("W-FRI")).tail(1).index.to_numpy()
            
        week_end_mask = np.zeros(n, dtype=bool)
        week_end_mask[week_ends] = True
        week_end_idxs = np.where(week_end_mask)[0]
        
        state_indices = np.zeros((n, 20), dtype=np.int64)
        state_indices[:, -1] = np.arange(n) # Last one is current daily
        
        positions = np.searchsorted(week_end_idxs, np.arange(n), side='left')
        offsets = np.arange(-19, 0)
        indexer = positions[:, None] + offsets[None, :]
        indexer = np.clip(indexer, 0, len(week_end_idxs) - 1)
        
        if len(week_end_idxs) > 0:
            history_idxs = week_end_idxs[indexer]
            # Avoid future leak: history idx must be < current idx
            mask_future = (history_idxs >= state_indices[:, -1][:, None])
            history_idxs[mask_future] = 0
            state_indices[:, :-1] = history_idxs
        else:
            state_indices[:, :-1] = 0
            
        return {
            'name': file_path.stem, 
            'market': market_data, 
            'prices': prices, 
            'state_indices': state_indices,
            'week_ids': week_ids,
            'len': len(df)
        }
    except: return None

def train():
    args = get_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.exp_dir) if args.exp_dir else PROJECT_ROOT / "experiments" / f"{args.exp_name}_{timestamp}"
    model_dir = exp_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    num_cores = psutil.cpu_count(logical=True) or 8
    all_files = sorted(list(Path(args.data_dir).glob("*.duckdb")))
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        raw_results = list(executor.map(load_and_prep_symbol, all_files))
    results = [r for r in raw_results if r is not None]
    
    if not results: print("No data!"); return
    
    # Symbol-level Train/Val Split (8:2)
    random.shuffle(results)
    split_idx = int(len(results) * 0.8)
    train_results = results[:split_idx]
    val_results = results[split_idx:]
    
    def build_bundle(subset_results):
        if not subset_results: return None
        m_len = max(r['len'] for r in subset_results)
        n_res = len(subset_results)
        b = {
            'names': [r['name'] for r in subset_results],
            'market_tensors': torch.zeros((n_res, m_len, 20), dtype=torch.float32),
            'price_tensors': torch.zeros((n_res, m_len), dtype=torch.float32),
            'state_indices_tensors': torch.zeros((n_res, m_len, 20), dtype=torch.long),
            'week_id_tensors': torch.zeros((n_res, m_len), dtype=torch.long),
            'lengths': torch.tensor([r['len'] for r in subset_results], dtype=torch.long)
        }
        for i, r in enumerate(subset_results):
            l = r['len']
            b['market_tensors'][i, :l] = torch.from_numpy(r['market'])
            b['price_tensors'][i, :l] = torch.from_numpy(r['prices'])
            b['state_indices_tensors'][i, :l] = torch.from_numpy(r['state_indices'])
            b['week_id_tensors'][i, :l] = torch.from_numpy(r['week_ids'])
        return b

    train_bundle = build_bundle(train_results)
    val_bundle = build_bundle(val_results)
    del results, raw_results, train_results, val_results
    
    # Instantiate Envs
    train_env = VectorizedTradingEnvGPU(train_bundle, device=device, lookback_window=args.lookback, num_envs=args.num_envs)
    val_env = VectorizedTradingEnvGPU(val_bundle, device=device, lookback_window=args.lookback, num_envs=args.num_envs)
    
    agent = DQNAgent(input_dim=train_env.state_dim, action_dim=3, device=device, buffer_size=args.buffer_size, batch_size=args.batch_size, tau=args.tau, lr=args.lr, gamma=args.gamma)
    
    legend = """
================================================================================
METRICS LEGEND (指标说明):
--------------------------------------------------------------------------------
[TRAIN - 训练监控]
  Rw (Avg Reward)      : 平均每步奖励
  Pos (Position)       : 持仓占比
  Q_a/m/M              : Q值 平均/最小/最大
  Act (Action Dist)    : 动作分布 (H:持/等, B:买, S:卖)
[VAL - 评估结算]
  ATR (Avg Trade Ret)  : 平均单笔收益率 (算术平均, 核心获利指标)
  MKR (Market Ret)     : 标的基准收益 (用于对比是否跑赢市场)
  AHD (Avg Hold Days)  : 平均持仓天数
  EFF (Efficiency)     : 持仓效率 (ATR / AHD)
  PF (Profit Factor)   : 盈亏比
  Shp (Sharpe Ratio)   : 交易级夏普率
  Win (Win Rate)       : 真实胜率
  Trd (Total Trades)   : 总交易笔数
================================================================================
    """
    print(legend)
    
    print(f"Pre-filling (Random Mode)...", flush=True)
    states = train_env.reset()
    for _ in range(args.warm_start_steps):
        prev_pos = train_env.env_pos.clone()
        next_states, rewards, dones, final_actions = train_env.step(
            torch.zeros(args.num_envs, dtype=torch.long, device=device), 
            random_mode=True,
            avg_ret_threshold=0.0 # Warm start with 0 threshold
        )
        agent.memory.push_batch(states, final_actions, rewards, next_states, dones, prev_pos.long())
        states = next_states
    
    epsilon = 1.0 
    eps_decay = 0.9995
    min_eps = 0.05
    
    states = train_env.reset(); total_blocks = 0
    steps_per_block = args.collect_steps_per_block * args.num_envs
    
    loss_accum, q_accum, err_accum = [torch.tensor(0.0, device=device) for _ in range(3)]
    q_min_accum = torch.tensor(1e9, device=device)
    q_max_accum = torch.tensor(-1e9, device=device)
    
    rew_accum = torch.tensor(0.0, device=device); rew_count = 0
    pos_accum = torch.tensor(0.0, device=device)
    act_detailed = torch.zeros(4, dtype=torch.long, device=device)
    best_val_score = -float('inf'); start_time = time.time()
    last_best_block = 0

    while total_blocks * steps_per_block < args.num_episodes * 200:
        
        for _ in range(args.collect_steps_per_block):
            use_random_mode = (random.random() < epsilon)
            
            if not use_random_mode:
                agent_actions = agent.select_action(states, eval_mode=True)
            else:
                agent_actions = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            
            prev_pos = train_env.env_pos.clone()
            
            next_states, rewards, dones, actual_actions = train_env.step(
                agent_actions, 
                random_mode=use_random_mode
            )
            
            rew_accum += rewards.sum(); rew_count += rewards.numel()
            pos_accum += prev_pos.sum()
            is_hold = (actual_actions == 0)
            act_detailed[0] += (is_hold & (prev_pos == 0)).sum()
            act_detailed[1] += (is_hold & (prev_pos == 1)).sum()
            act_detailed[2] += (actual_actions == 1).sum()
            act_detailed[3] += (actual_actions == 2).sum()

            agent.memory.push_batch(states, actual_actions, rewards, next_states, dones, prev_pos.long())
            states = next_states
        
        for _ in range(args.train_updates_per_block):
            stats = agent.update()
            if stats:
                loss_accum += stats['loss']; q_accum += stats['mean_q']; err_accum += stats['mean_err']
                q_min_accum = torch.min(q_min_accum, stats['min_q'])
                q_max_accum = torch.max(q_max_accum, stats['max_q'])
        
        if epsilon > min_eps: epsilon *= eps_decay
        total_blocks += 1
        
        if total_blocks % 20 == 0:
            avg_loss, avg_q, avg_err = [x.item() / 200 for x in [loss_accum, q_accum, err_accum]]
            avg_rew = (rew_accum / max(1, rew_count)).item()
            cur_long_pct = (pos_accum / max(1, rew_count)).item()
            
            # act_detailed now stores: [BuyExec, SellExec, BuyMiss, SellMiss]
            total_sigs = act_detailed.sum().item()
            ap = (act_detailed.float() / total_sigs).tolist()
            buf_total, br = agent.get_buffer_info()
            
            # Q Stats
            q_min = q_min_accum.item() if q_min_accum < 1e8 else 0
            q_max = q_max_accum.item() if q_max_accum > -1e8 else 0
            
            # Log threshold
            # BX:BuyExec, SX:SellExec, BM:BuyMiss, SM:SellMiss
            log_str = (f"Blk:{total_blocks}|E:{epsilon:.2f}|Rw:{avg_rew:.4f}|"
                       f"Pos:{cur_long_pct:.0%}|"
                       f"Buf(A/L/F):{br[0]:.0%}/{br[1]:.0%}/{br[2]:.0%}/{br[3]:.0%}|"
                       f"Sig:{ap[0]:.0%}/{ap[1]:.0%}/{ap[2]:.0%}/{ap[3]:.0%}|"
                       f"Q(Av/Mn/Mx):{avg_q:.1f}/{q_min:.1f}/{q_max:.1f}|L:{avg_loss:.1f}|T:{time.time()-start_time:.1f}s")
            print(log_str, flush=True)
            for x in [loss_accum, q_accum, err_accum, rew_accum, pos_accum, act_detailed]: x.zero_()
            q_min_accum = torch.tensor(1e9, device=device)
            q_max_accum = torch.tensor(-1e9, device=device)
            rew_count = 0; start_time = time.time()
            
        if total_blocks % args.eval_freq == 0:
            v_states = val_env.reset(); v_active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
            v_rew_sum = torch.tensor(0.0, device=device); v_steps = 0
            v_trade_rets = []
            v_hold_durations = []
            
            start_prices = val_env.all_prices[val_env.env_symbol_idx, val_env.env_current_step].clone()
            
            while v_active.any():
                v_acts = agent.select_action(v_states, eval_mode=True)
                curr_prices = val_env.all_prices[val_env.env_symbol_idx, val_env.env_current_step]
                prev_entry = val_env.env_entry_price.clone()
                prev_hold_days = val_env.env_hold_days.clone()
                
                v_states, v_rews, v_dones, final_acts = val_env.step(v_acts, random_mode=False)
                
                if v_active.any():
                    v_rew_sum += v_rews[v_active].sum()
                    v_steps += v_active.sum()
                    sell_mask = (final_acts == 2) & v_active
                    if sell_mask.any():
                        t_rets = (curr_prices[sell_mask] * 0.995 / (prev_entry[sell_mask] + 1e-8)) - 1.0
                        v_trade_rets.extend(t_rets.cpu().numpy().tolist())
                        v_hold_durations.extend(prev_hold_days[sell_mask].cpu().numpy().tolist())
                v_active &= ~v_dones
            
            
            # Geometric Mean for Market Return (Log-Sum-Exp Stability Fix)
            if env_has_finished.any():
                mkr_vals = 1.0 + env_final_rets[env_has_finished]
                # Filter valid and clamp for log safety (min -99.99% return)
                mkr_vals = torch.clamp(mkr_vals, min=1e-6)
                
                if mkr_vals.numel() > 0:
                    # exp(mean(log(x))) - 1
                    log_vals = torch.log(mkr_vals)
                    avg_market_ret = (torch.exp(torch.mean(log_vals)) - 1.0).item()
                else:
                    avg_market_ret = -1.0
            else:
                avg_market_ret = 0.0

            # Compute Final Metrics
            val_score = (v_rew_sum / max(1, v_steps)).item()
            n_trades = len(v_trade_rets)
            
            if n_trades > 0:
                tr_arr = np.array(v_trade_rets)
                hd_arr = np.array(v_hold_durations)
                
                # Geometric Mean for Trade Returns (Log-Sum-Exp Stability Fix)
                factors = 1.0 + tr_arr
                # Clamp for safety
                factors = np.maximum(factors, 1e-6)
                
                # ATR (Geometric Mean per trade)
                sum_log_returns = np.sum(np.log(factors))
                avg_trade_ret = np.exp(sum_log_returns / len(factors)) - 1.0
                
                # SymCRR (Average Cumulative Return per Symbol/Env Slot)
                sym_crr = np.exp(sum_log_returns / args.num_envs) - 1.0
                
                # Total Return (Sum of simple returns)
                total_ret = tr_arr.sum()
                
                win_rate = (tr_arr > 0).mean()
                avg_hold_days = hd_arr.mean()
                daily_eff = avg_trade_ret / (avg_hold_days + 1e-8)
                
                # Trade-level Sharpe
                sharpe = avg_trade_ret / (tr_arr.std() + 1e-9)
                
                # Profit Factor
                gross_win = tr_arr[tr_arr > 0].sum()
                gross_loss = abs(tr_arr[tr_arr <= 0].sum())
                pf = gross_win / (gross_loss + 1e-9)
            else:
                win_rate=0; avg_trade_ret=0; avg_hold_days=0; daily_eff=0; sharpe=0; pf=0; max_dd=0; total_ret=0.0; sym_crr=0.0
            
            print(f"--- VAL @ {total_blocks} | ATR: {avg_trade_ret:.2%} | SymCRR: {sym_crr:.2%} | TotRet: {total_ret:.1f} | MKR: {avg_market_ret:.2%} | Win: {win_rate:.1%} | PF: {pf:.2f} | Shp: {sharpe:.2f} | AHD: {avg_hold_days:.1f} | EFF: {daily_eff:.3%} | Trd: {n_trades} ---", flush=True)
            
            if avg_trade_ret > best_val_score:
                best_val_score = avg_trade_ret
                last_best_block = total_blocks
                agent.save(model_dir / "best_model.pth")
                print(f">>> BEST SAVED (ATR: {avg_trade_ret:.2%}) <<<", flush=True)
            agent.save(model_dir / "latest_model.pth")
            
        # Early Stopping
        if total_blocks > 20000 and (total_blocks - last_best_block) > 10000:
            print(f"Early Stopping triggered at block {total_blocks}. No improvement since {last_best_block}.", flush=True)

    agent.save(exp_dir / "models" / "final_model.pth"); print("Training finished.")

if __name__ == "__main__":
    train()