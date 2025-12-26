
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
import os

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploring.rl.env import VectorizedTradingEnvGPU
from exploring.rl.agent import DQNAgent

def get_args():
    parser = argparse.ArgumentParser(description="RL Random Search Worker")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--monitor_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "A"))
    
    # Tunable Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--collect_steps_per_block", type=int, default=20)
    
    # Fixed or Less Tuned
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=5000000) # Max steps cap
    parser.add_argument("--warm_start_steps", type=int, default=1000)
    parser.add_argument("--train_updates_per_block", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500000) # Short run for search (500k blocks approx) 
    
    return parser.parse_args()

def log_to_monitor(file_path, message):
    if not file_path: return
    try:
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        with open(file_path, "a") as f:
            f.write(f"{timestamp} {message}\n")
    except: pass

def load_and_prep_symbol(file_path):
    # (Same as original train_loop.py)
    import duckdb; import numpy as np
    import pandas as pd
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
        if "week_status" in df.columns: 
            week_ends = df[df["week_status"] == 3].index.to_numpy()
        else:
            week_ends = df.groupby(pd.to_datetime(df["ts"]).dt.to_period("W-FRI")).tail(1).index.to_numpy()
        week_end_mask = np.zeros(n, dtype=bool)
        week_end_mask[week_ends] = True
        week_end_idxs = np.where(week_end_mask)[0]
        state_indices = np.zeros((n, 20), dtype=np.int64)
        state_indices[:, -1] = np.arange(n)
        positions = np.searchsorted(week_end_idxs, np.arange(n), side='left')
        offsets = np.arange(-19, 0)
        indexer = positions[:, None] + offsets[None, :]
        indexer = np.clip(indexer, 0, len(week_end_idxs) - 1)
        if len(week_end_idxs) > 0:
            history_idxs = week_end_idxs[indexer]
            mask_future = (history_idxs >= state_indices[:, -1][:, None])
            history_idxs[mask_future] = 0
            state_indices[:, :-1] = history_idxs
        else:
            state_indices[:, :-1] = 0
        return {'name': file_path.stem, 'market': market_data, 'prices': prices, 'state_indices': state_indices, 'week_ids': week_ids, 'len': len(df)}
    except: return None

def train():
    args = get_args()
    
    # Set visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine Search Root from monitor file path (monitor file is usually in the root of search dir)
    search_root = Path(args.monitor_file).parent if args.monitor_file else Path("experiments/manual_run")
    exp_dir = search_root / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    
    # Log Start
    start_msg = f"[{args.exp_name}|GPU:{args.gpu_id}] STARTING (lr={args.lr:.1e}, gm={args.gamma}, b={args.batch_size})"
    legend = """
================================================================================
METRICS LEGEND (指标说明):
--------------------------------------------------------------------------------
[TRAIN - 训练监控]
  Rw (Avg Reward)      : 平均每步奖励 (反映即时评分)
  Pos (Position)       : 持仓占比 (Agent 处于持仓状态的比例)
  Q_a/m/M              : Q值 平均/最小/最大 (神经网络对未来的预期)
  QRR (Realism Ratio)  : Q值真实性比例 (Q_avg / Expected_Q). >1 表示过度乐观
  Grad (Grad Norm)     : 梯度范数 (监控训练稳定性)
  Act (Action Dist)    : 动作分布 (H:继续持或等, B:买入, S:卖出)
  BufS (Buffer Score)  : 经验池评分/容量
[VAL - 评估结算]
  ATR (Avg Trade Ret)  : 平均单笔收益率 (算术平均, 核心获利指标)
  MKR (Market Ret)     : 标的基准收益 (用于对比是否跑赢市场)
  AHD (Avg Hold Days)  : 平均持仓天数 (每笔单子拿多久)
  EFF (Efficiency)     : 持仓效率 (ATR / AHD, 日均赚钱能力)
  PF (Profit Factor)   : 盈亏比 (总盈利/总亏损)
  Shp (Sharpe Ratio)   : 交易级夏普率 (收益稳定性)
  Win (Win Rate)       : 真实胜率 (PnL > 0 的比例)
  Trd (Total Trades)   : 总交易笔数 (防止僵尸策略)
  Bst/Wst (Best/Worst) : 单笔最高收益 / 单笔最大亏损
================================================================================
    """
    print(start_msg + "\n" + legend)
    log_to_monitor(args.monitor_file, start_msg)
    
    # Load Data
    num_cores = psutil.cpu_count(logical=True) or 8
    all_files = sorted(list(Path(args.data_dir).glob("*.duckdb")))
    # Quick Check: Only use 50% data for faster search? No, full data for reliability.
    # But maybe limit symbols for faster loading if needed.
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        raw_results = list(executor.map(load_and_prep_symbol, all_files))
    results = [r for r in raw_results if r is not None]
    
    if not results: return
    random.shuffle(results)
    split_idx = int(len(results) * 0.8)
    train_results = results[:split_idx]
    val_results = results[split_idx:]
    
    # Helper to build bundle
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
    
    train_env = VectorizedTradingEnvGPU(train_bundle, device=device, lookback_window=args.lookback, num_envs=args.num_envs)
    val_env = VectorizedTradingEnvGPU(val_bundle, device=device, lookback_window=args.lookback, num_envs=args.num_envs)
    
    agent = DQNAgent(input_dim=train_env.state_dim, action_dim=3, device=device, buffer_size=args.buffer_size, batch_size=args.batch_size, tau=args.tau, lr=args.lr, gamma=args.gamma)
    
    # Warm Start
    print(f"Pre-filling (Random Mode)...", flush=True)
    states = train_env.reset()
    for _ in range(args.warm_start_steps):
        prev_pos = train_env.env_pos.clone()
        next_states, rewards, dones, actual_actions = train_env.step(
            torch.zeros(args.num_envs, dtype=torch.long, device=device), 
            random_mode=True
        )
        
        # Buffer Filtering: Only push Actionable Frames (4 Categories)
        sig_mask = (actual_actions != 0) | (rewards != 0)
        if sig_mask.any():
            agent.memory.push_batch(
                states[sig_mask], 
                actual_actions[sig_mask], 
                rewards[sig_mask], 
                next_states[sig_mask], 
                dones[sig_mask], 
                prev_pos.long()[sig_mask]
            )
        states = next_states
        
    epsilon = 1.0 
    eps_decay = 0.9995
    min_eps = 0.05
    
    states = train_env.reset(); total_blocks = 0
    steps_per_block = args.collect_steps_per_block * args.num_envs
    
    loss_accum, q_accum, err_accum = [torch.tensor(0.0, device=device) for _ in range(3)]
    q_min_accum, q_max_accum = torch.tensor(1e9, device=device), torch.tensor(-1e9, device=device)
    
    rew_accum = torch.tensor(0.0, device=device); rew_count = 0
    pos_accum = torch.tensor(0.0, device=device)
    act_detailed = torch.zeros(4, dtype=torch.long, device=device)
    best_val_score = -float('inf'); start_time = time.time()
    last_best_block = 0

    # Loop Limit for Search
    max_blocks = args.max_steps 
    
    while total_blocks < max_blocks:
        for _ in range(args.collect_steps_per_block):
            use_random_mode = (random.random() < epsilon)
            if not use_random_mode:
                agent_actions = agent.select_action(states, eval_mode=True)
            else:
                agent_actions = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            
            prev_pos = train_env.env_pos.clone()
            next_states, rewards, dones, actual_actions = train_env.step(agent_actions, random_mode=use_random_mode)
            
            rew_accum += rewards.sum(); rew_count += rewards.numel()
            pos_accum += (prev_pos == 1).sum()
            
            # Buffer Filtering & Logging
            sig_mask = (actual_actions != 0) | (rewards != 0)
            
            if sig_mask.any():
                # Logging Stats for this batch
                with torch.no_grad():
                    s_acts = actual_actions[sig_mask]
                    s_rews = rewards[sig_mask]
                    s_pos = prev_pos[sig_mask]
                    
                    n_bx = (s_acts == 1).sum()
                    n_sx = (s_acts == 2).sum()
                    miss_mask = (s_acts == 0)
                    n_bm = (miss_mask & (s_pos == 0)).sum()
                    n_sm = (miss_mask & (s_pos == 1)).sum()
                    
                    act_detailed[0] += n_bx
                    act_detailed[1] += n_sx
                    act_detailed[2] += n_bm
                    act_detailed[3] += n_sm

                agent.memory.push_batch(
                    states[sig_mask], 
                    actual_actions[sig_mask], 
                    rewards[sig_mask], 
                    next_states[sig_mask], 
                    dones[sig_mask], 
                    prev_pos.long()[sig_mask]
                )
            states = next_states
        
        for _ in range(args.train_updates_per_block):
            stats = agent.update()
            if stats:
                loss_accum += stats['loss']
                q_accum += stats['mean_q']
                err_accum += stats['mean_err']
                # Check if min_q/max_q are in stats (if agent updated)
                if 'min_q' in stats:
                    q_min_accum = torch.min(q_min_accum, stats['min_q'])
                    q_max_accum = torch.max(q_max_accum, stats['max_q'])
        
        if epsilon > min_eps: epsilon *= eps_decay
        total_blocks += 1
        
        # Periodic Heartbeat
        if total_blocks % 50 == 0:
            avg_rew = (rew_accum / max(1, rew_count)).item()
            cur_long_pct = (pos_accum / max(1, rew_count)).item()
            
            # Action Frequencies (Based on Significant Frames)
            total_sigs = act_detailed.sum().item()
            ap = (act_detailed.float() / max(1, total_sigs)).tolist()
            
            # Q-Realism Ratio Calculation
            q_avg = (q_accum / 500).item() # approx divisor
            expected_q = (avg_rew / (1.0 - args.gamma + 1e-8))
            qr_ratio = q_avg / (expected_q + 1e-8)
            
            buf_total, br = agent.get_buffer_info()
            q_min = q_min_accum.item() if q_min_accum < 1e8 else 0
            q_max = q_max_accum.item() if q_max_accum > -1e8 else 0
            
            log_str = (f"[{args.exp_name}|GPU:{args.gpu_id}] Blk:{total_blocks}|E:{epsilon:.2f}|Rw:{avg_rew:.3f}|"
                       f"Pos:{cur_long_pct:.0%}|Q(Av/Mn/Mx):{q_avg:.1f}/{q_min:.1f}/{q_max:.1f}|"
                       f"Sig:{ap[0]:.0%}/{ap[1]:.0%}/{ap[2]:.0%}/{ap[3]:.0%}|"
                       f"Buf:{br[0]:.0%}/{br[1]:.0%}/{br[2]:.0%}/{br[3]:.0%}|"
                       f"Size:{buf_total}")
            
            print(log_str, flush=True)
            if total_blocks % 200 == 0:
                log_to_monitor(args.monitor_file, log_str)
                
            for x in [loss_accum, q_accum, err_accum, rew_accum, pos_accum, act_detailed]: x.zero_()
            q_min_accum = torch.tensor(1e9, device=device)
            q_max_accum = torch.tensor(-1e9, device=device)
            rew_count = 0
            
        # VAL
        if total_blocks % args.eval_freq == 0:
            v_states = val_env.reset(); v_active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
            v_rew_sum = torch.tensor(0.0, device=device); v_steps = 0
            
            v_trade_rets = []
            v_hold_durations = []
            
            # MKR Fix: Capture Start Price per Env
            env_start_prices = val_env.all_prices[val_env.env_symbol_idx, val_env.env_current_step].clone()
            env_final_rets = torch.zeros(args.num_envs, device=device)
            env_has_finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            
            while v_active.any():
                v_acts = agent.select_action(v_states, eval_mode=True)
                
                # Snapshot before step for MKR calc
                curr_prices = val_env.all_prices[val_env.env_symbol_idx, val_env.env_current_step]
                prev_entry = val_env.env_entry_price.clone()
                prev_hold_days = val_env.env_hold_days.clone()
                
                v_states, v_rews, v_dones, final_acts = val_env.step(v_acts, random_mode=False)
                
                # Capture MKR on Done
                just_finished = v_dones & v_active & (~env_has_finished)
                if just_finished.any():
                    env_final_rets[just_finished] = (curr_prices[just_finished] / (env_start_prices[just_finished] + 1e-8)) - 1.0
                    env_has_finished |= just_finished
                
                if v_active.any():
                    act_rews = v_rews[v_active]
                    v_rew_sum += act_rews.sum()
                    v_steps += v_active.sum()
                    
                    # Capture Closed Trades
                    sell_mask = (final_acts == 2) & v_active
                    if sell_mask.any():
                        exit_p = curr_prices[sell_mask]
                        entry_p = prev_entry[sell_mask]
                        # Real Return (Net)
                        t_rets = (exit_p * 0.995 / (entry_p + 1e-8)) - 1.0
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
                avg_trade_ret = np.exp(np.mean(np.log(factors))) - 1.0
                
                # CRR (Cumulative Rate of Return for Agent)
                # prod(1+r) - 1. Using sum(log) for stability
                # exp(sum(log(factors))) - 1.0
                crr_val = np.exp(np.sum(np.log(factors))) - 1.0
                
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
                win_rate=0; avg_trade_ret=0; avg_hold_days=0; daily_eff=0; sharpe=0; pf=0; max_dd=0; crr_val=0.0
            
            # Construct Monitor Message (Rich Information)
            monitor_msg = (
                f"[{args.exp_name}|GPU:{args.gpu_id}] VAL @ {total_blocks} | "
                f"ATR(Geom): {avg_trade_ret:.2%} | CRR: {crr_val:.2%} | MKR(Geom): {avg_market_ret:.2%} | "
                f"Win: {win_rate:.1%} | PF: {pf:.2f} | Shp: {sharpe:.2f} | "
                f"AHD: {avg_hold_days:.1f} | EFF: {daily_eff:.3%} | Trd: {n_trades}"
            )
            print(monitor_msg, flush=True)
            log_to_monitor(args.monitor_file, monitor_msg)
            
            # Global Leaderboard (CSV)
            # Add CRR to CSV
            csv_header = "ExpName,GPU,Block,Score,AvgTradeRet,CRR,MarketRet,WinRate,PF,Sharpe,AvgHoldDays,Efficiency,Trades,Timestamp\n"
            csv_line = f"{args.exp_name},{args.gpu_id},{total_blocks},{val_score:.4f},{avg_trade_ret:.4f},{crr_val:.4f},{avg_market_ret:.4f},{win_rate:.4f},{pf:.4f},{sharpe:.4f},{avg_hold_days:.2f},{daily_eff:.6f},{n_trades},{datetime.now()}\n"
            
            global_csv = search_root / "leaderboard.csv"
            if not global_csv.exists():
                with open(global_csv, "w") as f: f.write(csv_header)
            with open(global_csv, "a") as f: f.write(csv_line)
            
            # Save Best Model based on ATR(Geom) - The Sniper Metric
            if avg_trade_ret > best_val_score:
                best_val_score = avg_trade_ret
                last_best_block = total_blocks # Update last best block
                agent.save(model_dir / "best_model.pth")
                best_msg = f"[{args.exp_name}|GPU:{args.gpu_id}] *** NEW BEST (ATR: {avg_trade_ret:.2%}) ***"
                log_to_monitor(args.monitor_file, best_msg)
            agent.save(model_dir / "latest_model.pth")
            
        # Early Stopping
        if total_blocks > 20000 and (total_blocks - last_best_block) > 10000:
            stop_msg = f"[{args.exp_name}|GPU:{args.gpu_id}] Early Stopping triggered at block {total_blocks}. No improvement since {last_best_block}."
            print(stop_msg, flush=True)
            log_to_monitor(args.monitor_file, stop_msg)
            break

    print(f"[{args.exp_name}] Finished.")
    log_to_monitor(args.monitor_file, f"[{args.exp_name}|GPU:{args.gpu_id}] FINISHED.")

if __name__ == "__main__":
    train()
