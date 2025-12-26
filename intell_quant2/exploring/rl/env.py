import numpy as np
import random
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class VectorizedTradingEnvGPU:
    """
    v-Final Spec:
    - Mixed Window State (19 Weekly + 1 Daily)
    - Rule-based Action Masking (Watch Period, Opp Points)
    - Hybrid Reward (Term for Action, Formula for Holding)
    - Auto-Reset
    - Precise Week ID Cancellation
    - Implicit Transaction Costs (0.5% Entry/Exit)
    - Non-linear Reward Formula
    """
    def __init__(self, data_bundle: Dict, device: str, lookback_window: int = 20, num_envs: int = 512):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.lookback_window = lookback_window
        self.state_dim = 34
        
        self.symbol_names = data_bundle['names']
        self.num_symbols = len(self.symbol_names)
        self.all_market_data = data_bundle['market_tensors'].to(self.device)
        self.all_prices = data_bundle['price_tensors'].to(self.device)
        self.all_state_indices = data_bundle['state_indices_tensors'].to(self.device)
        self.all_week_ids = data_bundle['week_id_tensors'].to(self.device) # New: Week IDs
        self.data_lengths = data_bundle['lengths'].to(self.device)
        
        self._precompute_terms()
        
        # State Tensors
        self.env_symbol_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_current_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_pos = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_entry_price = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_hold_days = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_trade_max_p = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_trade_max_dd = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_last_reward = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_trade_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_trade_start_week_id = torch.zeros(num_envs, dtype=torch.long, device=self.device) # Precise Week ID
        
        # Track if current trade was initiated in Random Mode
        self.env_is_random_trade = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        max_len = self.all_prices.size(1)
        self.env_acc_history = torch.zeros((num_envs, max_len + 10, 14), dtype=torch.float32, device=self.device)
        self.env_range = torch.arange(self.num_envs, device=self.device)
        print(f"Auto-Reset Env Ready: {num_envs} slots.")

    def _precompute_terms(self):
        padded = torch.nn.functional.pad(self.all_prices, (0, 25), mode='constant', value=0)
        p_future = padded[:, 1:] 
        windows = p_future.unfold(1, 25, 1) 
        
        w_max = windows.max(dim=2).values
        w_min = windows.min(dim=2).values
        mask_invalid = (w_min <= 1e-8)
        
        p = self.all_prices
        w_max[mask_invalid] = p[mask_invalid]
        w_min[mask_invalid] = p[mask_invalid]
        
        self.all_terms = ((w_max - p) - (p - w_min)) / (p + 1e-8)
        self.all_terms = torch.clamp(self.all_terms, -10.0, 10.0)

    def _reset_indices(self, env_indices: torch.Tensor):
        n = env_indices.numel()
        if n == 0: return
        
        new_sym_idxs = torch.randint(0, self.num_symbols, (n,), device=self.device)
        self.env_symbol_idx[env_indices] = new_sym_idxs
        
        for i, e_idx in enumerate(env_indices):
            s_idx = new_sym_idxs[i]
            max_s = int(self.data_lengths[s_idx].item()) - 30 
            start = 25
            if max_s <= start: max_s = start + 1
            self.env_current_step[e_idx] = random.randint(start, max_s)
            
        self.env_pos[env_indices] = 0.0
        self.env_entry_price[env_indices] = 0.0
        self.env_hold_days[env_indices] = 0.0
        self.env_trade_max_p[env_indices] = 0.0
        self.env_trade_max_dd[env_indices] = 0.0
        self.env_last_reward[env_indices] = 0.0
        self.env_trade_count[env_indices] = 0
        self.env_trade_start_week_id[env_indices] = -1
        self.env_is_random_trade[env_indices] = False
        self.env_acc_history[env_indices] = 0.0
        
        self._update_account_history(env_indices)

    def _update_account_history(self, env_indices: torch.Tensor):
        curr_t = self.env_current_step[env_indices]
        s_idx = self.env_symbol_idx[env_indices]
        prices = self.all_prices[s_idx, curr_t]
        
        eps = 1e-8
        is_long = (self.env_pos[env_indices] == 1)
        
        cur_ret = torch.zeros_like(prices)
        avg_ret = torch.zeros_like(prices)
        
        entry_prices = self.env_entry_price[env_indices]
        hold_days = self.env_hold_days[env_indices]
        max_dd = self.env_trade_max_dd[env_indices]
        last_rew = self.env_last_reward[env_indices]
        
        if is_long.any():
            # Standard return calc for state (can stick to raw or net, let's use net for consistency)
            # Net Value: (Price * 0.995 / Entry) - 1.0
            cur_ret[is_long] = (prices[is_long] * 0.995 / (entry_prices[is_long] + eps)) - 1.0
            avg_ret[is_long] = cur_ret[is_long] / torch.clamp(hold_days[is_long], min=5.0)
            
        m0 = torch.stack([self.env_pos[env_indices], hold_days, cur_ret, avg_ret, max_dd, last_rew], dim=1)
        
        m_1 = self.env_acc_history[env_indices, curr_t - 1, 2:6]
        m_2 = self.env_acc_history[env_indices, curr_t - 2, 2:6]
        
        full_vec = torch.cat([m0, m0[:, 2:6] - m_1, m_1 - m_2], dim=1)
        
        # Scaling Diffs: d1_reward and d2_reward should be scaled by 25.0
        # Since m0[5] (last_rew) is already * 5.0 from step(), 
        # d1 (index 9) and d2 (index 13) are raw_diff * 5.0.
        # To make them raw_diff * 25.0, we multiply by another 5.0.
        full_vec[:, 9] *= 5.0 # d1_last_rew
        full_vec[:, 13] *= 5.0 # d2_last_rew
        
        self.env_acc_history[env_indices, curr_t] = full_vec

    def reset(self, env_indices: torch.Tensor = None):
        if env_indices is None: env_indices = self.env_range
        self._reset_indices(env_indices)
        return self._get_states(env_indices)

    def _get_states(self, env_indices: torch.Tensor) -> torch.Tensor:
        curr_t = self.env_current_step[env_indices]
        s_idx = self.env_symbol_idx[env_indices]
        
        target_indices = self.all_state_indices[s_idx, curr_t] 
        s_idx_exp = s_idx.unsqueeze(1).expand(-1, 20)
        market_obs = self.all_market_data[s_idx_exp, target_indices]
        
        e_idx_exp = env_indices.unsqueeze(1).expand(-1, 20)
        acc_obs = self.env_acc_history[e_idx_exp, target_indices]
        
        return torch.cat([market_obs, acc_obs], dim=-1)

    def step(self, agent_actions: torch.Tensor, random_mode: bool = False, avg_ret_threshold: float = 0.001, term_scale: float = 3.0, inaction_penalty: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-8
        s_idx = self.env_symbol_idx
        curr_t = self.env_current_step
        
        m_data = self.all_market_data[s_idx, curr_t] 
        
        # Week Features
        w_dif = m_data[:, 0]; w_dea = m_data[:, 1]; w_bar = m_data[:, 2]
        w_k   = m_data[:, 3]; w_d   = m_data[:, 4]; w_dma = m_data[:, 5]; w_ama = m_data[:, 6]
        
        # Month Features (Offset +11)
        m_dif = m_data[:, 11]; m_dea = m_data[:, 12]; m_bar = m_data[:, 13]
        m_k   = m_data[:, 14]; m_d   = m_data[:, 15]; m_dma = m_data[:, 16]; m_ama = m_data[:, 17]
        
        m_strict = (m_k>0) & (m_d>0) & (m_bar>0) & (m_dif>0) & (m_dea>0) & (m_dma>0)
        m_loose = (m_dea>0) & (m_ama>0)
        month_good = m_strict | m_loose
        
        watch_period = month_good
        buy_opp = watch_period & (w_d == 2)
        sell_opp = (w_d == -2)
        
        # v-Final Spec: 6线全负 (Relaxed Force Sell condition)
        # Added skdj_k < 0 and macd_bar < 0
        force_sell_cond = (w_d < 0) & (w_k < 0) & \
                          (w_dif < 0) & (w_dea < 0) & (w_bar < 0) & \
                          (w_ama < 0)
        
        # Expanded Sell Opportunities (DEA Dead Cross or All-Negative Water)
        # Note: These are OPPORTUNITIES, Agent can choose (or forced by Miss penalty).
        dea_dead = (w_dea == -2)
        water_dead = (w_dea < 0) & (w_dif < 0) & (w_bar < 0)
        sell_opp = sell_opp | dea_dead | water_dead

        # Immediate Cancellation Condition (Precise Week ID)
        curr_week_ids = self.all_week_ids[s_idx, curr_t]
        is_same_week = (curr_week_ids == self.env_trade_start_week_id)
        immediate_cancel = (self.env_pos == 1) & is_same_week & (~buy_opp)

        # Decision Logic (Unified Random Mode)
        final_actions = torch.zeros_like(agent_actions)
        
        # 1. Determine Buy Actions (Pos=0)
        # Only check random_mode if we are currently empty
        buy_candidates = (self.env_pos == 0) & buy_opp
        if random_mode:
            # In random mode (warm start/epsilon), we buy with 0.8 prob
            rand_buys = (torch.rand(self.num_envs, device=self.device) < 0.8)
            do_buy = buy_candidates & rand_buys
            final_actions[do_buy] = 1
            # Mark these trades as Random Trades
            self.env_is_random_trade[do_buy] = True
        else:
            # Agent decides
            do_buy = buy_candidates & (agent_actions == 1)
            final_actions[do_buy] = 1
            # Mark these as Agent Trades (False)
            self.env_is_random_trade[do_buy] = False

        # 2. Determine Sell Actions (Pos=1)
        sell_candidates = (self.env_pos == 1) & sell_opp
        
        # Split into Random Trades and Agent Trades
        is_rnd = self.env_is_random_trade
        
        # A. Random Trades: Force 0.2 Prob on Sell Opp
        if sell_candidates.any():
            rnd_sell_mask = sell_candidates & is_rnd
            if rnd_sell_mask.any():
                # 0.2 Prob to sell
                do_rnd_sell = (torch.rand(self.num_envs, device=self.device) < 0.2)
                final_actions[rnd_sell_mask & do_rnd_sell] = 2
                
            # B. Agent Trades: Listen to Agent
            agt_sell_mask = sell_candidates & (~is_rnd)
            if agt_sell_mask.any():
                do_agt_sell = (agent_actions == 2)
                final_actions[agt_sell_mask & do_agt_sell] = 2

        # Decision Correction & Must-Sell Logic
        final_actions[(self.env_pos == 1) & (final_actions == 1)] = 0
        final_actions[(self.env_pos == 0) & (final_actions == 2)] = 0
        
        # Action Masking (Redundant but safe)
        final_actions[(~buy_opp) & (final_actions == 1)] = 0
        final_actions[(~sell_opp) & (final_actions == 2)] = 0
        
        # Calculate raw holding reward for must-sell check
        prices = self.all_prices[s_idx, curr_t]
        
        # Net Value Logic: (Price * 0.995 / Entry) - 1.0
        # Entry Price already includes Buy Cost (Price * 1.005)
        liquid_price = prices * 0.995
        cur_ret_tmp = (liquid_price / (self.env_entry_price + eps)) - 1.0
        
        h_days_tmp = self.env_hold_days + 1.0
        avg_ret_tmp = cur_ret_tmp / torch.clamp(h_days_tmp, min=5.0)
        t_max_tmp = torch.max(self.env_trade_max_p, prices)
        dd_tmp = (t_max_tmp - prices) / (t_max_tmp + eps)
        t_dd_tmp = torch.max(self.env_trade_max_dd, dd_tmp)

        # User Non-Linear Formula (Raw) - Calculated for ALL holding steps for State Update
        raw_holding_rew = torch.zeros(self.num_envs, device=self.device)
        pos_mask = (cur_ret_tmp > 0)
        neg_mask = (cur_ret_tmp <= 0)
        
        raw_holding_rew[pos_mask] = (1.0 + cur_ret_tmp[pos_mask])**0.5 - 1.0 + \
                                    (h_days_tmp[pos_mask]**0.33) * avg_ret_tmp[pos_mask] - t_dd_tmp[pos_mask]
        raw_holding_rew[neg_mask] = -(1.0 - cur_ret_tmp[neg_mask])**2 + 1.0 + \
                                    (h_days_tmp[neg_mask]**0.33) * avg_ret_tmp[neg_mask] - t_dd_tmp[neg_mask]

        # Force Sell Condition (4-line negative)
        must_sell = (self.env_pos == 1) & (force_sell_cond | immediate_cancel)
        final_actions[must_sell] = 2
        
        # Final Reward Calculation
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Action Reward (Term)
        terms = self.all_terms[s_idx, curr_t]
        term_rew = terms # Raw Term Value
        
        missed_buy = buy_opp & (self.env_pos == 0) & (final_actions == 0)
        rewards[missed_buy] = -term_rew[missed_buy] * 0.5 # Buy Miss Penalty Halved (Weighted 0.5)
        
        missed_sell = sell_opp & (self.env_pos == 1) & (final_actions == 0)
        rewards[missed_sell] = term_rew[missed_sell] * 2.0 # Sell Miss Penalty Doubled (Weighted 2.0)
        
        act_buy = (final_actions == 1)
        rewards[act_buy] = term_rew[act_buy]
        
        # Valid Holding or Sell Execution
        # 1. Sell Exec: PnL - Term (Reward for securing profit AND avoiding risk / sacrificing opportunity)
        act_sell = (final_actions == 2)
        rewards[act_sell] = raw_holding_rew[act_sell] - term_rew[act_sell]
        
        # 2. Holding (Action=0, Pos=1): Reward = 0.0 (Sparse)
        # We strictly DO NOT assign raw_holding_rew here for RL.
        # But we DO keep raw_holding_rew in env_acc_history implicitly via env_last_reward? 
        # Wait, env_last_reward is used for State Diffs. 
        # Requirement: "Holding 期间... 用于计算14维特征".
        # So we must update env_last_reward with raw_holding_rew even if we don't return it for RL?
        # NO. "名义上的reward=0". So RL reward is 0. 
        # But "用于计算d1_rew d2_rew".
        # d1_rew = curr_rew - prev_rew. If we set curr_rew=0, d1 will be wrong (0 - prev).
        # We need a separate tracker for "State Reward" vs "RL Reward".
        # Actually, env_last_reward IS that tracker. 
        # Current implementation of _update_account_history uses env_last_reward.
        # So we should update env_last_reward with raw_holding_rew, BUT return 0 in 'rewards' tensor.
        
        # FINAL SCALE: reward * 5.0 (Only apply to RL rewards)
        rewards = rewards * 5.0
        
        # Update State
        mask_buy = (final_actions == 1)
        self.env_pos[mask_buy] = 1.0
        # Buy Cost: Entry Price = Price * 1.005
        self.env_entry_price[mask_buy] = prices[mask_buy] * 1.005 
        self.env_hold_days[mask_buy] = 0.0
        self.env_trade_max_p[mask_buy] = prices[mask_buy]
        self.env_trade_max_dd[mask_buy] = 0.0
        self.env_trade_start_week_id[mask_buy] = self.all_week_ids[s_idx[mask_buy], curr_t[mask_buy]]
        
        mask_sell = (final_actions == 2)
        self.env_pos[mask_sell] = 0.0
        self.env_trade_count[mask_sell] += 1
        self.env_is_random_trade[mask_sell] = False # Reset flag on sell
        
        mask_hold = (self.env_pos == 1)
        self.env_hold_days[mask_hold] += 1.0
        next_prices = self.all_prices[s_idx, torch.clamp(curr_t + 1, max=self.all_prices.size(1)-1)]
        self.env_trade_max_p[mask_hold] = torch.max(self.env_trade_max_p[mask_hold], next_prices[mask_hold])
        cur_dd = (self.env_trade_max_p[mask_hold] - next_prices[mask_hold]) / (self.env_trade_max_p[mask_hold] + eps)
        self.env_trade_max_dd[mask_hold] = torch.max(self.env_trade_max_dd[mask_hold], cur_dd)
        
        self.env_current_step += 1
        
        limits = self.data_lengths[s_idx] - 25
        dones = (self.env_current_step >= limits)
        
        if dones.any():
            done_idxs = dones.nonzero().squeeze(1)
            self._reset_indices(done_idxs)
        
        # Critical: Update env_last_reward for State Construction (Feature Env)
        # For Holding steps, we store the computed raw_holding_rew (so diffs are correct).
        # For Buy/Sell steps, we store the Term Reward (or Term+Hold).
        # Note: If we missed buy/sell, RL gets penalty, but State Reward should probably be 0? 
        # Or should State Reward reflect the penalty? 
        # Usually State Reward tracks "What happened to the account". 
        # Miss penalties are fictitious RL signals. Account didn't change.
        # So for Miss, State Reward should be 0.
        # For Holding, State Reward should be raw_holding_rew.
        # For Buy/Sell, State Reward should be Term/Real PnL.
        
        state_rewards = torch.zeros_like(rewards)
        state_rewards[mask_hold] = raw_holding_rew[mask_hold] # Real Holding PnL score
        state_rewards[mask_buy] = term_rew[mask_buy]
        state_rewards[mask_sell] = term_rew[mask_sell] + raw_holding_rew[mask_sell]
        
        # Scaling for State (consistent with previous *5.0 logic for state features)
        self.env_last_reward.copy_(state_rewards * 5.0)
        
        self._update_account_history(self.env_range)
        
        return self._get_states(self.env_range), rewards, dones, final_actions
