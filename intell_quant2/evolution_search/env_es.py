import numpy as np
import random
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class VectorizedTradingEnvGPU:
    """
    Optimized Shared-Memory Env:
    - Symbols are stored ONCE on GPU.
    - Each env slot only tracks its current index.
    - Drastically reduces memory footprint.
    """
    def __init__(self, data_bundle: Dict, device: str, num_envs: int = 8192):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.state_dim = 34
        
        # SHARED DATA (One copy on GPU)
        self.all_market_data = data_bundle['market_tensors'].to(self.device) # [num_symbols, max_l, 20]
        self.all_prices = data_bundle['price_tensors'].to(self.device)       # [num_symbols, max_l]
        self.all_state_indices = data_bundle['state_indices_tensors'].to(self.device) # [num_symbols, max_l, 20]
        self.all_week_ids = data_bundle['week_id_tensors'].to(self.device)   # [num_symbols, max_l]
        self.data_lengths = data_bundle['lengths'].to(self.device)           # [num_symbols]
        self.num_symbols = self.all_market_data.size(0)
        
        self._precompute_terms()
        
        # PER-ENV STATE
        self.env_symbol_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_current_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_pos = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_entry_price = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_hold_days = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_trade_max_p = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_trade_max_dd = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_last_reward = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.env_trade_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_trade_start_week_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.env_is_random_trade = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # Account History must be per-env, but we only need history for the current sequence window
        # For simplicity, we use a rolling window or small cache if possible.
        # But here we stick to the original plan but smaller: [num_envs, 20, 14]
        # Wait, the state indices logic needs access to absolute time indices.
        # We'll use a fixed-size history buffer for the current 'active' segment.
        max_l = self.all_prices.size(1)
        self.env_acc_history = torch.zeros((num_envs, max_l + 10, 14), dtype=torch.float32, device=self.device)
        self.env_range = torch.arange(self.num_envs, device=self.device)

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
        s_idxs = new_sym_idxs
        max_s_vals = self.data_lengths[s_idxs] - 30
        start = 25
        max_s_vals = torch.clamp(max_s_vals, min=start + 1)
        range_widths = (max_s_vals - start + 1).float()
        offsets = (torch.rand(n, device=self.device) * range_widths).long()
        self.env_current_step[env_indices] = start + offsets
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
            cur_ret[is_long] = (prices[is_long] * 0.995 / (entry_prices[is_long] + eps)) - 1.0
            avg_ret[is_long] = cur_ret[is_long] / torch.clamp(hold_days[is_long], min=5.0)
        m0 = torch.stack([self.env_pos[env_indices], hold_days, cur_ret, avg_ret, max_dd, last_rew], dim=1)
        m_1 = self.env_acc_history[env_indices, curr_t - 1, 2:6]
        m_2 = self.env_acc_history[env_indices, curr_t - 2, 2:6]
        full_vec = torch.cat([m0, m0[:, 2:6] - m_1, m_1 - m_2], dim=1)
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
        # Fancy indexing for shared tensors
        target_indices = self.all_state_indices[s_idx, curr_t] 
        market_obs = self.all_market_data[s_idx.unsqueeze(1), target_indices]
        acc_obs = self.env_acc_history[env_indices.unsqueeze(1), target_indices]
        return torch.cat([market_obs, acc_obs], dim=-1)

    def step(self, agent_actions: torch.Tensor, rule_mask: torch.Tensor = None, random_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-8
        s_idx = self.env_symbol_idx
        curr_t = self.env_current_step
        m_data = self.all_market_data[s_idx, curr_t] 
        
        # ATOMIC RULES
        w_dif = m_data[:, 0]; w_dea = m_data[:, 1]; w_bar = m_data[:, 2]
        w_k   = m_data[:, 3]; w_d   = m_data[:, 4]
        m_dea = m_data[:, 12]; m_ama = m_data[:, 17]
        buy_rules = torch.stack([(w_d == 2), (w_k == 2), (w_dif > 0), (w_bar > 0), ((m_dea > 0) & (m_ama > 0))], dim=1)
        sell_rules = torch.stack([(w_d == -2), (w_dea == -2), ((w_dea < 0) & (w_dif < 0) & (w_bar < 0)), (m_dea < 0)], dim=1)
        
        if rule_mask is None:
            buy_opp = buy_rules[:, 4] & buy_rules[:, 0]
            sell_opp = sell_rules[:, 0] | sell_rules[:, 1] | sell_rules[:, 2]
        else:
            buy_opp = (buy_rules & rule_mask[:, :5]).any(dim=1)
            sell_opp = (sell_rules & rule_mask[:, 5:]).any(dim=1)
        
        force_sell_cond = (w_d < 0) & (w_k < 0) & (w_dif < 0) & (w_dea < 0) & (w_bar < 0)
        curr_week_ids = self.all_week_ids[s_idx, curr_t]
        is_same_week = (curr_week_ids == self.env_trade_start_week_id)
        immediate_cancel = (self.env_pos == 1) & is_same_week & (~buy_opp)

        final_actions = torch.zeros_like(agent_actions)
        buy_candidates = (self.env_pos == 0) & buy_opp
        if random_mode:
            do_buy = buy_candidates & (torch.rand(self.num_envs, device=self.device) < 0.8)
            final_actions[do_buy] = 1
            self.env_is_random_trade[do_buy] = True
        else:
            final_actions[buy_candidates & (agent_actions == 1)] = 1
            self.env_is_random_trade[buy_candidates & (agent_actions == 1)] = False

        sell_candidates = (self.env_pos == 1) & sell_opp
        is_rnd = self.env_is_random_trade
        if sell_candidates.any():
            rnd_sell_mask = sell_candidates & is_rnd
            final_actions[rnd_sell_mask & (torch.rand(self.num_envs, device=self.device) < 0.2)] = 2
            agt_sell_mask = sell_candidates & (~is_rnd)
            final_actions[agt_sell_mask & (agent_actions == 2)] = 2

        final_actions[(self.env_pos == 1) & (final_actions == 1)] = 0
        final_actions[(self.env_pos == 0) & (final_actions == 2)] = 0
        final_actions[(~buy_opp) & (final_actions == 1)] = 0
        final_actions[(~sell_opp) & (final_actions == 2)] = 0
        
        prices = self.all_prices[s_idx, curr_t]
        liquid_price = prices * 0.995
        cur_ret_tmp = (liquid_price / (self.env_entry_price + eps)) - 1.0
        h_days_tmp = self.env_hold_days + 1.0
        avg_ret_tmp = cur_ret_tmp / torch.clamp(h_days_tmp, min=5.0)
        t_max_tmp = torch.max(self.env_trade_max_p, prices)
        dd_tmp = (t_max_tmp - prices) / (t_max_tmp + eps)
        t_dd_tmp = torch.max(self.env_trade_max_dd, dd_tmp)

        raw_holding_rew = torch.zeros(self.num_envs, device=self.device)
        pos_mask, neg_mask = (cur_ret_tmp > 0), (cur_ret_tmp <= 0)
        raw_holding_rew[pos_mask] = (1.0 + cur_ret_tmp[pos_mask])**0.5 - 1.0 + (h_days_tmp[pos_mask]**0.33) * avg_ret_tmp[pos_mask] - t_dd_tmp[pos_mask]
        raw_holding_rew[neg_mask] = -(1.0 - cur_ret_tmp[neg_mask])**2 + 1.0 + (h_days_tmp[neg_mask]**0.33) * avg_ret_tmp[neg_mask] - t_dd_tmp[neg_mask]

        must_sell = (self.env_pos == 1) & (force_sell_cond | immediate_cancel)
        final_actions[must_sell] = 2
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        term_rew = self.all_terms[s_idx, curr_t]
        rewards[buy_opp & (self.env_pos == 0) & (final_actions == 0)] = -term_rew[buy_opp & (self.env_pos == 0) & (final_actions == 0)] * 0.5 
        rewards[(self.env_pos == 1) & sell_opp & (final_actions == 0)] = term_rew[(self.env_pos == 1) & sell_opp & (final_actions == 0)] 
        rewards[final_actions == 1] = term_rew[final_actions == 1]
        rewards[final_actions == 2] = torch.max(raw_holding_rew[final_actions == 2] - term_rew[final_actions == 2], raw_holding_rew[final_actions == 2])
        
        rewards = self._quantize_reward(rewards * 5.0)
        
        self.env_pos[final_actions == 1] = 1.0
        self.env_entry_price[final_actions == 1] = prices[final_actions == 1] * 1.005 
        self.env_hold_days[final_actions == 1] = 0.0
        self.env_trade_max_p[final_actions == 1] = prices[final_actions == 1]
        self.env_trade_max_dd[final_actions == 1] = 0.0
        self.env_trade_start_week_id[final_actions == 1] = curr_week_ids[final_actions == 1]
        self.env_pos[final_actions == 2] = 0.0
        self.env_trade_count[final_actions == 2] += 1
        self.env_is_random_trade[final_actions == 2] = False 
        
        mask_hold = (self.env_pos == 1)
        self.env_hold_days[mask_hold] += 1.0
        next_prices = self.all_prices[s_idx, torch.clamp(curr_t + 1, max=self.all_prices.size(1)-1)]
        self.env_trade_max_p[mask_hold] = torch.max(self.env_trade_max_p[mask_hold], next_prices[mask_hold])
        self.env_trade_max_dd[mask_hold] = torch.max(self.env_trade_max_dd[mask_hold], (self.env_trade_max_p[mask_hold] - next_prices[mask_hold]) / (self.env_trade_max_p[mask_hold] + eps))
        
        self.env_current_step += 1
        if (self.env_current_step >= self.data_lengths[s_idx] - 25).any(): self._reset_indices((self.env_current_step >= self.data_lengths[s_idx] - 25).nonzero().squeeze(1))
        
        state_rewards = torch.zeros_like(rewards)
        state_rewards[mask_hold] = raw_holding_rew[mask_hold] 
        state_rewards[final_actions == 1] = term_rew[final_actions == 1]
        state_rewards[final_actions == 2] = term_rew[final_actions == 2] + raw_holding_rew[final_actions == 2]
        self.env_last_reward.copy_(self._quantize_reward(state_rewards * 5.0))
        self._update_account_history(self.env_range)
        return self._get_states(self.env_range), rewards, (self.env_current_step >= self.data_lengths[s_idx] - 25), final_actions

    def _quantize_reward(self, r_tensor):
        bins = torch.tensor([-5.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0], device=self.device)
        diff = torch.abs(r_tensor.unsqueeze(1) - bins.unsqueeze(0))
        return bins[diff.argmin(dim=1)]