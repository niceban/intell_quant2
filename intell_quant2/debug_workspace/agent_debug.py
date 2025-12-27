import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from model_debug import LSTM_DDDQN

class EvolutionaryBufferGPU:
    # ... (rest of the class remains same, assume it's copied)

    def __init__(self, capacity, device, alpha=0.6):
        self.capacity = capacity
        self.device = torch.device(device)
        self.alpha = alpha
        
        # Monolithic Storage
        self.obs = torch.zeros((capacity, 20, 34), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((capacity, 20, 34), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.scores = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        
        # 4-Quarter Partitioning
        # 0: BX, 1: SX, 2: BM, 3: SM
        self.q_size = capacity // 4
        self.counts = [0, 0, 0, 0] # Valid items in each quarter
        self.starts = [0, self.q_size, self.q_size*2, self.q_size*3] # Absolute start indices

    def __len__(self): 
        return sum(self.counts)

    def get_stats(self):
        total = self.__len__()
        if total == 0: return 0, [0.0, 0.0, 0.0, 0.0]
        ratios = [c / max(1, total) for c in self.counts]
        return total, ratios

    def push_batch(self, states, actions, rewards, next_states, dones, pos_before):
        n = actions.size(0)
        if n == 0: return
        
        # Classify Samples into 4 Categories
        # 0: BX, 1: SX, 2: BM, 3: SM
        cats = torch.full((n,), -1, dtype=torch.long, device=self.device)
        cats[actions == 1] = 0 # BX
        cats[actions == 2] = 1 # SX
        
        miss_mask = (actions == 0) & (rewards != 0)
        cats[miss_mask & (pos_before == 0)] = 2 # BM
        cats[miss_mask & (pos_before == 1)] = 3 # SM
        
        valid_mask = (cats != -1)
        if not valid_mask.any(): return
        
        for c in range(4):
            c_mask = (cats == c)
            count = c_mask.sum().item()
            if count == 0: continue
            
            # Quarter metadata
            start_abs = self.starts[c]
            curr_count = self.counts[c]
            
            # Prepare Target Indices for this category
            if curr_count + count <= self.q_size:
                # CASE 1: Still has space, use sequential indices
                indices_to_overwrite = torch.arange(count, device=self.device) + curr_count
                abs_dest = indices_to_overwrite + start_abs
                self.counts[c] += count
            else:
                # CASE 2: Tournament Eviction required
                # If there's some remaining space, fill it first
                n_fill = max(0, self.q_size - curr_count)
                n_tourn = count - n_fill
                
                abs_dest_fill = torch.tensor([], dtype=torch.long, device=self.device)
                if n_fill > 0:
                    abs_dest_fill = torch.arange(n_fill, device=self.device) + curr_count + start_abs
                    self.counts[c] = self.q_size
                
                # Tournament for the rest
                abs_dest_tourn = torch.tensor([], dtype=torch.long, device=self.device)
                if n_tourn > 0:
                    # K=4 factor: pick candidates
                    k = 4
                    # If incoming count is larger than q_size (unlikely), clamp it
                    safe_n_tourn = min(n_tourn, self.q_size)
                    n_cand = min(self.q_size, safe_n_tourn * k)
                    
                    # Randomly pick candidates within the quarter
                    cand_rel = torch.randint(0, self.q_size, (n_cand,), device=self.device)
                    cand_abs = cand_rel + start_abs
                    
                    # Retrieve scores and find the N lowest
                    cand_scores = self.scores[cand_abs]
                    # topk(largest=False) gives us the indices of the minimum values
                    _, worst_in_cand = torch.topk(cand_scores, safe_n_tourn, largest=False)
                    abs_dest_tourn = cand_abs[worst_in_cand]
                
                abs_dest = torch.cat([abs_dest_fill, abs_dest_tourn])
                # In case of large batch > q_size, take only the last 'count' data
                if abs_dest.numel() < count:
                    # This only happens if a single batch is bigger than the entire quarter
                    # We just use all destination indices we found
                    pass 

            # Data to write (truncate if needed to match abs_dest)
            valid_count = abs_dest.numel()
            c_states = states[c_mask][:valid_count]
            c_next_states = next_states[c_mask][:valid_count]
            c_actions = actions[c_mask][:valid_count]
            c_rewards = rewards[c_mask][:valid_count]
            c_dones = dones[c_mask][:valid_count]
            
            # Write to Buffer
            self.obs[abs_dest] = c_states
            self.next_obs[abs_dest] = c_next_states
            self.actions[abs_dest] = c_actions
            self.rewards[abs_dest] = c_rewards
            self.dones[abs_dest] = c_dones
            
            # Initial Priority Score
            sc = torch.abs(c_rewards)
            sc = torch.where(c_rewards > 0, sc * 3.0, sc)
            self.scores[abs_dest] = 1.0 + sc


    def sample(self, batch_size, beta=0.4):
        # Enforce Balanced Sampling: 1/4 from each if available
        n_per_q = batch_size // 4
        all_indices = []
        
        for c in range(4):
            cnt = self.counts[c]
            if cnt == 0: continue
            
            start_abs = self.starts[c]
            # Prioritized Sampling within the Quarter
            # Slice scores for this quarter
            # Note: scores tensor is monolithic, we need to access [start:start+cnt] carefully
            # Because buffer might be full but ptr wrapped, so valid data is scattered?
            # No, 'counts' just tells us how many valid. 'ptrs' tells us where head is.
            # If not full, data is [start : start+cnt].
            # If full, data is [start : start+q_size].
            
            valid_len = self.counts[c]
            # Indices relative to start_abs
            rel_indices = torch.arange(valid_len, device=self.device)
            abs_indices = rel_indices + start_abs
            
            probs = self.scores[abs_indices] ** self.alpha
            # Sample
            n_target = n_per_q
            # If other quarters are empty, take more from here? 
            # For simplicity, stick to n_per_q, and fill remainder randomly from all valid.
            
            chosen_rel = torch.multinomial(probs, min(valid_len, n_target), replacement=True)
            all_indices.append(chosen_rel + start_abs)
            
        if not all_indices: return None, None
        
        indices = torch.cat(all_indices)
        
        # If we didn't get enough (e.g. some quarters empty), fill from whatever we have
        if indices.numel() < batch_size:
            rem = batch_size - indices.numel()
            # Pool all valid indices from all quarters
            # This is expensive to construct every time. 
            # Just resample from the indices we already picked (oversample)
            extra = indices[torch.randint(0, indices.numel(), (rem,), device=self.device)]
            indices = torch.cat([indices, extra])
            
        # If we got too many (e.g. n_per_q * 4 > batch due to rounding?), truncate
        if indices.numel() > batch_size:
            indices = indices[:batch_size]
            
        return (self.obs[indices], self.actions[indices], self.rewards[indices], self.next_obs[indices], self.dones[indices]), indices

    def update_priorities(self, indices, td_errors):
        re = self.rewards[indices]
        comp = torch.where(re > 0, torch.abs(re) * 3.0, torch.abs(re))
        self.scores[indices] = td_errors.squeeze() + comp + 1e-5

class DQNAgent:
    def __init__(self, input_dim, action_dim, lr=2e-5, gamma=0.99, buffer_size=100000, batch_size=1024, device="cpu", tau=0.005, head_hidden_dim=128):
        self.device = torch.device(device); self.tau = tau; self.batch_size = batch_size; self.gamma = gamma
        
        # Dual-Head Model
        self.policy_net = LSTM_DDDQN(input_dim, action_dim, head_hidden_dim=head_hidden_dim).to(self.device)
        self.target_net = LSTM_DDDQN(input_dim, action_dim, head_hidden_dim=head_hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = EvolutionaryBufferGPU(capacity=buffer_size, device=self.device)
        self.epsilon = 1.0; self.epsilon_min = 0.05; self.epsilon_decay = 0.9995 

    def get_buffer_info(self):
        return self.memory.get_stats()

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            # Random action: This should ideally follow the Unified Random Logic (0.8 buy, 0.2 sell)
            # But here we just return a placeholder, the Env will override it in Random Mode.
            # However, for consistency, let's return a valid random action range.
            if state.ndim == 3: return torch.randint(0, 3, (state.size(0),), device=self.device)
            return random.randrange(3)
            
        with torch.no_grad():
            st = state if isinstance(state, torch.Tensor) else torch.as_tensor(state, device=self.device, dtype=torch.float32)
            if st.ndim == 2: st = st.unsqueeze(0)
            
            # Forward Dual Heads
            q_entry, q_exit = self.policy_net(st)
            
            # Extract Pos from state (Batch, Sequence, Features) -> Last step, Index 20
            # Account features start at index 20. Pos is the first one.
            current_pos = st[:, -1, 20]
            
            # Select Action based on Pos
            # Pos=0 -> Entry Head (0:Wait, 1:Buy) -> Global (0, 1)
            # Pos=1 -> Exit Head (0:Hold, 1:Sell) -> Global (0, 2)
            
            actions = torch.zeros(st.size(0), dtype=torch.long, device=self.device)
            
            # Entry Logic
            empty_mask = (current_pos < 0.5)
            if empty_mask.any():
                act_entry = q_entry[empty_mask].argmax(dim=1) # 0 or 1
                actions[empty_mask] = act_entry
                
            # Exit Logic
            hold_mask = (current_pos >= 0.5)
            if hold_mask.any():
                act_exit = q_exit[hold_mask].argmax(dim=1) # 0 or 1
                # Map 1 (Sell) to Global 2
                global_act_exit = torch.where(act_exit == 1, torch.tensor(2, device=self.device), torch.tensor(0, device=self.device))
                actions[hold_mask] = global_act_exit
                
            return actions if state.ndim == 3 else actions.item()
            
    def update(self):
        if len(self.memory) < self.batch_size: return None
        (states, actions, rewards, next_states, dones), indices = self.memory.sample(self.batch_size)
        
        # 1. Get Current Q (Dual Head)
        curr_q_entry, curr_q_exit = self.policy_net(states)
        
        # Determine which head applies to which sample
        # Using 'actions' to infer is risky because Action 0 is ambiguous (Wait or Hold).
        # Must use 'states' Pos info.
        current_pos = states[:, -1, 20]
        empty_mask = (current_pos < 0.5)
        hold_mask = (current_pos >= 0.5)
        
        # Gather Q values corresponding to taken actions
        # Entry: Action 0 or 1. Gather directly.
        # Exit: Action 0 or 2. Map 2->1 for gather.
        
        q_pred = torch.zeros_like(rewards)
        
        # Entry Part
        if empty_mask.any():
            # Actions should be 0 or 1
            act_entry = actions[empty_mask]
            q_pred[empty_mask] = curr_q_entry[empty_mask].gather(1, act_entry.unsqueeze(1)).squeeze(1)
            
        # Exit Part
        if hold_mask.any():
            # Actions should be 0 or 2. Map 2->1.
            act_exit = actions[hold_mask]
            act_exit_mapped = (act_exit == 2).long()
            q_pred[hold_mask] = curr_q_exit[hold_mask].gather(1, act_exit_mapped.unsqueeze(1)).squeeze(1)

        # 2. Get Target Q (Dual Head)
        with torch.no_grad():
            next_q_entry, next_q_exit = self.target_net(next_states)
            # Route based on NEXT state pos
            next_pos = next_states[:, -1, 20]
            next_empty_mask = (next_pos < 0.5)
            next_hold_mask = (next_pos >= 0.5)
            
            # Double DQN Logic: Use Policy Net to select action, Target Net to eval
            # But for simplicity and stability with dual heads, let's stick to Max Q first.
            # Or Double DQN:
            # next_act_entry = self.policy_net(next_states)[0].argmax(1)
            # ...
            # Let's use Standard DQN Max for now to reduce complexity risks in this major refactor.
            
            max_next_q = torch.zeros_like(rewards)
            
            if next_empty_mask.any():
                max_next_q[next_empty_mask] = next_q_entry[next_empty_mask].max(1)[0]
            
            if next_hold_mask.any():
                max_next_q[next_hold_mask] = next_q_exit[next_hold_mask].max(1)[0]
                
            target_q = rewards + (~dones).float() * self.gamma * max_next_q

        # 3. Loss
        diff = q_pred - target_q
        td_errors = torch.abs(diff).detach()
        
        # --- HEAVY DEBUG PROBE ---
        if random.random() < 0.01: # 1% sample rate for logs
            with torch.no_grad():
                p_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100.0)
                w_norm = sum(p.norm().item() for p in self.policy_net.parameters())
                print(f"\n[DEBUG AGENT] Target Range: {target_q.min().item():.1f} to {target_q.max().item():.1f}")
                print(f"[DEBUG AGENT] Pred Range: {q_pred.min().item():.1f} to {q_pred.max().item():.1f}")
                print(f"[DEBUG AGENT] Weight Norm: {w_norm:.1f} | Grad Norm (clipped): {p_norm:.2f}")
                if q_pred.min() < -1000:
                    print("!!! ALARM: Q-Values are diving deep !!!")
        # -------------------------

        self.memory.update_priorities(indices, td_errors)
        
        loss = F.smooth_l1_loss(q_pred, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update()
        
        # Stats
        q_min = q_pred.min().detach()
        q_max = q_pred.max().detach()
        
        return {'loss': loss, 'mean_q': q_pred.mean().detach(), 'min_q': q_min, 'max_q': q_max, 'mean_err': td_errors.mean()}
    
    def soft_update(self):
        for t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t.data.copy_(self.tau * p.data + (1.0 - self.tau) * t.data)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    def save(self, path): 
        torch.save(self.policy_net.state_dict(), path)