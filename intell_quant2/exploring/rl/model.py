
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_DDDQN(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, num_layers=2, head_hidden_dim=128):
        """
        Dual-Head Sniper Architecture:
        - Backbone: LSTM (Shared)
        - Entry Head: Decisions for Pos=0 (Wait/Buy)
        - Exit Head: Decisions for Pos=1 (Hold/Sell)
        
        Args:
            head_hidden_dim: 0 for Light Head (Linear), >0 for Heavy Head (Dueling)
        """
        super(LSTM_DDDQN, self).__init__()
        
        self.raw_market_dim = 20
        self.account_dim = 14
        self.head_hidden_dim = head_hidden_dim
        
        # 1. Market Branch (Discrete One-Hot)
        self.register_buffer("active_values", torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32))
        self.market_embed_dim = self.raw_market_dim * 5
        
        # 2. Account Branch
        scales = [1.0, 0.01, 5.0, 150.0, 8.0, 1.0, 30.0, 500.0, 200.0, 5.0, 20.0, 300.0, 100.0, 5.0]
        self.register_buffer("account_scales", torch.tensor(scales, dtype=torch.float32))
        
        self.account_adapter = nn.Sequential(
            nn.Linear(self.account_dim, self.account_dim),
            nn.Tanh()
        )
        
        # 3. Core LSTM
        self.lstm_input_dim = self.market_embed_dim + self.account_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 4. Dual Heads (Always Dueling)
        # Entry Actions: 0=Wait, 1=Buy
        # Exit Actions: 0=Hold, 1=Sell (Mapped from Global Action 2)
        
        if head_hidden_dim == 0:
            # Light Mode: Direct Dueling (No Hidden MLP)
            self.entry_v = nn.Linear(hidden_dim, 1)
            self.entry_a = nn.Linear(hidden_dim, 2)
            self.exit_v = nn.Linear(hidden_dim, 1)
            self.exit_a = nn.Linear(hidden_dim, 2)
        else:
            # Heavy Mode: Dueling with MLP
            self.entry_mlp = nn.Sequential(nn.Linear(hidden_dim, head_hidden_dim), nn.ReLU())
            self.entry_v = nn.Linear(head_hidden_dim, 1)
            self.entry_a = nn.Linear(head_hidden_dim, 2)
            
            self.exit_mlp = nn.Sequential(nn.Linear(hidden_dim, head_hidden_dim), nn.ReLU())
            self.exit_v = nn.Linear(head_hidden_dim, 1)
            self.exit_a = nn.Linear(head_hidden_dim, 2)
        
    def forward(self, x):
        # x: (B, S, 34)
        B, S, _ = x.size()
        
        market_part = x[..., :self.raw_market_dim]
        account_part = x[..., self.raw_market_dim:]
        
        oh = (market_part.unsqueeze(-1) == self.active_values).float()
        market_flat = oh.reshape(B, S, -1)
        
        scaled_account = account_part * self.account_scales
        account_adapted = self.account_adapter(scaled_account)
        
        combined = torch.cat([market_flat, account_adapted], dim=-1)
        
        out, _ = self.lstm(combined)
        last_hidden = out[:, -1, :]
        
        if self.head_hidden_dim == 0:
            # Direct Dueling
            e_v = self.entry_v(last_hidden)
            e_a = self.entry_a(last_hidden)
            q_entry = e_v + (e_a - e_a.mean(dim=1, keepdim=True))
            
            x_v = self.exit_v(last_hidden)
            x_a = self.exit_a(last_hidden)
            q_exit = x_v + (x_a - x_a.mean(dim=1, keepdim=True))
        else:
            # MLP + Dueling
            e_feat = self.entry_mlp(last_hidden)
            e_v = self.entry_v(e_feat)
            e_a = self.entry_a(e_feat)
            q_entry = e_v + (e_a - e_a.mean(dim=1, keepdim=True))
            
            x_feat = self.exit_mlp(last_hidden)
            x_v = self.exit_v(x_feat)
            x_a = self.exit_a(x_feat)
            q_exit = x_v + (x_a - x_a.mean(dim=1, keepdim=True))
            
        return q_entry, q_exit
