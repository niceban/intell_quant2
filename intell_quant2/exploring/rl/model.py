
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_DDDQN(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256, num_layers=2):
        """
        DETERMINISTIC SENSITIVITY-BOOSTED ARCHITECTURE:
        - Market: One-Hot [-2, -1, 1, 2] -> 80 dims.
        - Account: Fixed Pre-Scaling -> Linear -> Tanh -> 14 dims.
        - Diffs are scaled 10x/100x harder to capture micro-trends.
        """
        super(LSTM_DDDQN, self).__init__()
        
        self.raw_market_dim = 20
        self.account_dim = 14
        
        # 1. Market Branch (Discrete One-Hot)
        self.register_buffer("active_values", torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32))
        self.market_embed_dim = self.raw_market_dim * 5
        
        # 2. Account Branch (Deterministic Pre-Scaling + Tanh)
        # Vector: [Pos, Days, CurRet, AvgRet, MaxDD, Rew, D1_Cur, D1_Avg, D1_DD, D1_Rew, D2_Cur, D2_Avg, D2_DD, D2_Rew]
        scales = [
            # --- Base (6) ---
            1.0,    # Pos
            0.01,   # Days (AM:42 -> 0.42, 100days->1.0)
            5.0,    # CurRet (AM:0.26 -> 1.3, 20% -> 1.0)
            150.0,  # AvgRet (AM:0.006 -> 0.9)
            8.0,    # MaxDD (AM:0.12 -> 0.96)
            1.0,    # Rew (AM:0.44 -> 0.44, Better discrimination for large rewards)
            
            # --- D1 (4) ---
            30.0,   # D1_CurRet
            500.0,  # D1_AvgRet
            200.0,  # D1_MaxDD
            5.0,    # D1_Rew (AM:0.09 -> 0.45)
            
            # --- D2 (4) ---
            20.0,   # D2_CurRet
            300.0,  # D2_AvgRet
            100.0,  # D2_MaxDD
            5.0     # D2_Rew (AM:0.14 -> 0.7)
        ]
        self.register_buffer("account_scales", torch.tensor(scales, dtype=torch.float32))
        
        # Learnable Adapter: Maps scaled inputs to Tanh-friendly latent space
        self.account_adapter = nn.Sequential(
            nn.Linear(self.account_dim, self.account_dim),
            nn.Tanh() # Deterministic Bounded Output [-1, 1]
        )
        
        # 3. Core LSTM
        self.lstm_input_dim = self.market_embed_dim + self.account_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 4. Heads
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        # x: (B, S, 34)
        B, S, _ = x.size()
        
        market_part = x[..., :self.raw_market_dim]
        account_part = x[..., self.raw_market_dim:]
        
        # 1. Market One-Hot
        oh = (market_part.unsqueeze(-1) == self.active_values).float()
        market_flat = oh.reshape(B, S, -1)
        
        # 2. Account Pre-Scaling + Tanh
        # Apply fixed manual scales first to bring data to O(1)
        scaled_account = account_part * self.account_scales
        # Apply Learnable Linear + Tanh
        account_adapted = self.account_adapter(scaled_account)
        
        # 3. Combine
        combined = torch.cat([market_flat, account_adapted], dim=-1)
        
        # 4. LSTM
        out, _ = self.lstm(combined)
        last_hidden = out[:, -1, :]
        
        value = self.value_head(last_hidden)
        advantage = self.advantage_head(last_hidden)
        
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
