import numpy as np
import torch
import copy
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier

from model_es import LSTM_DDDQN

class EvolutionAgent:
    def __init__(self, weights_dict, rule_mask):
        self.weights_dict = weights_dict # Dictionary of GPU Tensors
        self.rule_mask = rule_mask 
        self.fitness = 0.0

class BayesianES:
    def __init__(self, input_dim, action_dim, pop_size=64, sigma=0.05, device="cpu"):
        self.pop_size = pop_size
        self.sigma = sigma
        
        if device == "cpu":
            if torch.backends.mps.is_available(): self.device = torch.device("mps")
            elif torch.cuda.is_available(): self.device = torch.device("cuda")
            else: self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.num_rules = 9 
        self.rule_names = ["Buy_A1", "Buy_A2", "Buy_A3", "Buy_A4", "Buy_A5", "Sell_B1", "Sell_B2", "Sell_B3", "Sell_B4"]
        
        self.template_model = LSTM_DDDQN(input_dim, action_dim, head_hidden_dim=128).to(self.device)
        # Store mean weights as GPU Tensors
        self.mean_params = {k: v.clone().detach() for k, v in self.template_model.named_parameters()}
        
        self.rule_priors = np.ones(self.num_rules) * 0.5 
        self.history_configs = []
        self.history_rewards = []

    def ask(self) -> List[EvolutionAgent]:
        population = []
        for i in range(self.pop_size):
            # 1. Mutate Weights directly on GPU
            perturbed_weights = {}
            for k, v in self.mean_params.items():
                noise = torch.randn(v.shape, device=self.device)
                perturbed_weights[k] = v + self.sigma * noise
            
            # 2. Sample Rules (Bayesian Guidance)
            r_mask = torch.zeros(self.num_rules, dtype=torch.bool, device=self.device)
            for r_idx in range(self.num_rules):
                prob = 0.5 if np.random.random() < 0.2 else self.rule_priors[r_idx]
                r_mask[r_idx] = bool(np.random.random() < prob)
            
            if not r_mask[:5].any(): r_mask[np.random.randint(0, 5)] = True
            if not r_mask[5:].any(): r_mask[np.random.randint(5, 9)] = True
            
            population.append(EvolutionAgent(perturbed_weights, r_mask))
        return population

    def tell(self, population: List[EvolutionAgent]):
        # Sort by fitness (CPU side)
        population.sort(key=lambda x: x.fitness, reverse=True)
        top_k = population[:max(1, self.pop_size // 10)]
        
        # 1. Update Mean Weights on GPU
        with torch.no_grad():
            for k in self.mean_params.keys():
                # Average of top performers
                new_v = torch.stack([a.weights_dict[k] for a in top_k]).mean(dim=0)
                self.mean_params[k] = 0.9 * self.mean_params[k] + 0.1 * new_v
        
        # 2. Update Bayesian Priors
        for r_idx in range(self.num_rules):
            avg_presence = np.mean([a.rule_mask[r_idx].item() for a in top_k])
            self.rule_priors[r_idx] = 0.95 * self.rule_priors[r_idx] + 0.05 * avg_presence
            
        for a in population:
            self.history_configs.append(a.rule_mask.cpu().numpy())
            self.history_rewards.append(a.fitness)
            
        if len(self.history_configs) % (self.pop_size * 5) == 0:
            self._apply_ml_distillation()
            
        return top_k[0].fitness, np.mean([a.fitness for a in population])

    def _apply_ml_distillation(self):
        X = np.array(self.history_configs[-1000:])
        y = np.array(self.history_rewards[-1000:])
        threshold = np.percentile(y, 80)
        y_labels = (y >= threshold).astype(int)
        sample_weights = np.abs(y) + 0.01
        
        if len(np.unique(y_labels)) < 2: return
        
        rf = RandomForestClassifier(n_estimators=50)
        rf.fit(X, y_labels, sample_weight=sample_weights)
        
        importances = rf.feature_importances_
        for i in range(self.num_rules):
            winners_X = X[y_labels == 1]
            winning_avg = winners_X[:, i].mean()
            shift = importances[i] * (winning_avg - self.rule_priors[i])
            self.rule_priors[i] = np.clip(self.rule_priors[i] + shift, 0.1, 0.9)

    def distill_rules(self):
        report = ["--- RULE PRIORS (EVOLVED) ---"]
        for i, name in enumerate(self.rule_names):
            report.append(f"{name:8s}: {self.rule_priors[i]:.2f}")
        return "\n".join(report)

    def get_master_state_dict(self):
        return self.mean_params
