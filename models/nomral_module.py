import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class NormalModule(nn.Module):
    def __init__(self, num_problems, init_mu=2.0, init_sigma=1.0, dim=64):
        super(NormalModule, self).__init__()

        self.mu_emb = nn.Embedding(num_problems + 1, dim, padding_idx=0)
        self.sigma_emb = nn.Embedding(num_problems + 1, dim, padding_idx=0)

        nn.init.normal_(self.mu_emb.weight, mean=np.log(init_mu), std=0.1)
        nn.init.constant_(self.sigma_emb.weight, np.log(np.exp(init_sigma)-1))  # 逆softplus

    def forward(self, problems, behavior_data):
        mu = self.mu_emb(problems)  # [batch_size, seq_len, 1]
        sigma = F.softplus(self.sigma_emb(problems)) # [batch_size, seq_len, 1]
        log_data = torch.log((behavior_data.float() + 1e-9).unsqueeze(-1))
        normal_dist = torch.distributions.Normal(mu, sigma)
        #  P(X ≤ logT)
        factor = normal_dist.cdf(log_data)
        factor = torch.clamp(factor, min=1e-6, max=1-1e-6)
        return factor