import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class GammaModule(nn.Module):
    def __init__(self, num_problems, init_lambda=1.0, dim=1):
        super(GammaModule, self).__init__()
        self.lambda_emb = nn.Embedding(num_problems + 1, dim)
        nn.init.constant_(self.lambda_emb.weight, np.log(np.exp(init_lambda)-1)) # 逆softplus初始化

    def forward(self, problems, behavior_data):
        lambdas = F.softplus(self.lambda_emb(problems))  # [batch_size, seq_len, 1]
        k = behavior_data.float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        cdf_input = torch.where(k > 0, k-1, torch.zeros_like(k))
        factor = torch.special.gammainc(cdf_input , lambdas)
        
        return factor
