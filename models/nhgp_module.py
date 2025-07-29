import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class NHGProcessModule(nn.Module):
    def __init__(self, num_problems, init_mu=2.0, init_sigma=1.0, dim=64):
        super(NHGProcessModule, self).__init__()
        # 可学习的对数时间分布参数
        self.mu_emb = nn.Embedding(num_problems + 1, dim, padding_idx=0)
        self.sigma_emb = nn.Embedding(num_problems + 1, dim, padding_idx=0)
        # 基于先验统计量初始化
        nn.init.normal_(self.mu_emb.weight, mean=np.log(init_mu), std=0.1)
        nn.init.constant_(self.sigma_emb.weight, np.log(np.exp(init_sigma)-1))  # 逆softplus

        init_lambda = 1.0
        self.lambda_emb = nn.Embedding(num_problems + 1, dim, padding_idx=0)
        nn.init.constant_(self.lambda_emb.weight, np.log(np.exp(init_lambda)-1)) # 逆softplus初始化

    
    def forward(self, problems, time_data, behavior_data): 
        log_time = torch.log(time_data.float() + 1e-9).unsqueeze(-1)
        behavior_data = behavior_data.float().unsqueeze(-1)

        # 获取题目特定的lambda参数
        lambdas = F.softplus(self.lambda_emb(problems)) # [batch_size, seq_len, 1]
        
        # 转换数据类型并扩展维度
        cdf_input =  log_time * torch.ones_like(log_time) # [batch_size, seq_len, 1]
        phi = 0.5 *cdf_input * cdf_input
        factors = torch.special.gammainc(phi,  (behavior_data) / lambdas + 1e-9)


        return factors