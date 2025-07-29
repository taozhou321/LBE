import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Embedding,
    Linear,
    Dropout,
    ModuleList,
    Sequential,
)
from torch.nn.modules.activation import GELU
from .akt_transformer_layer import AKTTransformerLayer
from .gamma_module import GammaModule
from .nomral_module import NormalModule

class AKT_LBE1(Module):
    def __init__(self, num_questions, num_skills, num_problems, device, **kwargs):
        super(AKT_LBE1 , self).__init__()
        self.num_questions = num_questions
        self.num_skills = num_skills
        self.num_problems = num_problems
        self.device = device
        self.args = kwargs
        self.hidden_size = self.args["hidden_size"]
        self.num_blocks = self.args["num_blocks"]
        self.num_attn_heads = self.args["num_attn_heads"]
        self.kq_same = self.args["kq_same"]
        self.final_fc_dim = self.args["final_fc_dim"]
        self.d_ff = self.args["d_ff"]
        self.l2 = self.args["l2"]
        self.dropout = self.args["dropout"]
        self.model_name = self.args["model_name"]
        
        self.question_embed = Embedding(
            self.num_questions + 1, self.hidden_size, padding_idx=0
        )
        
        self.skill_embed = Embedding(
            self.num_skills + 1, self.hidden_size, padding_idx=0
        )
        
        self.problem_diff_embed = Embedding( # 问题差异
            self.num_problems + 1, 1, padding_idx=0
        )
        self.question_diff_embed = Embedding( # question差异
            self.num_questions + 1, self.hidden_size, padding_idx=0
        ) 
        self.interaction_embed = Embedding(
            2 * (self.num_questions + 1) + 1, self.hidden_size, padding_idx=0
        )

        self.question_encoder = ModuleList(
            [
                AKTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                AKTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                AKTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.behavior_encoder = ModuleList(
            [
                AKTTransformerLayer(
                    d_model=2 * self.hidden_size ,
                    d_feature=(2 * self.hidden_size ) // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.behavior_affect_net = Sequential(
            Linear(2 * self.hidden_size, self.hidden_size),
            GELU(),
        )
        self.behavior_proj = nn.Linear(2 * self.hidden_size + 3* self.hidden_size, 2 * self.hidden_size)

        self.time_module = NormalModule(self.num_problems, dim=self.hidden_size)
        self.hint_module = GammaModule(self.num_problems, dim=self.hidden_size)
        self.attempt_module = GammaModule(self.num_problems, dim=self.hidden_size)

        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(), 
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )

        self.guess_net =  nn.Linear(self.hidden_size, 1)
        self.slip_net = nn.Linear(self.hidden_size, 1)
        self.reset()
        self.loss_fn = nn.BCELoss(reduction="mean")

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_problems + 1 and self.num_problems > 0:
                torch.nn.init.constant_(p, 0.)
    def forward(self, batch):
        problems = batch["problems"]
        questions = batch["questions"] # 用skills表示的question
        responses = batch["responses"]
        q_mat = batch["q_mat"] # (bs, seql_len, num_skills)
        time_taken = batch["time_taken"]
        attempt_count = batch["attempt_count"]
        hint_count = batch["hint_count"]

        pro_diff = self.problem_diff_embed(problems) # 此处不能加sigmoid会导致性能下降
        ques_diff = self.question_diff_embed(questions)
        question_embed_data = self.question_embed(questions) # (bs, seq_len, hs)

        q_embed_data = question_embed_data + pro_diff * ques_diff # (bs, seq_len, hs)
        i_embed_data = self.get_interaction_embed(questions, responses, problems) 

        x, y = q_embed_data, i_embed_data
        for block in self.question_encoder:
            x = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for block in self.interaction_encoder:
            y = block(mask=1, query=y, key=y, values=y, apply_pos=True)
        
        for block in self.knoweldge_retriever:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        implict_skill_state_data = x

        input_x = torch.cat([implict_skill_state_data, q_embed_data], dim=-1)
        o = torch.sigmoid(self.out(input_x)).squeeze() # (bs, seq_len)
        time_factor = self.time_module(problems, time_taken)
        hint_factor = self.hint_module(problems, hint_count)
        attempt_factor = self.attempt_module(problems, attempt_count)
        behaviors = [input_x, # (bs, seq_len, 2 * self.hidden_size)
                     time_factor, # (bs, seq_len) --> (bs, seq_len, 1)
                     hint_factor, # (bs, seq_len) --> (bs, seq_len, 1)
                     attempt_factor
                    ]
        factor_proj = self.behavior_proj(torch.cat(behaviors, dim=-1))

        for block in self.behavior_encoder:
            query_padding = block(mask=0, query=input_x, key=factor_proj, values=factor_proj, apply_pos=True, sparse=True)
        
        behavior_affect = self.behavior_affect_net(query_padding)
        g = torch.sigmoid(self.guess_net(behavior_affect)).squeeze(-1) #(bs, seq_len, 1) --> (bs, seq_len)
        s = torch.sigmoid(self.slip_net(behavior_affect)).squeeze(-1)

        output = (1 - s) * o + g * (1 - o)

        if self.training:
            c_reg_loss = (pro_diff ** 2.).sum() * self.l2
            out_dict = {
                    "pred": output,
                    "true": responses.float(),
                    "c_reg_loss": c_reg_loss,
                }
        else:
            out_dict = {
                "pred": output[:,1:],
                "true": responses[:,1:].float(),
            }
        

        return out_dict
    
    def loss(self, feed_dict, out_dict):
        
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]

        mask = true > -0.9

        loss = self.loss_fn(pred[mask], true[mask]) 
    
        return loss + c_reg_loss, len(pred[mask]), true[mask].sum().item()


    def get_interaction_embed(self, questions, responses, problems):
        masked_responses = responses * (responses > -1).long()
        interactions = questions + self.num_questions * masked_responses
        i_embed_data = self.interaction_embed(interactions)

        return i_embed_data
