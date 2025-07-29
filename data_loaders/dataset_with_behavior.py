import torch
from torch.utils.data import Dataset
from .data_wit_behavior import DATAWithBehavior

class KT_DatasetWithBehavior(Dataset):
    def __init__(self, seqlen, separate_char, min_seq_len, q_mat, data_path, device=torch.device("cuda:0")) -> None:
        super().__init__()
        self.device = device
        self.dat = DATAWithBehavior(seqlen, separate_char, min_seq_len)
        self.q_data, self.p_data, self.r_data,  \
            self.time_taken_data, self.attempt_cnt_data, self.hint_cnt_data = self.dat.load_data(data_path)
        self.q_mat = q_mat.astype(int)

    def __getitem__(self, index):

        return {"questions": torch.from_numpy(self.q_data[index]).long().to(self.device),  # 用技能组合表示问题以防止过拟合
                "problems": torch.from_numpy(self.p_data[index]).long().to(self.device),
                "responses": torch.from_numpy(self.r_data[index]).long().to(self.device), 
                
                "time_taken": torch.from_numpy(self.time_taken_data[index]).float().to(self.device),
                "attempt_count": torch.from_numpy(self.attempt_cnt_data[index]).long().to(self.device),
                "hint_count": torch.from_numpy(self.hint_cnt_data[index]).long().to(self.device),

                "q_mat": torch.from_numpy(self.q_mat[self.p_data[index].astype(int)]).int().to(self.device),
                }
    
    def __len__(self):
        return len(self.q_data)