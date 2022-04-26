import numpy as np
import torch
from torch.utils.data import Dataset


class WPFDataset(Dataset):
    def __init__(self,
                 data_type: str,
                 data: np.ndarray,
                 input_len: int,
                 output_len: int):
        self.type_list = ['train', 'test', 'val']
        assert data_type in self.type_list
        self.data_type = data_type  # 0: train, 1: val, 2: test
        # (train_size(153*24*6), feature) or (train_size(16*24*6), feature) or (train_size(15*24*6), feature)
        self.data = data    # default feature: 10
        self.input_len = input_len  # default: 24*6
        self.output_len = output_len    # default: 2*24*6

    def __len__(self):
        # Otherwise, if sliding window is not adopted
        if self.data_type not in self.type_list:
            return int((len(self.data) - self.input_len) / self.output_len)
        # In our case, the sliding window is adopted, the number of samples is calculated as follows
        return len(self.data) - self.input_len - self.output_len + 1

    def __getitem__(self, index):
        # Only for customized use.
        # When sliding window not used, e.g. prediction without overlapped input/output sequences
        if self.data_type not in self.type_list:
            index = index * self.output_len
        # Standard use goes here.
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        data_x = torch.tensor(self.data[s_begin:s_end]).to(torch.float32)  # [input_len, feature]
        data_y = torch.tensor(self.data[r_begin:r_end]).to(torch.float32)  # [output_len, feature]
        return data_x, data_y
