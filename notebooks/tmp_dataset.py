import numpy as np
import torch
from torch.utils.data import Dataset


class TMPDataset(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 input_len: int = 288,
                 output_len: int = 1):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len

        self.mean = np.nanmean(self.data)
        self.std = np.nanstd(self.data)
        self.threshold_low = self.mean - 3 * self.std
        self.threshold_high = self.mean + 3 * self.std

        self.data[self.data > self.threshold_high] = np.nan
        self.data[self.data < self.threshold_low] = np.nan

        self.mean = np.nanmean(self.data)
        self.std = np.nanstd(self.data)
        self.threshold_low = self.mean - 3 * self.std
        self.threshold_high = self.mean + 3 * self.std

        self.valid_begin = input_len
        self.normal = (self.data > self.threshold_low) & (self.data < self.threshold_high) & ~np.isnan(self.data)

        self.valid_index = self.input_len
        # 存在3sigma外的异常值
        # 循环找到训练数据和验证数据全部正常的切片
        while not self.normal[self.valid_index - self.input_len: self.valid_index + 1].all():
            # 找到index后第一个正常值作为新的index值
            self.valid_index += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index为要预测的值的下标
        data_x, data_y = None, None
        if index >= self.valid_begin:
            # 若index位置之前能够分出训练集
            # 若存在3sigma外的异常值，循环找到训练数据和验证数据全部正常的切片
            while not self.normal[index - self.input_len: index + 1].all():
                # 找到index后第一个正常值作为新的index值
                index += 1
                if index > self.__len__():
                    index = self.valid_index
            data_x, data_y = self.data[index - self.input_len: index], self.data[index]
        else:
            # 若index位置之前不能分出训练集
            if self.normal[: index + 1].all():
                # 若数据全部正常，在前面补0
                pad_size = self.valid_begin - index
                if index == 0:
                    data_x, data_y = np.zeros(pad_size), self.data[index]
                else:
                    data_x, data_y = np.concatenate([np.zeros(pad_size), self.data[: index]]), self.data[index]
            else:
                # 若存在异常数据，则使用之前保存的合法index
                data_x, data_y = self.data[self.valid_index - self.input_len: self.valid_index], \
                                 self.data[self.valid_index]

        data_x = torch.unsqueeze(torch.tensor(data_x).to(torch.float32), 1)
        data_y = torch.unsqueeze(torch.tensor(data_y).to(torch.float32), 0)
        return data_x, data_y
