import numpy
from torch.utils.data import Dataset
import pandas as pd
import torch

class dataloader(Dataset):
    def __init__(self, dataDir):
        self.data = pd.read_csv(dataDir)
        self.mean_value = []
        self.std_value = []

        for j in range(188):
            self.mean_value.append(torch.mean(torch.Tensor(self.data.iloc[:, j])))
            self.std_value.append(torch.std(torch.Tensor(self.data.iloc[:, j])))


    def __getitem__(self, idx):
        # taking the data according to the index
        input_tensor = []

        for i in range(188):
            # 减去均值再除方差
            input_tensor.append((self.data.iloc[idx, i] - self.mean_value[i]) / self.std_value[i])

        # labels
        label = self.data.iloc[idx, 188]
        input_tensor = torch.Tensor(input_tensor)
        # 188 attributes and 1 label
        # 因为是4分类的问题，标签的大小也为4
        tag = numpy.zeros(4)
        tag[label - 1] = 1
        # 扩张一个维度
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor, torch.Tensor(tag)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    myTrainData = dataloader(r'ecg_data.csv')
    x_data, labels = myTrainData.__getitem__(6)
    print(x_data.shape)
    print(labels)