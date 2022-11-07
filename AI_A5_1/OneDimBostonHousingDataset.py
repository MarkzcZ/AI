# CRIM input MEDV as target
from torch.utils.data import Dataset
import pandas as pd
import torch


# 将数据导入到DataSet
class MyDataSet(Dataset):
    # 初始化时从文件中把数据读进来
    def __init__(self, dataDir):
        self.data = pd.read_csv(dataDir)

    def __getitem__(self, idx):
        # 将数据转换成二维Tensor，并且选择asv的某一列进行提取
        x1_tensor = torch.Tensor(self.data['CRIM'].to_list()).reshape(-1, 1)
        x2_tensor = torch.Tensor(self.data['RM'].to_list()).reshape(-1, 1)
        x3_tensor = torch.Tensor(self.data['AGE'].to_list()).reshape(-1, 1)
        y_tensor = torch.Tensor(self.data['MEDV'].to_list()).reshape(-1, 1)
        return x1_tensor[idx],x2_tensor[idx],x3_tensor[idx],y_tensor[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    myTrainData = MyDataSet("data/boston_housing_data.csv")
    x1, y1 = myTrainData.__getitem__(0)
    print(x1)
    print(y1)
