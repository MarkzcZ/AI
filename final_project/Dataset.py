
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
        x1_tensor = torch.Tensor(self.data.iloc[idx,1])
        
        list1 = [0,0,0,0]
        if self.data.iloc[idx,188]=='1':
            list1 = [1,0,0,0]
        elif self.data.iloc[idx,188]=='2':
            list1 = [0,1,0,0]
        elif self.data.iloc[idx,188]=='3':
            list1 = [0,0,1,0]
        elif self.data.iloc[idx,188]=='4':
            list1 = [0,0,0,1]
        
       
        y_tensor = torch.Tensor(list1).reshape(-1,1)
        
        
        return x1_tensor,y_tensor

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    myTrainData = MyDataSet("origin_breast_cancer_data.csv")
    x1, y1 = myTrainData.__getitem__(19)
    print(x1)
    print(y1)
