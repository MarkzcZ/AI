
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
        x1_tensor = (self.data['area_mean']/50).to_list()
        
        list1 = []
        for x in self.data['diagnosis']:
            if( x == 'M'):
                list1.append(1)
            else:
                list1.append(0)
       
        y_tensor = list1
        
        
        return x1_tensor[idx],y_tensor[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    myTrainData = MyDataSet("origin_breast_cancer_data.csv")
    x1, y1 = myTrainData.__getitem__(19)
    print(x1)
    print(y1)
