import torch
from torch import nn


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(188, 70),
            nn.ReLU(inplace=False),
            nn.Linear(70, 40),
            nn.ReLU(inplace=False),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(40, 4)
        )

        # softmax激活函数，返回十个概率值
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_x):
        y_out = self.layer1(input_x)
        y_out = self.layer2(y_out)
        y_out = self.softmax(y_out)
        return y_out


if __name__ == '__main__':
    x = torch.randn(1, 188)
    print(f'输入尺寸为:{x.shape}')
    model = FCN()
    output = model(x)
    print(f'输出尺寸为:{output}')
    print(model)
