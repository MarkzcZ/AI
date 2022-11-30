from torch import nn


class LinearRegress(nn.Module):
    def __init__(self, inputsize, outputsize):  # 输入输出均为一维的
        super(LinearRegress, self).__init__()
        self.Linear1 = nn.Linear(in_features=inputsize, out_features=outputsize)

    # forward函数，依次并且输出
    def forward(self, x):
        return self.Linear1(x)


if __name__ == '__main__':
    model = LinearRegress()
