import torch
import torch.nn as nn

torch.manual_seed(4321)
x = torch.rand(size=(8, 2), requires_grad=True)
y = torch.randint(low=0, high=3, size=(8,))

for_pass_a1 = torch.zeros(8, 1)
for_pass_a2 = torch.zeros(8, 1)

y1_in = torch.zeros(8, 1)
y2_in = torch.zeros(8, 1)
y3_in = torch.zeros(8, 1)

for i in range(0, 8, 1):
    for_pass_a1[i] = 0.48 * x[i, 0] - 0.51 * x[i, 1] + 0.23
    for_pass_a2[i] = -0.43 * x[i, 0] - 0.48 * x[i, 1] + 0.05

for i in range(0, 8, 1):
    y1_in[i] = -0.99 * for_pass_a1[i] - 0.66 * for_pass_a2[i] + 0.32
    y2_in[i] = 0.36 * for_pass_a1[i] + 0.34 * for_pass_a2[i] - 0.44
    y3_in[i] = -0.75 * for_pass_a1[i] - 0.66 * for_pass_a2[i] + 0.7
# softmax
y_hat = torch.zeros(8, 3)

net_1 = nn.Softmax(dim=0)
for i in range(0, 8, 1):
    temp = torch.Tensor([y1_in[i], y2_in[i], y3_in[i]])
    y_hat[i, :] = net_1(temp)

max_index = torch.argmax(y_hat, dim=1)
print(y_hat)
print(max_index)

print(y)
y.long()
max_index.long()

# 计算损失值
net_loss = nn.CrossEntropyLoss()
total_loss = net_loss(max_index, y)