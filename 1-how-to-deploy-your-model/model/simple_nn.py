import torch
import torch.nn as nn

# 定义简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 28*28 = 784 输入节点, 128 输出节点
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)    # 输出10类，对应MNIST的10个数字

    def forward(self, x):
        x = x.view(-1, 784)  # flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

