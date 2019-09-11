"""
Define neural network architecture for training model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MySmallModel(nn.Module):
    def __init__(self):
        super(MySmallModel, self).__init__()
        self.fc1 = nn.Linear(5, 2)
        self.fc2 = nn.Linear(2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.network1 = MySmallModel()
        self.network2 = MySmallModel()
        self.network3 = MySmallModel()
        
        self.fc1 = nn.Linear(3, 2)
        self.fc_out = nn.Linear(2, 1)
        
    def forward(self, x1, x2, x3):
        x1 = F.relu(self.network1(x1))
        x2 = F.relu(self.network2(x2))
        x3 = F.relu(self.network3(x3))
        
        x = torch.cat((x1, x2, x3), 1)
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
        return x

model = MyModel()
N = 10
x1, x2, x3 = torch.randn(N, 5), torch.randn(N, 5), torch.randn(N, 5)

output = model(x1, x2, x3)