import torch.nn as nn
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_data = torch.randn(*(64, 64))
        self.x1 = torch.randn(*(64, 64))
        self.L1 = nn.Linear(64, 64, bias=True)
        self.L2 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x1 = self.input_data
        x2 = self.x1
        x3 = self.L1(x2)
        output = self.L2(x3)
        return output
