import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.x3 = torch.randn(*(64, 64))
        self.xw = torch.randn(*(64, 64))
        self.SUM2 = None
        self.x1 = torch.randn(*(64, 64))
        self.SUM = None
        self.ADD = None

    def forward(self, x):
        x1 = self.x3
        x2 = self.xw
        x3 = x2.sum(dim=1, keepdim=False)
        x4 = self.x1
        x5 = x4.sum(dim=1, keepdim=False)
        output = x5 + x3
        return output
