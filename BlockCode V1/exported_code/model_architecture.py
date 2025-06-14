import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L3 = nn.Linear(64, 64, bias=True)
        self.x3 = torch.randn(*(64, 64))
        self.L1 = nn.Linear(64, 64, bias=True)
        self.x1 = torch.randn(*(64, 64))
        self.L2 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x1 = self.L3(self.x3)
        x4 = self.L1(self.x1)
        x2 = self.L2(x4)
        output = x2 + x1
        return output
