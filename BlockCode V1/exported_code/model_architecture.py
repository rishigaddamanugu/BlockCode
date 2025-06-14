import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64, 64, bias=True)
        self.L2 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x0 = self.L1(x)
        output = self.L2(x0)
        return output
