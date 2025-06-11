import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ADD = None
        self.X2 = torch.randn(*(64, 64))
        self.L1 = torch.randn(*(64, 64))

    def forward(self, x):
        output = self.L1 + self.X2
        return output

