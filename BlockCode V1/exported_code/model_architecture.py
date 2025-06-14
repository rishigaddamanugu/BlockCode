import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64, 64, bias=True)
        self.x1 = torch.randn(*(64, 64))

    def forward(self, x):
        output = self.L1(self.x1)
        return output
