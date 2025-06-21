import torch.nn as nn

import torch
import torch.nn as nn

class composite_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64, 64, bias=True)
        self.L2 = nn.Linear(64, 64, bias=True)
        self.L3 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x0 = self.L1(x)
        x1 = self.L2(x0)
        output = self.L3(x1)
        return output
