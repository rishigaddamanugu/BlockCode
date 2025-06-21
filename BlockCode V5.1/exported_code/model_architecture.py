import torch.nn as nn

import torch
import torch.nn as nn

class composite_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_data2 = torch.randn(*(64, 64))
        self.input_data = torch.randn(*(64, 64))
        self.L1 = nn.Linear(64, 64, bias=True)
        self.L2 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x0 = self.input_data2
        x1 = self.input_data
        x2 = self.L1(x1)
        x3 = self.L2(x0)
        output = x2 + x3
        return output
