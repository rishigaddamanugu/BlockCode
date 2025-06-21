import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_data = torch.randn(*(64, 64))

    def forward(self, x):
        x0 = self.input_data
        output = x0.sum()
        return output
