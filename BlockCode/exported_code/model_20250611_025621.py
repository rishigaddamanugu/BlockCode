import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64, 64, bias=True)
        self.L2 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x0 = self.L1(x)
        x1 = self.L2(x0)
        return output

if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 64)  # Example input
    output = model(x)
    print(f'Output shape: {output.shape}')
