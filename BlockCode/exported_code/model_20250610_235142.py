import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.X = torch.randn((64,64))
        self.L1 = nn.Linear(64, 64, bias=True)
        self.L2 = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x1 = self.L1(x0)
        output = self.L2(x1)
        return output

if __name__ == '__main__':
    model = Model()
    x0 = torch.randn((64,64))
    output = model(x0)
    print(f'Output shape: {output}')
    print(f'Output shape: {output.shape}')
