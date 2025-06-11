import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.x2 = torch.randn((5, 5))
        self.x1 = torch.randn((5, 5))
        self.ADD = None

    def forward(self, x):
        output = x1 + x0
        return output

if __name__ == '__main__':
    model = Model()
    x0 = torch.randn((5, 5))
    x1 = torch.randn((5, 5))
    output = model(x0)
    print(f'Output shape: {output}')
    print(f'Output shape: {output.shape}')
