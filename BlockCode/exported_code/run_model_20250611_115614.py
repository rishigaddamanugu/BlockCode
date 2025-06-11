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

def main():
    # Setup model and device
    device = torch.device('cpu')
    model = Model()
    model.load_state_dict(torch.load('None'))
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        batch_size = 1
        # Using data from X1
        data = tensor([[ 0.0070,  0.4016, -0.5514,  ...,  1.6123,  0.2881, -0.2747],
        [ 0.4331, -1.0462, -0.3631,  ..., -0.2142, -0.5590,  0.3889],
        [-0.4175,  1.1378,  0.1881,  ...,  0.5389,  0.9635, -0.9182],
        ...,
        [-0.3645, -1.5509,  0.6740,  ..., -0.1254, -0.1029, -0.8152],
        [ 0.3524, -0.9746, -0.5814,  ...,  1.0655,  2.2142, -1.3619],
        [-0.4260, -0.4017,  0.7406,  ..., -0.3476, -1.1544,  1.3526]])
        output = model(data)
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
