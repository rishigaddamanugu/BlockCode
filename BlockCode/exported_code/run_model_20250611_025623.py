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
        x1 = self.L2(x0)
        return output

def main():
    # Setup model and device
    device = torch.device('cpu')
    model = Model()
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        batch_size = 1
        # Using data from T1
        data = tensor([[-0.0185,  0.6945,  0.5872,  ...,  1.2400, -1.8838,  1.5601],
        [ 1.1138,  0.7905, -1.1051,  ...,  1.8461, -0.8419,  2.8595],
        [ 0.9395,  0.7794,  1.1538,  ...,  0.3527, -0.5981,  0.6640],
        ...,
        [ 0.4729, -0.1444, -0.5825,  ...,  3.1207,  1.3390,  1.7242],
        [-0.7498,  0.1317,  1.1741,  ..., -0.5419, -1.6963, -0.3489],
        [-0.5171, -0.6935,  1.1918,  ..., -0.2864,  0.8219,  0.3598]])
        output = model(data)
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
