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
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        batch_size = 1
        # Using data from T
        data = tensor([[-0.7424,  0.8155, -0.2028,  ...,  0.3542, -0.2192, -0.6226],
        [ 1.4250,  0.2623, -1.2634,  ..., -0.3497, -0.5425, -0.3071],
        [ 1.4442, -0.9559, -0.9689,  ...,  2.6294, -1.0731, -1.0512],
        ...,
        [-0.2750,  1.4199, -0.2297,  ..., -0.5306, -0.2122,  1.0271],
        [ 0.5015, -1.0021, -0.4475,  ...,  0.3010,  0.0803,  0.8841],
        [ 0.4988, -0.5692,  1.8664,  ...,  0.3515, -0.7274, -0.1230]])
        output = model(data)
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
