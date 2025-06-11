import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64, 64, bias=True)
        self.X = torch.randn(*(64, 64))

    def forward(self, x):
        output = self.L1(x)
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
        # Using data from X
        data = tensor([[-1.6839,  0.6579, -0.2620,  ...,  0.0449, -1.2418, -0.9837],
        [ 0.2500, -1.1552, -1.1534,  ...,  0.1014,  1.3535,  1.3529],
        [ 0.2396, -1.1242, -0.6055,  ...,  0.8610,  0.1706, -0.6161],
        ...,
        [-0.5465, -0.5392, -0.1889,  ...,  0.3188,  0.2048,  1.3318],
        [-1.7514,  0.7057, -1.1412,  ..., -2.5670, -0.0087, -0.0450],
        [ 0.3866, -1.4233,  0.1527,  ...,  0.5883,  0.9101,  0.3297]])
        output = model(data)
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
