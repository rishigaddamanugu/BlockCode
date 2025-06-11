import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64, 64, bias=True)

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
        # Using data from x1
        data = tensor([[-0.8335, -0.4254, -0.9035,  ...,  1.1762, -0.9446,  0.3073],
        [ 1.4816,  0.8511,  0.3796,  ...,  1.3173, -0.4668,  0.7311],
        [ 0.4215, -1.4590, -0.0761,  ..., -0.8032,  0.7129,  1.6264],
        ...,
        [-0.1311,  0.8992, -0.8245,  ..., -0.2904,  1.7698,  0.7909],
        [-0.6092, -0.8413,  1.7020,  ..., -0.4900, -0.4855, -1.7885],
        [ 0.3973, -0.4169,  2.4531,  ..., -0.3983, -1.3176,  0.1056]])
        output = model(data)
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
