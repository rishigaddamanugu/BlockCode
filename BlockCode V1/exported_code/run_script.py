import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from model_architecture import Model

def main():
    # Setup model and device
    device = torch.device('cpu')
    model = Model()
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        batch_size = 1
        # Using data from tokenizer
        data = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')
        output = model(data)
    print(f'Inference output: {output}')
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
