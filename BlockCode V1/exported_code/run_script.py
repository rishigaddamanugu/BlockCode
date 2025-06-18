import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
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
        data = AutoTokenizer.from_pretrained('distilgpt2')
        output = model(data)
    print(f'Inference output: {output}')
    print(f'Inference output shape: {output.shape}')


if __name__ == '__main__':
    main()
