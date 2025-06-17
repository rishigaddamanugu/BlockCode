import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM
import pandas as pd

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.LLM = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
        self.text_input = open('text.txt', 'r').read()
        self.tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')

    def forward(self, x):
        x0 = self.LLM(**x)
        x1 = self.text_input
        output = self.tokenizer(x1, return_tensors='pt')
        return output
