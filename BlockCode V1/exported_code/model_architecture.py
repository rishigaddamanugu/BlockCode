import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.text_input = open('text.txt', 'r').read()
        self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

    def forward(self, x):
        x0 = self.model(**x)
        x1 = self.text_input
        output = self.tokenizer(x1, return_tensors='pt')
        return output
