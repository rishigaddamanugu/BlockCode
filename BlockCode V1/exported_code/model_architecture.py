import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.LLM = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
        self.input_text = open('text.txt', 'r').read()
        self.tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')

    def forward(self, x):
        x0 = self.LLM(**x)
        x1 = self.input_text
        output = self.tokenizer(x1, return_tensors='pt')
        return output
