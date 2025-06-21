from transformers import AutoTokenizer
from block_types.composite_block import CompositeBlock  # Your base class
from typing import List, Dict, Any, Tuple
import torch
import pandas as pd

class DataConverter(CompositeBlock):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, params or {})
        # This class doesn't override behavior, but adds semantic grouping


class AutoTokenizerBlock(DataConverter):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, params)
        self.tokenizer = None

    def get_mandatory_params(self) -> List[str]:
        return ["model_name"]

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return [("model_name", "str", "distilgpt2")]

    def get_num_input_ports(self) -> int:
        return 1

    def get_num_output_ports(self) -> int:
        return 1

    def to_source_code(self) -> str:
        return f"AutoTokenizer.from_pretrained('{self.params['model_name']}')"

    def forward_expr(self, inputs: List[str]) -> str:
        var = inputs[0] if inputs else "x"
        return f"self.{self.name}({var}, return_tensors='pt')"

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["model_name"])

    def generate_data(self, text: str):
        if self.tokenizer is None:
            self.load_tokenizer()
        return self.tokenizer(text, return_tensors="pt")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": self.params,
            "sub_blocks": [b.to_dict() for b in self.sub_blocks]
        }
    
    def required_imports(self) -> List[str]:
        return ["from transformers import AutoTokenizer"]



class CSVtoTensorBlock(DataConverter):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, params)

    def get_mandatory_params(self):
        return ["file_path"]

    def get_param_info(self):
        return [
            ("file_path", "str", ""),
            ("delimiter", "str", ","),
            ("header", "bool", True)
        ]

    def get_num_input_ports(self) -> int:
        return 1

    def get_num_output_ports(self) -> int:
        return 1

    def to_source_code(self):
        return (
            f"torch.tensor(pd.read_csv('{self.params['file_path']}', "
            f"delimiter='{self.params['delimiter']}', "
            f"header=0 if {self.params['header']} else None).values)"
        )

    def generate_data(self):
        df = pd.read_csv(
            self.params["file_path"],
            delimiter=self.params["delimiter"],
            header=0 if self.params["header"] else None
        )
        return torch.tensor(df.values)

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": self.params,
            "sub_blocks": [b.to_dict() for b in self.sub_blocks]
        }

    def required_imports(self) -> List[str]:
        return ["import torch", "import pandas as pd"]