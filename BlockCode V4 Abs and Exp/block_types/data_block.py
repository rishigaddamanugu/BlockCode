from block_types.composite_block import CompositeBlock
import torch
import pandas as pd
from typing import List, Tuple, Any
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)


class DataBlock(CompositeBlock):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}
        self.validate_params()

    def validate_params(self):
        """Validate that all mandatory parameters are present."""
        missing = [param for param in self.get_mandatory_params() if param not in self.params]
        if missing:
            raise ValueError(f"Missing mandatory parameters: {', '.join(missing)}")

    def get_mandatory_params(self):
        """Return list of mandatory parameter names."""
        return []

    def get_param_info(self):
        """Return list of (param_name, param_type, default_value) tuples."""
        return []

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 0  # Data blocks don't take inputs by default

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # Data blocks provide one output by default
    
    def to_source_code(self):
        """Return the source code for the data block."""
        raise NotImplementedError("Subclasses must implement to_source_code")
    
    def generate_data(self):
        """Generate a random tensor."""
        raise NotImplementedError("Subclasses must implement generate_data")
    
    def required_imports(self) -> List[str]:
        return ["import torch"]

class RandomTensorBlock(DataBlock):
    def get_mandatory_params(self):
        return ["shape"]

    def get_param_info(self):
        return [
            ("shape", "tuple", (64, 64)),  # Shape of the tensor
        ]

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 0  # RandomTensor doesn't take any inputs

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # RandomTensor provides one output tensor
    
    def to_source_code(self):
        """Return the source code for the data block."""
        return f"torch.randn(*{self.params['shape']})"
    
    def forward_expr(self, inputs):
        return f"self.{self.name}" # This allows us to keep export code generic while setting a variable to itself basically as the output
        
    def generate_data(self):
        """Generate a random tensor."""
        return torch.randn(*self.params['shape'])
    
    def required_imports(self) -> List[str]:
        return ["import torch"]



class TextFileDataBlock(DataBlock):
    def get_mandatory_params(self):
        return ["file_path"]

    def get_param_info(self):
        return [
            ("file_path", "str", "text.txt"),
        ]

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 0  # It doesn't require any input

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # It outputs the text content
    
    def to_source_code(self):
        """Generate the code for loading text from a file."""
        return f"open('{self.params['file_path']}', 'r').read()"

    def forward_expr(self, inputs):
        return f"self.{self.name}" # This allows us to keep export code generic while setting a variable to itself basically as the output
    
    def generate_data(self):
        """Actually read the file contents."""
        with open(self.params["file_path"], "r") as f:
            return f.read()
    
    def required_imports(self) -> List[str]:
        return ["import pandas as pd"]

