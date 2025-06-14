from model_block import ModelBlock
import torch
import pandas as pd
from typing import List, Tuple, Any

class DataBlock:
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
    
    def generate_data(self):
        """Generate a random tensor."""
        return torch.randn(*self.params['shape'])

class CSVtoTensorBlock(DataBlock):
    def get_mandatory_params(self):
        return ["file_path"]

    def get_param_info(self):
        return [
            ("file_path", "str", ""),  # Path to CSV file
            ("delimiter", "str", ","),  # CSV delimiter
            ("header", "bool", True)  # Whether CSV has header
        ]

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 1  # CSVtoTensor takes one input (the file path)

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # CSVtoTensor provides one output tensor

    def to_source_code(self):
        """Return the source code for the data block."""
        return f"torch.tensor(pd.read_csv('{self.params['file_path']}', delimiter='{self.params['delimiter']}', header=0 if {self.params['header']} else None).values)"
    
    def generate_data(self):
        """Generate a tensor from a CSV file."""
        return torch.tensor(pd.read_csv(self.params['file_path'], delimiter=self.params['delimiter'], header=0 if self.params['header'] else None).values)


class TextFileDataBlock(DataBlock):
    def get_mandatory_params(self):
        return ["file_path"]

    def get_param_info(self):
        return [
            ("file_path", "str", "path/to/file.txt"),
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
    
    def generate_data(self):
        """Actually read the file contents."""
        with open(self.params["file_path"], "r") as f:
            return f.read()
