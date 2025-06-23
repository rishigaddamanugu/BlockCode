import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from block_types.composite_block import CompositeBlock

class InstanceMethodBlock(CompositeBlock):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name  # Variable name to assign result to
        self.params = params or {}
        self.inputs = []  # Connected input blocks (first must be the object)
        self.outputs = []  # Connected output blocks

    def get_mandatory_params(self) -> List[str]:
        return []

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return []

    def required_imports(self) -> List[str]:
        return []

    def get_num_input_ports(self) -> int:
        return 1  # [object, arg1, ...]

    def get_num_output_ports(self) -> int:
        return 1

    def to_source_code(self):
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method_name,
            "params": self.params,
            "type": self.__class__.__name__,
        }

class InstanceSumBlock(InstanceMethodBlock):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name  # Variable name to assign result to
        self.params = params or {}
        self.inputs = []  # Connected input blocks (first must be the object)
        self.outputs = []  # Connected output blocks

    def get_mandatory_params(self) -> List[str]:
        return []

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return []

    def required_imports(self) -> List[str]:
        return []

    def get_num_input_ports(self) -> int:
        return 1  # [object, arg1, ...]

    def get_num_output_ports(self) -> int:
        return 1

    def _hidden_forward_expr(self, input_vars: List[str]) -> str:
        if not input_vars:
            raise ValueError("No inputs provided to MethodBlock")

        object_var = input_vars[0]
        method_args = ", ".join(input_vars[1:]) if len(input_vars) > 1 else ""
        return f"{object_var}.sum({method_args})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method_name,
            "params": self.params,
            "type": self.__class__.__name__,
        }