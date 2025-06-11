import torch
from typing import Dict, Any, List, Tuple
from model_block import ModelBlock


class OperationBlock(ModelBlock):
    """Base class for operation blocks that perform functional operations rather than layer transformations.
    
    Operations are fundamentally different from layers in that they:
    1. Don't maintain state (no parameters to learn)
    2. Don't need to be instantiated in __init__
    3. Always return None for to_source_code()
    4. Perform functional operations on their inputs
    """
    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, params)
        # Operations don't have sub-blocks
        self.sub_blocks = []

    def to_source_code(self):
        """Operations don't need to be instantiated in __init__."""
        return ""

    def get_mandatory_params(self) -> List[str]:
        """Operations typically don't have mandatory parameters."""
        return []


class AddBlock(OperationBlock):
    def get_num_input_ports(self) -> int:
        return 2  # Add requires 2 inputs

    def get_num_output_ports(self) -> int:
        return 1

    def forward_expr(self, inputs):
        # For Add, we need at least two inputs
        input1 = inputs[0]
        input2 = inputs[1]
        return f"{input1} + {input2}"


class SumBlock(OperationBlock):
    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return [
            ("dim", "int", 1),
            ("keepdim", "bool", False)
        ]

    def get_num_input_ports(self) -> int:
        return 1

    def get_num_output_ports(self) -> int:
        return 1

    def forward_expr(self, inputs):
        input_var = inputs[0] if inputs else "x"
        dim = self.params.get("dim", 1)
        keepdim = self.params.get("keepdim", False)
        return f"{input_var}.sum(dim={dim}, keepdim={keepdim})"


class MatmulBlock(OperationBlock):
    def get_num_input_ports(self) -> int:
        return 2  # Matrix multiplication requires 2 inputs

    def get_num_output_ports(self) -> int:
        return 1

    def forward_expr(self, inputs):
        # For Matmul, we need at least one input, use 'x' for missing inputs
        input1 = inputs[0] if inputs else "x"
        input2 = inputs[1] if len(inputs) > 1 else "x"
        return f"torch.matmul({input1}, {input2})" 