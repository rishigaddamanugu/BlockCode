import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


class ModelBlock:
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.sub_blocks = []  # List of nested blocks
        self.validate_params()

    def validate_params(self):
        """Validate that all mandatory parameters are present."""
        missing = [param for param in self.get_mandatory_params() if param not in self.params]
        if missing:
            raise ValueError(f"Missing mandatory parameters: {', '.join(missing)}")

    def get_mandatory_params(self) -> List[str]:
        """Return list of mandatory parameter names."""
        return []

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        """Return list of (param_name, param_type, default_value) tuples."""
        return []

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 1  # Default to 1 input port

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # Default to 1 output port

    def add_block(self, block: 'ModelBlock'):
        """Add a sub-block to this block."""
        self.sub_blocks.append(block)

    def to_source_code(self):
        """Return PyTorch layer code for __init__, or None if this is a functional op."""
        if not self.sub_blocks:
            raise NotImplementedError()
        else:
            # For composite blocks, generate code for all sub-blocks
            code = []
            for block in self.sub_blocks:
                code.append(f"        self.{block.name} = {block.to_source_code()}")
            return "\n".join(code)

    def forward_expr(self, inputs):
        """Return a line of forward-pass code using inputs (list of var names).
        
        Args:
            inputs: List of input variable names, one for each input port.
                   If an input port has no connection, its corresponding input will be 'x'.
        """
        if not self.sub_blocks:
            # If no inputs provided, use 'x' as default
            if not inputs:
                return f"self.{self.name}(x)"
            # If only one input, use it directly
            if len(inputs) == 1:
                return f"self.{self.name}({inputs[0]})"
            # For multiple inputs, pass them as separate arguments
            return f"self.{self.name}({', '.join(inputs)})"
        else:
            # For composite blocks, chain the forward expressions
            current_input = inputs[0] if inputs else "x"
            for block in self.sub_blocks:
                current_input = block.forward_expr([current_input])
            return current_input

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": self.params,
            "sub_blocks": [block.to_dict() for block in self.sub_blocks]
        }


class LinearBlock(ModelBlock):
    def get_mandatory_params(self) -> List[str]:
        return ["in_features", "out_features"]

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return [
            ("in_features", "int", 64),
            ("out_features", "int", 64),
            ("bias", "bool", True)
        ]

    def get_num_input_ports(self) -> int:
        return 1

    def get_num_output_ports(self) -> int:
        return 1

    def to_source_code(self):
        bias_str = f", bias={self.params.get('bias', True)}"
        return f"nn.Linear({self.params['in_features']}, {self.params['out_features']}{bias_str})"

    def forward_expr(self, inputs):
        input_var = inputs[0] if inputs else "x"
        return f"self.{self.name}({input_var})"


class ReLUBlock(ModelBlock):
    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return [
            ("inplace", "bool", False)
        ]

    def get_num_input_ports(self) -> int:
        return 1

    def get_num_output_ports(self) -> int:
        return 1

    def to_source_code(self):
        inplace_str = f", inplace={self.params.get('inplace', False)}"
        return f"nn.ReLU({inplace_str})"

    def forward_expr(self, inputs):
        input_var = inputs[0] if inputs else "x"
        return f"self.{self.name}({input_var})"


class Conv2dBlock(ModelBlock):
    def get_mandatory_params(self) -> List[str]:
        return ["in_channels", "out_channels", "kernel_size"]

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return [
            ("in_channels", "int", 1),
            ("out_channels", "int", 32),
            ("kernel_size", "int", 3),
            ("stride", "int", 1),
            ("padding", "int", 0),
            ("dilation", "int", 1),
            ("groups", "int", 1),
            ("bias", "bool", True),
            ("padding_mode", "str", "zeros")
        ]

    def get_num_input_ports(self) -> int:
        return 1

    def get_num_output_ports(self) -> int:
        return 1

    def to_source_code(self):
        params = self.params
        return f"nn.Conv2d({params['in_channels']}, {params['out_channels']}, " \
               f"kernel_size={params['kernel_size']}, stride={params.get('stride', 1)}, " \
               f"padding={params.get('padding', 0)}, dilation={params.get('dilation', 1)}, " \
               f"groups={params.get('groups', 1)}, bias={params.get('bias', True)}, " \
               f"padding_mode='{params.get('padding_mode', 'zeros')}')"

    def forward_expr(self, inputs):
        input_var = inputs[0] if inputs else "x"
        return f"self.{self.name}({input_var})"


class CompositeBlock(ModelBlock):
    """A block that can contain other blocks."""
    def __init__(self, name: str, blocks: List[ModelBlock] = None):
        super().__init__(name)
        self.sub_blocks = blocks or []

    def get_num_input_ports(self) -> int:
        if not self.sub_blocks:
            return 1
        # Count input blocks (blocks with no inputs)
        return len([block for block in self.sub_blocks if not block.inputs])

    def get_num_output_ports(self) -> int:
        if not self.sub_blocks:
            return 1
        # Count output blocks (blocks with no outputs)
        return len([block for block in self.sub_blocks if not block.outputs])

    def to_source_code(self):
        if not self.sub_blocks:
            return None
        code = []
        for block in self.sub_blocks:
            if block.to_source_code() is not None:  # Only add if not None
                code.append(f"        self.{block.name} = {block.to_source_code()}")
        return "\n".join(code)

    def forward_expr(self, inputs):
        if not self.sub_blocks:
            return inputs[0] if inputs else "x"
        
        # Create a mapping of block names to their outputs
        var_names = {}
        current_input = inputs[0] if inputs else "x"
        
        # Process each block in sequence
        for block in self.sub_blocks:
            if block in self.sub_blocks:
                # Get the input for this block
                if block == self.sub_blocks[0]:  # First block
                    block_input = current_input
                else:
                    # Find the last block's output
                    prev_block = self.sub_blocks[self.sub_blocks.index(block) - 1]
                    block_input = var_names.get(prev_block.name, current_input)
                
                # Generate the forward expression
                output = block.forward_expr([block_input])
                var_names[block.name] = output
        
        # Return the output of the last block
        return var_names[self.sub_blocks[-1].name]

    def to_dict(self):
        return {
            "name": self.name,
            "type": "CompositeBlock",
            "sub_blocks": [block.to_dict() for block in self.sub_blocks]
        }