import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)


class ModelBlock:
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.sub_blocks = []
        self.validate_params()

    def validate_params(self):
        missing = [param for param in self.get_mandatory_params() if param not in self.params]
        if missing:
            raise ValueError(f"Missing mandatory parameters: {', '.join(missing)}")

    def get_mandatory_params(self) -> List[str]:
        return []

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return []

    def required_imports(self) -> List[str]:
        """Returns a list of required imports for this block type.
        
        Returns:
            List[str]: List of import statements required for this block type.
        """
        return []

    def get_num_input_ports(self) -> int:
        if not self.sub_blocks:
            return 1
        return len([b for b in self.sub_blocks if not b.inputs])

    def get_num_output_ports(self) -> int:
        if not self.sub_blocks:
            return 1
        return len([b for b in self.sub_blocks if not b.outputs])

    def add_block(self, block: 'ModelBlock'):
        self.sub_blocks.append(block)

    def to_source_code(self):
        if not self.sub_blocks:
            raise NotImplementedError()
        return "\n".join(
            f"        self.{b.name} = {b.to_source_code()}"
            for b in self.sub_blocks if b.to_source_code() is not None
        )

    def forward_expr(self, inputs):

        if inputs:
            if not self.sub_blocks:
                return f"self.{self.name}({', '.join(inputs)})"

            current_input = inputs[0]
            for b in self.sub_blocks:
                current_input = b.forward_expr([current_input])
            return current_input
        return f"self.{self.name}(x)"

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": self.params,
            "sub_blocks": [b.to_dict() for b in self.sub_blocks]
        }


class LinearBlock(ModelBlock):
    def get_mandatory_params(self) -> List[str]:
        return ["in_features", "out_features"]

    def required_imports(self) -> List[str]:
        return ["import torch.nn as nn"]

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

    def required_imports(self) -> List[str]:
        return ["import torch.nn as nn"]

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

    def required_imports(self) -> List[str]:
        return ["import torch.nn as nn"]

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
    


from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from model_block import ModelBlock  # Your base class
from typing import List, Dict, Any
import torch


class HuggingFaceModelBlock(ModelBlock):
    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, params or {})
        self.model = None  # Lazy-loaded at runtime
        self.is_expanded = False

        # Supported HuggingFace task types
        self.supported_task_types = {
            "causal_lm": AutoModelForCausalLM,
            "masked_lm": AutoModelForMaskedLM,
            "seq2seq": AutoModelForSeq2SeqLM,
            "classification": AutoModelForSequenceClassification,
            "generic": AutoModel
        }

    def required_imports(self) -> List[str]:
        return [
            "from transformers import AutoModel",
            "from transformers import AutoModelForCausalLM",
            "from transformers import AutoModelForMaskedLM",
            "from transformers import AutoModelForSeq2SeqLM",
            "from transformers import AutoModelForSequenceClassification"
        ]

    def load_model(self):
        """Instantiate the model from Hugging Face."""
        if self.model is not None:
            return
        task_type = self.params["task_type"]
        model_class = self.supported_task_types.get(task_type, AutoModel)
        self.model = model_class.from_pretrained(self.params["model_name"])

    def to_source_code(self):
        """Generate model instantiation line (no variable assignment)."""
        task_type = self.params["task_type"]
        model_name = self.params["model_name"]
        class_map = {
            "causal_lm": "AutoModelForCausalLM",
            "masked_lm": "AutoModelForMaskedLM",
            "seq2seq": "AutoModelForSeq2SeqLM",
            "classification": "AutoModelForSequenceClassification",
            "generic": "AutoModel"
        }
        model_class = class_map.get(task_type, "AutoModel")
        return f"{model_class}.from_pretrained('{model_name}')"

    def forward_expr(self, inputs: List[str]) -> str:
        """Generate the forward call expression (no assignment)."""
        input_var = inputs[0] if inputs else "x"
        return f"self.{self.name}(**{input_var})"

    def get_mandatory_params(self) -> List[str]:
        return ["model_name", "task_type"]

    def get_param_info(self) -> List[Tuple[str, str, Any]]:
        return [
            ("model_name", "str", "sshleifer/tiny-gpt2"),
            ("task_type", "str", "causal_lm"),
            ("device", "str", "cpu"),
            ("max_length", "int", 512),
            ("num_labels", "int", 2)  # Only relevant for classification
        ]

    def get_num_input_ports(self) -> int:
        return 1  # Tokenized input dict

    def get_num_output_ports(self) -> int:
        return 1  # Model output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": self.params,
            "sub_blocks": [b.to_dict() for b in self.sub_blocks]
        }