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


class CompositeBlock:
    def __init__(self, name, params=None, sub_blocks=None, input_blocks=None, output_blocks=None, sub_connection_info=None):
        self.name = name
        self.params = params or {}
        self.sub_blocks = sub_blocks or []
        self.input_blocks = input_blocks or []
        self.output_blocks = output_blocks or []
        self.sub_connection_info = sub_connection_info or []  # ✅ new line


    def get_mandatory_params(self):
        return []

    def get_param_info(self):
        return []

    def required_imports(self):
        return []

    def get_num_input_ports(self):
        return len(self.input_blocks)

    def get_num_output_ports(self):
        return len(self.output_blocks)

    def add_block(self, block):
        self.sub_blocks.append(block)

    def to_source_code(self):
        return f"{self.name}()"

    def forward_expr(self, inputs):
        return f"{self.name}({', '.join(inputs)})"



    def to_dict(self):
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": self.params,
            "sub_blocks": [b.to_dict() for b in self.sub_blocks]
        }


class LinearBlock(CompositeBlock):
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


class HuggingFaceCompositeBlock(CompositeBlock):
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
            ("model_name", "str", "distilgpt2"),
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