import torch
import torch.nn as nn
from model_block import ModelBlock
from data_block import DataBlock
from typing import Dict, Any, List, Tuple

class RunBlock:
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
        return 2  # Run blocks take both model and data inputs

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # Run blocks provide one output by default

    def generate_code(self, model_block, data_block):
        """Generate the code for this block."""
        if not model_block:
            raise ValueError(f"No model block provided to {self.name}")
        if not data_block:
            raise ValueError(f"No data block provided to {self.name}")
            
        raise NotImplementedError("Subclasses must implement generate_code")

class InferenceBlock(RunBlock):
    def get_mandatory_params(self):
        return ["batch_size"]

    def get_param_info(self):
        return [
            ("batch_size", "int", 1),
            ("device", "str", "cpu"),
            ("model_path", "str", ""),       # Optional: For loading weights
            ("task_type", "str", "generic"), # Optional: Determines forward behavior
        ]

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 2  # Takes both model and data inputs

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # Provides model predictions

    def generate_code(self, model_block, data_block, data_converter):
        data_block = data_block or data_converter
        ## **TODO** Reconfigure to use model_block to import and use specific model instead of generic model
        params = self.params
        task_type = params.get("task_type", "generic")

        code = []
        code.append("    # Setup model and device")
        code.append(f"    device = torch.device('{params['device']}')")
        code.append("    model = Model()")  # This will be replaced in final export

        if params.get('model_path'):
            code.append(f"    model.load_state_dict(torch.load('{params['model_path']}'))")

        code.append("    model.to(device)")
        code.append("    model.eval()")
        code.append("")
        code.append("    # Run inference")
        code.append("    with torch.no_grad():")
        code.append(f"        batch_size = {params['batch_size']}")
        code.append(f"        # Using data from {data_block.name}")
        code.append(f"        data = {data_block.to_source_code()}")

        print("TASK TYPE")
        if task_type == "causal_lm":
            code.append("        output = model.generate(**data)")
        elif task_type in {"classification", "masked_lm", "seq2seq"}:
            code.append("        output = model(**data)")
        else:
            code.append("        output = model(data)")

        code.append("    print(f'Inference output: {output}')")
        code.append("    print(f'Inference output shape: {output.shape}')")
        return code


class TrainingBlock(RunBlock):
    def get_mandatory_params(self):
        return ["epochs", "batch_size", "learning_rate"]

    def get_param_info(self):
        # return [
        #     ("epochs", "int", 10),
        #     ("batch_size", "int", 32),
        #     ("learning_rate", "float", 0.001),
        #     ("optimizer", "str", "adam"),
        #     ("device", "str", "cpu"),
        #     ("save_path", "str", None),  # For saving trained models
        # ]

        return [
            ("epochs", "int", 10),
            ("batch_size", "int", 32),
            ("learning_rate", "float", 0.001),
            ("optimizer", "str", "adam"),
            ("device", "str", "cpu"),
            ("save_path", "str", "trained_model"),  # For saving trained models
            ("loss_type", "str", "mse_loss"),
            ("optimizer", "str", "Adam")
        ]

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 3  # Takes both model and data inputs

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # Provides trained model

    def generate_code(self, model_block, data_block):
        params = self.params
        code = []
        code.append("    # Setup model, device, and optimizer")
        code.append(f"    device = torch.device('{params['device']}')")
        code.append("    model = Model()")
        code.append("    model.to(device)")
        code.append("    model.train()")
        
        
        code.append(f"    optimizer = torch.optim.{params['optimizer']}(model.parameters(), lr={params['learning_rate']})")
        code.append("")
        code.append("    # Training loop")
        code.append(f"    epochs = {params['epochs']}")
        code.append(f"    batch_size = {params['batch_size']}")
        code.append("    for epoch in range(epochs):")
        code.append(f"        # Using data from {data_block.name}")
        code.append(f"        data = {data_block.to_source_code()}")
        code.append("        optimizer.zero_grad()")
        code.append("        output = model(data)")
        code.append(f"        loss = nn.functional.{params['loss_type']}(output, data)  # Example loss")
        code.append("        loss.backward()")
        code.append("        optimizer.step()")
        code.append("        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')")
        
        if params.get('save_path'):
            code.append("")
            code.append("    # Save the trained model")
            code.append(f"    torch.save(model.state_dict(), '{params['save_path']}.pt')")
        
        return code

class EvaluationBlock(RunBlock):
    def get_mandatory_params(self):
        return ["metrics"]

    def get_param_info(self):
        return [
            ("metrics", "list", ["accuracy"]),
            ("batch_size", "int", 32),
            ("device", "str", "cpu"),
        ]

    def get_num_input_ports(self) -> int:
        """Return the number of input ports this block requires."""
        return 2  # Takes both model and data inputs

    def get_num_output_ports(self) -> int:
        """Return the number of output ports this block provides."""
        return 1  # Provides evaluation metrics

    def generate_code(self, model_block, data_block):
        params = self.params
        code = []
        code.append("    # Setup model and device")
        code.append(f"    device = torch.device('{params['device']}')")
        code.append("    model = Model()")
        code.append("    model.to(device)")
        code.append("    model.eval()")
        code.append("")
        code.append("    # Evaluation")
        code.append("    with torch.no_grad():")
        code.append(f"        batch_size = {params['batch_size']}")
        code.append(f"        # Using data from {data_block.name}")
        code.append(f"        data = {data_block.to_source_code()}")
        code.append("        output = model(data)")
        
        # Add metric computations
        metrics = params.get('metrics', ['accuracy'])
        for metric in metrics:
            if metric == 'accuracy':
                code.append("        # Compute accuracy")
                code.append("        predictions = output.argmax(dim=1)")
                code.append("        accuracy = (predictions == data_1).float().mean()")
                code.append("        print(f'Accuracy: {accuracy.item():.4f}')")
            elif metric == 'mse':
                code.append("        # Compute MSE")
                code.append("        mse = nn.functional.mse_loss(output, data_1)")
                code.append("        print(f'MSE: {mse.item():.4f}')")
        
        return code