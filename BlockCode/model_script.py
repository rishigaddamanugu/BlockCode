import torch
import torch.nn as nn
from model_block import ModelBlock
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

    def generate_code(self):
        """Generate the code for this block."""
        raise NotImplementedError("Subclasses must implement generate_code")

class InferenceBlock(RunBlock):
    def get_mandatory_params(self):
        return ["batch_size"]

    def get_param_info(self):
        return [
            ("batch_size", "int", 1),
            ("device", "str", "cpu"),
            ("model_path", "str", None),  # For loading saved models
        ]

    def generate_code(self):
        params = self.params
        code = []
        code.append("    # Setup model and device")
        code.append(f"    device = torch.device('{params['device']}')")
        code.append("    model = Model()")
        
        if params.get('model_path'):
            code.append(f"    model.load_state_dict(torch.load('{params['model_path']}'))")
        
        code.append("    model.to(device)")
        code.append("    model.eval()")
        code.append("")
        code.append("    # Run inference")
        code.append("    with torch.no_grad():")
        code.append(f"        batch_size = {params['batch_size']}")
        code.append("        # Assuming data_0 is the input tensor")
        code.append("        output = model(data_0)")
        code.append("    print(f'Inference output shape: {output.shape}')")
        return code

class TrainingBlock(RunBlock):
    def get_mandatory_params(self):
        return ["epochs", "batch_size", "learning_rate"]

    def get_param_info(self):
        return [
            ("epochs", "int", 10),
            ("batch_size", "int", 32),
            ("learning_rate", "float", 0.001),
            ("optimizer", "str", "adam"),
            ("device", "str", "cpu"),
            ("save_path", "str", None),  # For saving trained models
        ]

    def generate_code(self):
        params = self.params
        code = []
        code.append("    # Setup model, device, and optimizer")
        code.append(f"    device = torch.device('{params['device']}')")
        code.append("    model = Model()")
        code.append("    model.to(device)")
        code.append("    model.train()")
        
        # Setup optimizer
        optimizer_name = params.get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            code.append(f"    optimizer = torch.optim.Adam(model.parameters(), lr={params['learning_rate']})")
        elif optimizer_name == 'sgd':
            code.append(f"    optimizer = torch.optim.SGD(model.parameters(), lr={params['learning_rate']})")
        
        code.append("")
        code.append("    # Training loop")
        code.append(f"    epochs = {params['epochs']}")
        code.append(f"    batch_size = {params['batch_size']}")
        code.append("    for epoch in range(epochs):")
        code.append("        # Assuming data_0 is the input tensor and data_1 is the target")
        code.append("        optimizer.zero_grad()")
        code.append("        output = model(data_0)")
        code.append("        loss = nn.functional.mse_loss(output, data_1)  # Example loss")
        code.append("        loss.backward()")
        code.append("        optimizer.step()")
        code.append("        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')")
        
        if params.get('save_path'):
            code.append("")
            code.append("    # Save the trained model")
            code.append(f"    torch.save(model.state_dict(), '{params['save_path']}')")
        
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

    def generate_code(self):
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
        code.append("        # Assuming data_0 is the input tensor and data_1 is the target")
        code.append("        output = model(data_0)")
        
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

# if __name__ == "__main__":
#     # Example usage
#     from model_block import LinearBlock, ReLUBlock, CompositeBlock
    
#     # Create a simple model: Linear -> ReLU -> Linear
#     model = CompositeBlock("simple_model", [
#         LinearBlock("linear1", {"in_features": 784, "out_features": 128}),
#         ReLUBlock("relu1"),
#         LinearBlock("linear2", {"in_features": 128, "out_features": 10})
#     ])
    
#     # Create data block
#     data = DataBlock("data", {
#         "shape": (784,),
#         "num_samples": 1000,
#         "num_classes": 10,
#         "val_split": 0.2
#     })
    
#     # Create training configuration
#     training = TrainingBlock("training", {
#         "epochs": 5,
#         "batch_size": 16,
#         "learning_rate": 0.001
#     })
    
#     # Create inference configuration
#     inference = InferenceBlock("inference", {
#         "batch_size": 32
#     })
    
#     # Generate the code for training and inference
#     training_code = training.generate_code()
#     inference_code = inference.generate_code()
    
#     print("Training code:")
#     print(training_code)
#     print("\nInference code:")
#     print(inference_code)
