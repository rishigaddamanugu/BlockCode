import torch
import torch.nn as nn
from model_block import ModelBlock
class RunBlock:
    def get_param_info(self):
        return [
            ("epochs", "int", 10),
            ("batch_size", "int", 32),
            ("learning_rate", "float", 0.001),
            ("optimizer", "str", "adam"),
            ("loss_fn", "str", "cross_entropy")
        ]

    def to_source_code(self):
        return None

    def forward_expr(self, inputs):
        return None

class TrainingBlock(RunBlock):
    def get_param_info(self):
        return [
            ("epochs", "int", 10),
            ("batch_size", "int", 32),
            ("learning_rate", "float", 0.001),
            ("optimizer", "str", "adam"),
            ("loss_fn", "str", "cross_entropy")
        ]

    def to_source_code(self):
        return None

    def forward_expr(self, inputs):
        # inputs[0] is the model, inputs[1] is the data block
        params = self.params
        code = f"""
        # Get data from data block
        train_data, train_labels, val_data, val_labels = {inputs[1]}.generate_data()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = {inputs[0]}.to(device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size={params['batch_size']},
            shuffle=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size={params['batch_size']}
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr={params['learning_rate']})
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range({params['epochs']}):
            model.train()
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = loss_fn(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {{epoch+1}}/{params['epochs']}, Loss: {{avg_loss:.4f}}")
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_data)
                    loss = loss_fn(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Validation Loss: {{avg_val_loss:.4f}}, Accuracy: {{accuracy:.2f}}%")
        """
        return code

class InferenceBlock(RunBlock):
    def get_param_info(self):
        return [
            ("batch_size", "int", 32)
        ]

    def to_source_code(self):
        return None

    def forward_expr(self, inputs):
        # inputs[0] is the model, inputs[1] is the data block
        params = self.params
        code = f"""
        # Get data from data block
        train_data, train_labels, val_data, val_labels = {inputs[1]}.generate_data()
        input_data = val_data  # Use validation data for inference
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = {inputs[0]}.to(device)
        model.eval()
        
        with torch.no_grad():
            input_data = input_data.to(device)
            predictions = model(input_data)
            return predictions
        """
        return code

if __name__ == "__main__":
    # Example usage
    from model_block import LinearBlock, ReLUBlock, CompositeBlock
    
    # Create a simple model: Linear -> ReLU -> Linear
    model = CompositeBlock("simple_model", [
        LinearBlock("linear1", {"in_features": 784, "out_features": 128}),
        ReLUBlock("relu1"),
        LinearBlock("linear2", {"in_features": 128, "out_features": 10})
    ])
    
    # Create data block
    data = DataBlock("data", {
        "shape": (784,),
        "num_samples": 1000,
        "num_classes": 10,
        "val_split": 0.2
    })
    
    # Create training configuration
    training = TrainingBlock("training", {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.001
    })
    
    # Create inference configuration
    inference = InferenceBlock("inference", {
        "batch_size": 32
    })
    
    # Generate the code for training and inference
    training_code = training.forward_expr(["model", "data"])
    inference_code = inference.forward_expr(["model", "data"])
    
    print("Training code:")
    print(training_code)
    print("\nInference code:")
    print(inference_code)
