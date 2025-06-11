import os
import shutil
import tempfile
import subprocess
from pathlib import Path

def dfs_topsort(block, visited=None, temp_visited=None, result=None):
    if visited is None:
        visited = set()
        temp_visited = set()
        result = []

    if block in temp_visited:
        print("Warning: Graph contains a cycle!")
        return None

    if block in visited:
        return result

    temp_visited.add(block)
    for neighbor in block.outputs:
        if dfs_topsort(neighbor, visited, temp_visited, result) is None:
            return None

    temp_visited.remove(block)
    visited.add(block)
    result.append(block)
    return result

def run_topsort(blocks):
    if len(blocks) >= 1:
        print("\nRunning topological sort:")
        print("Initial blocks:", [block.label for block in blocks])
        print("Block connections:")
        for block in blocks:
            print(f"  {block.label}:")
            print(f"    inputs: {[b.label for b in block.inputs]}")
            print(f"    outputs: {[b.label for b in block.outputs]}")
        
        visited = set()
        temp_visited = set()
        result = []
        
        for block in blocks:
            if block not in visited:
                component_result = dfs_topsort(block, visited, temp_visited, result)
                if component_result is None:
                    return None
        
        result.reverse()
        print("Topological order:", " -> ".join(block.label for block in result))
        return result
    return []

def generate_model_architecture(model_blocks):
    """Generate the PyTorch model class code."""
    code = []
    # Add model layers
    for block in model_blocks:
        code.append(f"self.{block.label} = {block.model_block.to_source_code()}")
    return code

def generate_forward_pass(model_blocks):
    """Generate the forward pass code for the model."""
    code = []
    var_names = {}
    
    # Assign variable names
    for i, block in enumerate(model_blocks):
        var_names[block] = f"x{i}"
    
    # Generate forward pass code
    for block in model_blocks:
        input_vars = []
        if not block.input_ports:  # If block has no input ports, use 'x'
            input_vars = ["x"]
        else:
            for port in block.input_ports:
                for conn in port.connections:
                    if conn.to_port == port:
                        input_block = conn.from_port.block
                        input_vars.append(var_names[input_block])
                        break
                if not input_vars:  # If no connection found, use 'x'
                    input_vars = ["x"]
        code.append(f"{var_names[block]} = {block.model_block.forward_expr(input_vars)}")
    
    return code

def _export_model_to_file(blocks, filename="model.py"):
    """Export the complete model code to a file."""
    # Run topological sort to ensure no cycles
    result = run_topsort(blocks)
    if not result:
        print("Error: Graph contains cycles")
        return False

    # Filter out non-model blocks
    model_blocks = [block for block in result if hasattr(block, 'model_block') and block.model_block is not None]

    # Generate code sections
    model_code = generate_model_architecture(model_blocks)
    forward_pass = generate_forward_pass(model_blocks)

    # Combine all code sections
    with open(filename, "w") as f:
        # Write imports and class definition
        f.write("import torch\n")
        f.write("import torch.nn as nn\n\n")
        f.write("class Model(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        
        # Write model architecture
        for line in model_code:
            f.write(f"        {line}\n")
        
        # Write forward method
        f.write("\n    def forward(self, x):\n")
        for line in forward_pass:
            f.write(f"        {line}\n")  # Add extra indentation for forward pass lines
        f.write("        return output\n\n")
        
        # Add simple test code
        f.write("if __name__ == '__main__':\n")
        f.write("    model = Model()\n")
        f.write("    x = torch.randn(1, 64)  # Example input\n")
        f.write("    output = model(x)\n")
        f.write("    print(f'Output shape: {output.shape}')\n")

    print(f"[✔] Model exported to: {filename}")
    return True

def save_code(blocks, model_name):
    """Store the model code in the models directory with a specific name."""
    # Create models directory if it doesn't exist
    models_dir = Path("exported_code")
    models_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp to avoid conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = models_dir / f"{model_name}_{timestamp}.py"
    
    # Export the code
    success = _export_model_to_file(blocks, str(filename))
    if success:
        print(f"[✔] Model stored as: {filename}")
    return success

def run_model(blocks):
    """Run the model code in a temporary directory and clean up afterward."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export code to temporary directory
        temp_file = os.path.join(temp_dir, "temp_model.py")
        if not _export_model_to_file(blocks, temp_file):
            return False
        
        try:
            # Run the model code
            result = subprocess.run(
                ["python3", temp_file],
                capture_output=True,
                text=True,
                check=True
            )
            print("\nModel Output:")
            print(result.stdout)
            if result.stderr:
                print("\nWarnings/Errors:")
                print(result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print("\nError running model:")
            print(e.stderr)
            return False

def run_and_save_code(blocks, model_name):
    """Run the model and save its code."""
    # if run_model(blocks):
    #     return save_code(blocks, model_name)
    # return False

    save_code(blocks, model_name)
    run_model(blocks)