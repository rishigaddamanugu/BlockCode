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
        # Print block labels
        print("Topological order:", " -> ".join(block.label for block in result))
        # Print model information if available
        model_info = []
        for block in result:
            if hasattr(block, 'model_block') and hasattr(block.model_block, 'to_dict'):
                model_info.append(str(block.model_block.to_dict()))
            else:
                model_info.append("No model info")
        print("Model info:", " -> ".join(model_info))
        return result
    return []

def _export_model_to_file(blocks, filename="model.py"):
    # Run topological sort to ensure no cycles
    result = run_topsort(blocks)
    if not result:
        print("Error: Graph contains cycles")
        return False

    # Build variable names and code sections
    var_names = {}
    code_sections = []
    main_sections = []
    input_var = None

    for block in result:
        if not block.outputs:  # This is an output block
            var_names[block] = "output"
        else:
            var_names[block] = f"x{len(var_names)}"

        # Find input variable(s)
        if not block.inputs:  # This is an input block
            input_var = var_names[block]
            main_sections.append(f"    {var_names[block]} = {block.model_block.forward_expr()}")
        else:
            # Get inputs from connected blocks in the correct order
            input_vars = []
            # For each input port, find the connected block
            for port in block.input_ports:
                for conn in port.connections:
                    if conn.to_port == port:  # This is the input port
                        input_block = conn.from_port.block
                        input_vars.append(var_names[input_block])
                        break
            
            # Pass all inputs to the block's forward_expr
            code_sections.append(f"        {var_names[block]} = {block.model_block.forward_expr(input_vars)}")

    # Write the model to file
    with open(filename, "w") as f:
        f.write("import torch\n")
        f.write("import torch.nn as nn\n\n")
        f.write("class Model(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        for block in result:
            f.write(f"        self.{block.label} = {block.model_block.to_source_code()}\n")
        f.write("\n    def forward(self, x):\n")
        for section in code_sections:
            f.write(f"{section}\n")
        f.write("        return output\n\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    model = Model()\n")
        for section in main_sections:
            f.write(f"{section}\n")
        f.write("    output = model(x0)\n")
        f.write("    print(f'Output shape: {output}')\n")
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
        # Temporary directory is automatically cleaned up when the context manager exits

def run_and_save_code(blocks, model_name):
    """Run the model code and save the code to a file."""
    # # Run the model code
    # if not run_model(blocks):
    #     return False
    
    # # Save the code
    # return save_code(blocks, model_name)

    save_code(blocks, model_name)
    run_model(blocks)