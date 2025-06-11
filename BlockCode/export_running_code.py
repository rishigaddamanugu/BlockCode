import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from data_block import DataBlock

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

def generate_model_architecture(model_blocks, data_blocks):
    """Generate the PyTorch model class code."""
    code = []
    # Add model layers
    for block in model_blocks:
        code.append(f"self.{block.label} = {block.model_block.to_source_code()}")
    
    # Add data instantiation
    for block in data_blocks:
        code.append(f"self.{block.label} = {block.data_block.generate_data()}")
    
    return code

def generate_forward_pass(model_blocks, data_blocks):
    """Generate the forward pass code for the model."""
    code = []
    var_names = {}
    
    # First, collect all blocks that are connected to model blocks
    all_blocks = set(model_blocks)
    for block in model_blocks:
        for port in block.input_ports:
            for conn in port.connections:
                if conn.to_port == port:
                    all_blocks.add(conn.from_port.block)
    
    # Assign variable names to all blocks
    for i, block in enumerate(all_blocks):
        var_names[block] = f"x{i}"
    
    # Generate forward pass code
    for i, block in enumerate(model_blocks):
        input_vars = []
        if not block.input_ports:  # If block has no input ports, use 'x'
            input_vars = ["x"]
        else:
            # For each input port, find its connected input
            for port in block.input_ports:
                port_input = None
                for conn in port.connections:
                    if conn.to_port == port:
                        input_block = conn.from_port.block
                        # If the input is a data block, use self.data_block_name
                        if input_block in data_blocks:
                            port_input = f"self.{input_block.label}"
                        else:
                            port_input = var_names[input_block]
                        break
                # If no connection found for this port, use 'x'
                input_vars.append(port_input if port_input else "x")
        
        # If this is the last block, name its output "output"
        if i == len(model_blocks) - 1:
            code.append(f"output = {block.model_block.forward_expr(input_vars)}")
        else:
            code.append(f"{var_names[block]} = {block.model_block.forward_expr(input_vars)}")
    
    return code

def generate_data_preparation(data_blocks):
    """Generate code for data preparation using data blocks."""
    code = []
    var_names = {}
    
    # Assign variable names to data blocks
    for i, block in enumerate(data_blocks):
        var_names[block] = f"data_{i}"
    
    # Generate data preparation code
    for block in data_blocks:
        code.append(f"    # Generate data using {block.label}")
        code.append(f"    {var_names[block]} = {block.run_block.generate_data()}")
    
    return code, var_names

def generate_run_code(run_blocks, data_vars):
    """Generate code for running the model using run blocks."""
    code = []
    
    # Generate code for each run block
    for block in run_blocks:
        if hasattr(block, 'run_block'):
            # Find model and data blocks from inputs
            model_block = None
            data_block = None
            for input_block in block.inputs:
                if hasattr(input_block, 'model_block') and input_block.model_block:
                    model_block = input_block.model_block
                elif hasattr(input_block, 'data_block') and input_block.data_block:
                    data_block = input_block.data_block
            
            # Generate code with the found blocks
            run_code = block.run_block.generate_code(model_block, data_block)
            code.extend(run_code)
            code.append("")  # Add blank line between blocks
    
    return code

def _export_running_code_to_file(blocks, filename="run_model.py"):
    """Export the complete running code to a file."""
    # Run topological sort to ensure no cycles
    result = run_topsort(blocks)
    if not result:
        print("Error: Graph contains cycles")
        return False

    # Separate blocks by type
    model_blocks = [block for block in result if hasattr(block, 'model_block') and block.model_block is not None]
    data_blocks = [block for block in result if hasattr(block, 'data_block') and block.data_block is not None]
    run_blocks = [block for block in result if hasattr(block, 'run_block') and block.run_block is not None]

    # Generate code sections
    model_code = generate_model_architecture(model_blocks, data_blocks)
    forward_pass = generate_forward_pass(model_blocks, data_blocks)
    run_code = generate_run_code(run_blocks, {})

    # Combine all code sections
    with open(filename, "w") as f:
        # Write imports
        f.write("import torch\n")
        f.write("import torch.nn as nn\n")
        f.write("from typing import Dict, Any, List, Tuple\n\n")
        
        # Write model class
        f.write("class Model(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        
        # Write model architecture and data instantiation
        for line in model_code:
            f.write(f"        {line}\n")
        
        # Write forward method
        f.write("\n    def forward(self, x):\n")
        for line in forward_pass:
            f.write(f"        {line}\n")
        f.write("        return output\n\n")
        
        # Write main function
        f.write("def main():\n")
        
        # Write run code
        for line in run_code:
            f.write(f"{line}\n")
        
        # Add main call
        f.write("\nif __name__ == '__main__':\n")
        f.write("    main()\n")

    print(f"[✔] Running code exported to: {filename}")
    return True

def save_running_code(blocks, code_name):
    """Store the running code in the exported_code directory with a specific name."""
    # Create directory if it doesn't exist
    code_dir = Path("exported_code")
    code_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp to avoid conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = code_dir / f"{code_name}_{timestamp}.py"
    
    # Export the code
    success = _export_running_code_to_file(blocks, str(filename))
    if success:
        print(f"[✔] Running code stored as: {filename}")
    return success

def run_running_code(blocks):
    """Run the running code in a temporary directory and clean up afterward."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export code to temporary directory
        temp_file = os.path.join(temp_dir, "temp_run.py")
        if not _export_running_code_to_file(blocks, temp_file):
            return False
        
        try:
            # Run the code
            result = subprocess.run(
                ["python3", temp_file],
                capture_output=True,
                text=True,
                check=True
            )
            print("\nRunning Code Output:")
            print(result.stdout)
            if result.stderr:
                print("\nWarnings/Errors:")
                print(result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print("\nError running code:")
            print(e.stderr)
            return False

def run_and_save_running_code(blocks, code_name):
    """Run the code and save it."""
    save_running_code(blocks, code_name)
    run_running_code(blocks)
