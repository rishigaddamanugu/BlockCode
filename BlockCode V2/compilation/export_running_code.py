import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from block_types.data_block import DataBlock, RandomTensorBlock
from block_types.data_converter import DataConverter, AutoTokenizerBlock
from block_types.model_block import ModelBlock

block_names = set()

def _dfs_topsort(block, visited=None, temp_visited=None, result=None):
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
        if _dfs_topsort(neighbor, visited, temp_visited, result) is None:
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
        labels = set()
        temp_visited = set()
        result = []
        
        for block in blocks:
            if block not in visited:
                component_result = _dfs_topsort(block, visited, temp_visited, result)
                if component_result is None:
                    return None
            
            if block.label in labels:
                raise AttributeError("All block names must be unique!")
            labels.add(block.label)
        
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
        # for block_input in block.inputs:
        #     if hasattr(block_input, 'data_block') and block_input.data_block is not None:
        #         code.append(f"self.{block_input.label} = {block_input.data_block.to_source_code()}")
        #     if hasattr(block_input, 'data_converter') and block_input.data_converter is not None:
        #         code.append(f"self.{block_input.label} = {block_input.data_converter.to_source_code()}")
        
    
    return code

def generate_forward_pass(model_blocks):
    """Generate the forward pass code for the model."""
    code = []
    var_names = {}
    # Assign variable names to all blocks
    for i, block in enumerate(model_blocks):
        if f"x{i}" in block_names:
            i += 1
        block_names.add(f"x{i}")
        print("BLOCK:", block)
        var_names[block] = f"x{i}"
    
    # Generate forward pass code
    for i, block in enumerate(model_blocks):
        input_vars = []
        for input_block in block.inputs:
            print("INPUT BLOCK:", input_block)
            input_vars.append(var_names[input_block])
        
        # If this is the last block, name its output "output"
        if i == len(model_blocks) - 1:
            if block.model_block:
                print("DEBUG:", block.model_block.forward_expr(input_vars))
                code.append(f"output = {block.model_block.forward_expr(input_vars)}")
        else:
            if block.model_block:
                code.append(f"{var_names[block]} = {block.model_block.forward_expr(input_vars)}")
    
    return code


def generate_run_code(run_blocks):
    ## **TODO**: Generalize this function to not need to type check for data blocks
    """Generate code for running the model using run blocks."""
    code = []
    
    # Generate code for each run block
    for block in run_blocks:
        # Find model and data blocks from inputs
        data_block = None
        for input_block in block.inputs:
            print("TYPE:", type(input_block.model_block))
            if hasattr(input_block, 'model_block') and input_block.model_block and type(input_block.model_block) == RandomTensorBlock:
                data_block = input_block.model_block
            
        
        # Generate code with the found blocks
        if data_block:
            run_code = block.run_block.generate_code(data_block)
            code.extend(run_code)
            code.append("")  # Add blank line between blocks

    return code

def generate_imports(blocks):
    """Generate the imports for the running code."""
    imports = set()
    for block in blocks:
        if block.model_block:
            imports.update(block.model_block.required_imports())
    return list(imports)

def _export_running_code_to_file(blocks, filename="run_model.py"):
    """Export the complete running code to separate files for model architecture and run."""
    # Run topological sort to ensure no cycles
    result = run_topsort(blocks)
    if not result:
        print("Error: Graph contains cycles")
        return False
    for block in result:
        block_names.add(block.label)
    
    # Get the directory from the filename
    code_dir = Path(filename).parent
    code_dir.mkdir(exist_ok=True)

    model_blocks = [block for block in result if block.run_block is None]
    # Separate blocks by type
    run_blocks = [block for block in result if hasattr(block, 'run_block') and block.run_block is not None]

    architecture_code = generate_model_architecture(model_blocks)
    forward_pass = generate_forward_pass(model_blocks)

    # Create model architecture file
    model_filename = code_dir / "model_architecture.py"
    with open(model_filename, "w") as f:
        # Write imports
        # f.write("import torch\n")
        # f.write("import torch.nn as nn\n")
        # f.write("from typing import Dict, Any, List, Tuple\n\n")
        f.write("import torch.nn as nn\n")
        for import_line in generate_imports(blocks):
            f.write(f"{import_line}\n")
        f.write("\n")
        # Write model class
        f.write("class Model(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n")
        
        # Write model architecture and data instantiation
        for line in architecture_code:
            f.write(f"        {line}\n")
        
        # Write forward method
        f.write("\n    def forward(self, x):\n")
        for line in forward_pass:
            f.write(f"        {line}\n")
        f.write("        return output\n")

    # Create run file
    run_filename = code_dir / "run_script.py"
    with open(run_filename, "w") as f:
        # Write imports
        # f.write("import torch\n")
        # f.write("import torch.nn as nn\n")
        # f.write("from typing import Dict, Any, List, Tuple\n\n")
        f.write("import torch\n")
        for import_line in generate_imports(blocks):
            f.write(f"{import_line}\n")
        f.write("from model_architecture import Model\n\n")
        f.write("\n")
        
        if run_blocks:
            run_code = generate_run_code(run_blocks)
            # Write main function
            f.write("def main():\n")
            
            # Write run code
            for line in run_code:
                f.write(f"{line}\n")
            
            # Add main call
            f.write("\nif __name__ == '__main__':\n")
            f.write("    main()\n")

    print(f"[✔] Model architecture exported to: {model_filename}")
    print(f"[✔] run code exported to: {run_filename}")
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
        temp_dir_path = Path(temp_dir)
        if not _export_running_code_to_file(blocks, str(temp_dir_path / "run_model.py")):
            return False
        
        try:
            # Run the code
            result = subprocess.run(
                ["python3", str(temp_dir_path / "run_script.py")],
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
