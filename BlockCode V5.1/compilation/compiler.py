import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from block_types.data_block import DataBlock, RandomTensorBlock
from block_types.data_converter import DataConverter, AutoTokenizerBlock
from block_types.composite_block import CompositeBlock
from collections import deque

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
## This could be recursively structured where we dive into a composite block until reaching a block whose children are "leaves" (have no children)
## Then the block whose children are all leaves will be used to create a new class with its label name
def generate_model_architecture(model_name, composite_blocks):
    """Generate the PyTorch model class code."""
    code = []
    code.append(f"import torch\nimport torch.nn as nn\n\n")
    code.append(f"class {model_name}(nn.Module):\n")
    code.append("    def __init__(self):\n")
    code.append("        super().__init__()\n")
    
    for block in composite_blocks:
        init_line = block.composite_block.to_source_code()
        if init_line:
            code.append(f"        self.{block.label} = {init_line}\n")
    
    return code



def generate_forward_pass(composite_blocks):
    """Generate the forward pass code for the model."""
    code = []
    var_names = {}

    # Write forward method
    code.append("\n    def forward(self, x):\n")

    # Assign variable names to all blocks
    for i, block in enumerate(composite_blocks):
        var_name = f"x{i}"
        while var_name in block_names:  # Ensure uniqueness
            i += 1
            var_name = f"x{i}"
        block_names.add(var_name)
        var_names[block] = var_name

    # Generate forward pass code
    for i, block in enumerate(composite_blocks):
        input_vars = [var_names[input_block] for input_block in block.inputs]
        if block.composite_block:
            line = block.composite_block.forward_expr(input_vars)

            # Last block → assign to "output"
            if i == len(composite_blocks) - 1:
                code.append(f"        output = {line}\n")
            else:
                code.append(f"        {var_names[block]} = {line}\n")

    code.append("        return output\n")
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
            print("TYPE:", type(input_block.composite_block))
            if hasattr(input_block, 'composite_block') and input_block.composite_block and type(input_block.composite_block) == RandomTensorBlock:
                data_block = input_block.composite_block
            
        
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
        if block.composite_block:
            imports.update(block.composite_block.required_imports())
    return list(imports)

## This could be recursively structured where we dive into a composite block until reaching a block whose children are "leaves" (have no children)
## Then the block whose children are all leaves will be used to create a new class with its label name
def generate_models(composite_blocks):
    ## Decide between BFS and DFS here for correct generation (I dont actually think it matters because code generation doesnt have such dependencies)
    ## BFS might make it easier to combine all the returned code with newline characters so we don't need reference passing
    code = []
    q = deque()
    q.extend(composite_blocks)

    while q:
        block = q.popleft()
        architecture_code = generate_model_architecture(block.composite_block.name, block.composite_block.sub_blocks)
        forward_pass = generate_forward_pass(block.composite_block.sub_blocks)
        code.extend(architecture_code)
        code.extend(forward_pass)
        
        q.extend(block.outputs)
    
    return code



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

    composite_blocks = [block for block in result if block.run_block is None]
    # Separate blocks by type
    run_blocks = [block for block in result if hasattr(block, 'run_block') and block.run_block is not None]

    model_code = generate_models(composite_blocks)
    

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
        
        # Write model architecture and forward code
        for line in model_code:
            f.write(f"{line}")


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
