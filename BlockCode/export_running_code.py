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
    """Generate code for running blocks (training/inference/evaluation)."""
    code = []
    
    # Generate run code
    for block in run_blocks:
        code.append(f"    # Run {block.label}")
        # Get the code from the run block
        block_code = block.run_block.generate_code()
        # Replace data_0, data_1 etc. with actual variable names
        for i, (_, var_name) in enumerate(data_vars.items()):
            block_code = [line.replace(f"data_{i}", var_name) for line in block_code]
        code.extend(block_code)
    
    return code

def _export_running_code_to_file(blocks, filename="run_model.py"):
    """Export the complete running code to a file."""
    # Run topological sort to ensure no cycles
    result = run_topsort(blocks)
    if not result:
        print("Error: Graph contains cycles")
        return False

    # Separate blocks by type
    data_blocks = [block for block in result if hasattr(block, 'run_block') and block.run_block is not None]
    run_blocks = [block for block in result if not hasattr(block, 'run_block') and not hasattr(block, 'model_block')]

    # Generate code sections
    data_code, data_vars = generate_data_preparation(data_blocks)
    run_code = generate_run_code(run_blocks, data_vars)

    # Combine all code sections
    with open(filename, "w") as f:
        # Write imports
        f.write("import torch\n")
        f.write("import torch.nn as nn\n")
        f.write("from model import Model\n\n")
        
        # Write main function
        f.write("def main():\n")
        
        # Write data preparation
        for line in data_code:
            f.write(f"{line}\n")
        
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
