import sys
import os
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    print("1. Importing tensorus_operational_core...")
    from tensorus.tensorus_operational_core import create_tensorus_core
    print("Successfully imported tensorus_operational_core")
    
    print("\n2. Creating test tensor...")
    x = torch.randn(2, 2)
    print("Tensor created:")
    print(x)
    
    print("\n3. Creating test directory...")
    test_dir = "./test_data"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory: {test_dir}")
    else:
        print(f"Directory exists: {test_dir}")
    
    print("\n4. Initializing core...")
    core = create_tensorus_core(test_dir)
    print("Core initialized successfully")
    
    print("\n5. Storing tensor...")
    tensor_id = core.store_tensor(x, {"test": "data"})
    print(f"Stored tensor with ID: {tensor_id}")
    
    print("\n6. Retrieving tensor...")
    op_tensor = core.get_tensor(tensor_id)
    print(f"Retrieved tensor with ID: {op_tensor.tensor_id}")
    print("Tensor data:")
    print(op_tensor.data)
    
    print("\n7. Testing complete!")
    
except Exception as e:
    print("\nError:", e)
    import traceback
    traceback.print_exc()
    
finally:
    # Clean up
    import shutil
    if os.path.exists("./test_data"):
        shutil.rmtree("./test_data")
