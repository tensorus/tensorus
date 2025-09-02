import os
import sys
import torch

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

def test_core_basic():
    print("Testing basic Tensorus core functionality...")
    
    try:
        # Import the core module
        print("1. Importing tensorus_operational_core...")
        from tensorus.tensorus_operational_core import create_tensorus_core
        print("Successfully imported tensorus_operational_core")
        
        # Create a test directory
        test_dir = "./test_core_data"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            print(f"Created test directory: {test_dir}")
        
        # Initialize the core
        print("\n2. Initializing Tensorus core...")
        core = create_tensorus_core(test_dir)
        print("Core initialized successfully")
        
        # Create a test tensor
        print("\n3. Creating a test tensor...")
        test_tensor = torch.randn(3, 3)
        print("Test tensor created:")
        print(test_tensor)
        
        # Store the tensor
        print("\n4. Storing tensor...")
        metadata = {"test": "data", "category": "test"}
        tensor_id = core.store_tensor(test_tensor, metadata)
        print(f"Tensor stored with ID: {tensor_id}")
        
        # List all tensors
        print("\n5. Listing all tensors...")
        tensor_ids = core.list_tensors()
        print(f"Found {len(tensor_ids)} tensors: {tensor_ids}")
        
        # Get the tensor back
        print("\n6. Retrieving tensor...")
        op_tensor = core.get_tensor(tensor_id)
        print(f"Retrieved tensor with ID: {op_tensor.tensor_id}")
        print("Tensor data:")
        print(op_tensor.data)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"\nCleaned up test directory: {test_dir}")

if __name__ == "__main__":
    test_core_basic()
