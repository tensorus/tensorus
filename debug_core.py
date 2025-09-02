import torch
import os
import shutil
import sys
import traceback
from tensorus.tensorus_operational_core import create_tensorus_core

def debug_core():
    # Set up test directory
    test_dir = './test_data'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        print("1. Initializing core...")
        core = create_tensorus_core(test_dir)
        print("Core initialized successfully")
        
        print("\n2. Creating test tensor...")
        tensor = torch.randn(3, 3)
        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        print("\n3. Storing tensor with metadata...")
        metadata = {"category": "test", "size": "small", "batch": 1}
        print(f"Metadata: {metadata}")
        tensor_id = core.store_tensor(tensor, metadata)
        print(f"Stored tensor with ID: {tensor_id}")
        
        print("\n4. Listing all tensors...")
        tensor_ids = core.list_tensors()
        print(f"Found {len(tensor_ids)} tensors: {tensor_ids}")
        
        print("\n5. Getting tensor metadata...")
        try:
            op_tensor = core.get_tensor(tensor_id)
            print(f"Got operational tensor for ID: {tensor_id}")
            print(f"Tensor metadata: {op_tensor.metadata}")
        except Exception as e:
            print(f"Error getting tensor: {e}")
            traceback.print_exc()
        
        print("\n6. Querying tensors by category='test'...")
        try:
            results = core.select({"category": "test"})
            print(f"Query returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"Result {i+1}: ID={result.tensor_id}, shape={result.data.shape}")
        except Exception as e:
            print(f"Error querying tensors: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    debug_core()
