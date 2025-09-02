import torch
import os
import shutil
from tensorus.tensorus_operational_core import create_tensorus_core

def test_core():
    # Set up test directory
    test_dir = './test_data'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Initialize core
        print("Initializing core...")
        core = create_tensorus_core(test_dir)
        
        # Create test tensor with metadata
        print("\nStoring test tensor...")
        tensor = torch.randn(3, 3)
        metadata = {"category": "test", "size": "small"}
        tensor_id = core.store_tensor(tensor, metadata)
        print(f"Stored tensor with ID: {tensor_id}")
        
        # List tensors
        print("\nListing all tensors:")
        tensor_ids = core.list_tensors()
        print(f"Found {len(tensor_ids)} tensors: {tensor_ids}")
        
        # Query tensors by category
        print("\nQuerying tensors with category='test':")
        results = core.select({"category": "test"})
        print(f"Found {len(results)} results")
        for result in results:
            print(f"Result tensor ID: {result.tensor_id}")
            print(f"Metadata: {result.metadata}")
            print(f"Data: {result.data}")
            
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_core()
