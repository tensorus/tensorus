import os
import sys
import tempfile
import torch
from tensorus.tensorus_operational_core import create_tensorus_core

def test_query():
    # Set up test directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize core
        print("Initializing core...")
        core = create_tensorus_core(temp_dir)
        
        # Create test data
        print("\nCreating test tensors...")
        tensor_ids = []
        for i in range(3):
            tensor = torch.randn(2, 2)
            metadata = {
                "category": "test",
                "size": "small",
                "batch": i % 2
            }
            tensor_id = core.store_tensor(tensor, metadata)
            tensor_ids.append(tensor_id)
            print(f"Stored tensor {i+1}: {tensor_id}")
        
        # List all tensors
        print("\nListing all tensors:")
        all_tensors = core.list_tensors()
        print(f"Found {len(all_tensors)} tensors: {all_tensors}")
        
        # Query tensors by category
        print("\nQuerying tensors with category='test':")
        results = core.select({"category": "test"})
        print(f"Query returned {len(results)} results")
        
        # Print results
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Tensor ID: {result.tensor_id}")
            print(f"Metadata: {result.metadata}")
            print(f"Data: \n{result.data}")
            
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up {temp_dir}")

if __name__ == "__main__":
    test_query()
