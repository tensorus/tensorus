import os
import sys
import torch

# Add current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_storage():
    print("Testing basic tensor storage...")
    
    try:
        # Import required modules
        print("1. Importing tensor_storage...")
        from tensorus.tensor_storage import TensorStorage
        print("Successfully imported tensor_storage")
        
        # Create a test directory
        test_dir = "./test_storage"
        os.makedirs(test_dir, exist_ok=True)
        print(f"Using test directory: {test_dir}")
        
        # Initialize storage
        print("\n2. Initializing TensorStorage...")
        storage = TensorStorage()
        print("TensorStorage initialized")
        
        # Create a test tensor
        print("\n3. Creating test tensor...")
        test_tensor = torch.randn(3, 3)
        print("Test tensor:")
        print(test_tensor)
        
        # Store the tensor
        print("\n4. Storing tensor...")
        metadata = {"test": "data", "category": "test"}
        tensor_id = storage.store_tensor(test_tensor, metadata)
        print(f"Stored tensor with ID: {tensor_id}")
        
        # List all tensors
        print("\n5. Listing all tensors...")
        tensor_ids = storage.list_tensors()
        print(f"Found {len(tensor_ids)} tensors: {tensor_ids}")
        
        # Get the tensor back
        print("\n6. Retrieving tensor...")
        loaded_tensor, loaded_metadata = storage.get_tensor(tensor_id)
        print("Retrieved tensor:")
        print(loaded_tensor)
        print("With metadata:", loaded_metadata)
        
        # Verify the data matches
        assert torch.allclose(test_tensor, loaded_tensor), "Tensor data mismatch!"
        assert loaded_metadata["test"] == "data", "Metadata mismatch!"
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"\nCleaned up test directory: {test_dir}")

if __name__ == "__main__":
    test_storage()
