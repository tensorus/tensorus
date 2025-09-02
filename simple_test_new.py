import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import torch
    from tensorus.tensor_storage import TensorStorage
    print("SUCCESS: Import worked correctly")
    
    # Create a simple test
    storage = TensorStorage()
    storage.create_dataset("test_dataset")
    tensor = torch.rand(3, 10)
    record_id = storage.insert("test_dataset", tensor, {"source": "test"})
    print(f"SUCCESS: Inserted tensor with record_id: {record_id}")
    
    # Retrieve the tensor
    retrieved = storage.get_tensor_by_id("test_dataset", record_id)
    print(f"SUCCESS: Retrieved tensor with shape: {retrieved['tensor'].shape}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"ERROR: Test failed with error: {e}")
    import traceback
    traceback.print_exc()
