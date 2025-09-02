import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from tensorus.tensor_storage_new import TensorStorage
    print("SUCCESS: Import worked correctly")
    print(f"TensorStorage class: {TensorStorage}")
except Exception as e:
    print(f"ERROR: Import failed with error: {e}")
    import traceback
    traceback.print_exc()
