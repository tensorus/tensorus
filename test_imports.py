print("Testing Tensorus imports...")

# Test basic imports
try:
    import torch
    print("✅ PyTorch import successful")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

# Test tensorus imports
try:
    from tensorus.tensorus_operational_core import create_tensorus_core, TensorusOperationalCore
    print("✅ Tensorus core imports successful")
    
    # Test creating a core instance
    try:
        core = create_tensorus_core("./test_core")
        print("✅ Core creation successful")
    except Exception as e:
        print(f"❌ Core creation failed: {e}")
        
except ImportError as e:
    print(f"❌ Tensorus imports failed: {e}")
    
    # Show more detailed import error
    import traceback
    traceback.print_exc()

print("\nPython path:")
import sys
for path in sys.path:
    print(f" - {path}")
