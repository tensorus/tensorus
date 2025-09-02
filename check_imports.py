print("Checking imports...")
try:
    import torch
    print("PyTorch version:", torch.__version__)
    
    from tensorus.tensorus_operational_core import create_tensorus_core
    print("Successfully imported tensorus_operational_core")
    
    # Create a simple tensor
    x = torch.randn(2, 2)
    print("Created tensor:")
    print(x)
    
    print("\nAll imports and basic functionality working!")
    
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
