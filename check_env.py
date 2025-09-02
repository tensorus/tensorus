def main():
    print("=== Python Environment Check ===")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    print("\n=== Checking Dependencies ===")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
    
    try:
        from tensorus import __version__
        print(f"✅ Tensorus: {__version__}")
    except ImportError as e:
        print(f"❌ Tensorus not found: {e}")
    
    print("\n=== System Path ===")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")

if __name__ == "__main__":
    import sys
    import os
    main()
