import sys
import os

def main():
    print("=== Python Environment Test ===")
    print(f"Python Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    print("Environment variables:")
    for key in ["PATH", "PYTHONPATH"]:
        print(f"  {key}: {os.environ.get(key, 'Not set')}")

if __name__ == "__main__":
    main()
