import sys
import os

def main():
    print("Python test script started")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print("Test completed successfully")

if __name__ == "__main__":
    main()
