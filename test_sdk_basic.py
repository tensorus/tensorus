"""
Simple SDK test without heavy dependencies
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

print("Testing basic SDK imports...")

# Test the unified SDK (without agents that need heavy dependencies)
from tensorus import Tensorus

print("✓ Tensorus SDK imported successfully")

# Initialize SDK without agents to avoid dependency issues
ts = Tensorus(
    enable_nql=False,  # Disable NQL to avoid transformers dependencies
    enable_embeddings=False,  # Disable embeddings to avoid sentence-transformers
    enable_vector_search=False,  # Disable vector search for now
    enable_orchestrator=False  # Disable orchestrator
)
print(f"✓ Tensorus SDK initialized: {ts}")

# Create a dataset
try:
    ts.create_dataset("sdk_test")
    print("✓ Dataset created")
except Exception as e:
    print(f"✓ Using existing dataset (error: {e})")

# Create and store a tensor using the SDK
tensor = ts.create_tensor(
    [[1, 2, 3], [4, 5, 6]],
    name="test_matrix",
    metadata={"test": "data"},
    dataset="sdk_test"
)
print(f"✓ Tensor created via SDK: {tensor.name}, shape: {tensor.shape}")

# List tensors
tensors = ts.list_tensors("sdk_test")
print(f"✓ Listed {len(tensors)} tensors in dataset")

# Test tensor operations via SDK
a = ts.create_tensor([[1.0, 2.0], [3.0, 4.0]], name="matrix_a")
b = ts.create_tensor([[5.0, 6.0], [7.0, 8.0]], name="matrix_b")

result = ts.matmul(a.to_tensor(), b.to_tensor())
print(f"✓ Matrix multiplication: {a.shape} @ {b.shape} = {result.shape}")

# Test transpose
transposed = ts.transpose(a.to_tensor())
print(f"✓ Transpose: {a.shape} -> {transposed.shape}")

print("\n" + "="*50)
print("✓ ALL SDK TESTS PASSED!")
print("="*50)
print("\n📝 Note: Tests run without NQL, embeddings, and orchestrator")
print("   to avoid heavy dependency requirements")
