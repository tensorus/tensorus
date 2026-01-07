"""
Simple Tensorus Usage - Core Functionality Only

This example demonstrates the fundamental tensor database operations
without any advanced features.
"""
import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorus import Tensorus
from tensorus.tensor_ops import TensorOps


def main():
    """Demonstrate core Tensorus functionality."""
    print("=== Tensorus: Basic Tensor Database ===\n")
    
    # Initialize Tensorus with default settings (no complex features)
    ts = Tensorus()
    print("✓ Initialized Tensorus")
    
    # 1. Create a dataset
    print("\n1. Creating dataset...")
    ts.create_dataset("experiments")
    datasets = ts.list_datasets()
    print(f"✓ Created dataset. Total datasets: {len(datasets)}")
    
    # 2. Store tensors with metadata
    print("\n2. Storing tensors...")
    
    # Store a training batch
    batch_data = torch.randn(32, 128)  # 32 samples, 128 features
    tensor1 = ts.create_tensor(
        batch_data,
        name="train_batch_1",
        dataset="experiments",
        metadata={
            "type": "training_data",
            "epoch": 1,
            "batch_size": 32
        }
    )
    print(f"✓ Stored tensor: {tensor1.name} with shape {tensor1.shape}")
    
    # Store model weights
    weights = torch.randn(128, 10)  # 128 input features, 10 output classes
    tensor2 = ts.create_tensor(
        weights,
        name="model_weights_epoch_1",
        dataset="experiments",
        metadata={
            "type": "model_weights",
            "epoch": 1,
            "layer": "output"
        }
    )
    print(f"✓ Stored tensor: {tensor2.name} with shape {tensor2.shape}")
    
    # 3. List all tensors in dataset
    print("\n3. Listing tensors...")
    tensors = ts.list_tensors("experiments")
    print(f"✓ Dataset contains {len(tensors)} tensors:")
    for t in tensors:
        print(f"  - ID: {t['id']}")
        print(f"    Metadata: {t['metadata']}")
    
    # 4. Perform basic tensor operations
    print("\n4. Performing tensor operations...")
    
    # Create two simple matrices
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    # Matrix multiplication
    result = ts.matmul(a, b)
    print(f"✓ Matrix multiplication:")
    print(f"  A shape: {a.shape}")
    print(f"  B shape: {b.shape}")
    print(f"  Result shape: {result.shape}")
    print(f"  Result:\n{result}")
    
    # Transpose
    transposed = ts.transpose(a)
    print(f"\n✓ Transpose:")
    print(f"  Original shape: {a.shape}")
    print(f"  Transposed shape: {transposed.shape}")
    
    # Mean
    mean_val = TensorOps.mean(a)
    print(f"\n✓ Mean: {mean_val.item():.2f}")
    
    # 5. Summary (skipping tensor retrieval - API being simplified)
    print("\n=== Summary ===")
    print(f"✓ Datasets: {len(ts.list_datasets())}")
    print(f"✓ Tensors in 'experiments': {len(ts.list_tensors('experiments'))}")
    print(f"✓ Operations performed: matmul, transpose, mean")
    print("\nCore tensor database operations completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
