#!/usr/bin/env python3
"""
Tensorus Storage-Connected Operations Demo

This script demonstrates the integration between TensorOps and TensorStorage,
showing how to perform operations directly on database-resident tensors
without manual retrieval and storage.

Features demonstrated:
- Direct operations on stored tensors using tensor IDs
- Result caching for performance optimization
- Batch operations on multiple tensors
- Automatic metadata tracking for operations
- Benchmarking and performance analysis
- Different operation types: arithmetic, linear algebra, reductions
"""

import torch
import numpy as np
import sys
from pathlib import Path
import time

# Add tensorus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorus.tensor_storage import TensorStorage
    from tensorus.storage_ops import StorageConnectedTensorOps, TensorStorageWithOps
except ImportError as e:
    print(f"Error importing tensorus modules: {e}")
    sys.exit(1)


def create_demo_dataset(storage, dataset_name: str):
    """Create a dataset with various tensors for demonstration."""
    print(f"\n=== Creating Demo Dataset: {dataset_name} ===")
    
    storage.create_dataset(dataset_name)
    
    # Create various types of tensors
    tensors = {
        "matrix_a": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        "matrix_b": torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
        "vector_x": torch.tensor([1.0, 2.0, 3.0]),
        "vector_y": torch.tensor([4.0, 5.0, 6.0]),
        "large_matrix": torch.randn(100, 100),
        "image_like": torch.randn(3, 224, 224),  # Simulates RGB image
        "weights": torch.randn(512, 256),        # Simulates neural network weights
        "bias": torch.randn(256),                # Simulates bias vector
    }
    
    tensor_ids = {}
    
    for name, tensor in tensors.items():
        metadata = {
            "name": name,
            "type": "matrix" if tensor.ndim == 2 else "vector" if tensor.ndim == 1 else "tensor",
            "created_for": "demo",
            "size": tensor.numel(),
            "shape": list(tensor.shape)
        }
        
        tensor_id = storage.insert(dataset_name, tensor, metadata)
        tensor_ids[name] = tensor_id
        print(f"  Stored {name}: {tensor.shape} -> ID: {tensor_id[:8]}...")
    
    print(f"  Total tensors stored: {len(tensor_ids)}")
    return tensor_ids


def demonstrate_basic_operations(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate basic arithmetic operations on stored tensors."""
    print(f"\n=== Basic Arithmetic Operations ===")
    
    ops = StorageConnectedTensorOps(storage)
    
    # Matrix addition
    print("1. Matrix Addition (A + B):")
    start_time = time.time()
    sum_id = ops.add(dataset_name, tensor_ids["matrix_a"], tensor_ids["matrix_b"])
    operation_time = time.time() - start_time
    
    result = storage.get_tensor_by_id(dataset_name, sum_id)
    print(f"   Result shape: {result['tensor'].shape}")
    print(f"   Computation time: {operation_time:.4f}s")
    print(f"   Result preview: {result['tensor'][:2, :2]}")
    print(f"   Operation metadata: {result['metadata']['operation']}")
    
    # Vector operations
    print("\n2. Vector Addition (x + y):")
    vector_sum_id = ops.add(dataset_name, tensor_ids["vector_x"], tensor_ids["vector_y"])
    vector_result = storage.get_tensor_by_id(dataset_name, vector_sum_id)
    print(f"   Result: {vector_result['tensor']}")
    
    # Scalar operations
    print("\n3. Scalar Multiplication (A * 2.5):")
    scalar_mul_id = ops.multiply(dataset_name, tensor_ids["matrix_a"], 2.5)
    scalar_result = storage.get_tensor_by_id(dataset_name, scalar_mul_id)
    print(f"   Result preview: {scalar_result['tensor'][:2, :2]}")
    
    return {"sum_id": sum_id, "vector_sum_id": vector_sum_id, "scalar_mul_id": scalar_mul_id}


def demonstrate_linear_algebra(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate linear algebra operations."""
    print(f"\n=== Linear Algebra Operations ===")
    
    ops = StorageConnectedTensorOps(storage)
    
    # Matrix multiplication
    print("1. Matrix Multiplication (A @ B):")
    matmul_id = ops.matmul(dataset_name, tensor_ids["matrix_a"], tensor_ids["matrix_b"])
    matmul_result = storage.get_tensor_by_id(dataset_name, matmul_id)
    print(f"   Result shape: {matmul_result['tensor'].shape}")
    print(f"   Result preview: {matmul_result['tensor'][:2, :2]}")
    
    # SVD decomposition
    print("\n2. Singular Value Decomposition of large_matrix:")
    u_id, s_id, vt_id = ops.svd(dataset_name, tensor_ids["large_matrix"])
    
    u_result = storage.get_tensor_by_id(dataset_name, u_id)
    s_result = storage.get_tensor_by_id(dataset_name, s_id)
    vt_result = storage.get_tensor_by_id(dataset_name, vt_id)
    
    print(f"   U shape: {u_result['tensor'].shape}")
    print(f"   S shape: {s_result['tensor'].shape}")  
    print(f"   Vt shape: {vt_result['tensor'].shape}")
    print(f"   Largest singular value: {s_result['tensor'][0]:.4f}")
    
    # Verify reconstruction
    original = storage.get_tensor_by_id(dataset_name, tensor_ids["large_matrix"])["tensor"]
    reconstructed = u_result["tensor"] @ torch.diag(s_result["tensor"]) @ vt_result["tensor"]
    reconstruction_error = torch.norm(original - reconstructed).item()
    print(f"   Reconstruction error: {reconstruction_error:.2e}")


def demonstrate_reductions_and_reshaping(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate reduction and reshaping operations."""
    print(f"\n=== Reduction and Reshaping Operations ===")
    
    ops = StorageConnectedTensorOps(storage)
    
    # Reductions
    print("1. Matrix Sum and Mean:")
    sum_id = ops.sum(dataset_name, tensor_ids["matrix_a"])
    mean_id = ops.mean(dataset_name, tensor_ids["matrix_a"])
    
    sum_result = storage.get_tensor_by_id(dataset_name, sum_id)
    mean_result = storage.get_tensor_by_id(dataset_name, mean_id)
    
    print(f"   Sum: {sum_result['tensor'].item()}")
    print(f"   Mean: {mean_result['tensor'].item()}")
    
    # Dimensional reductions
    print("\n2. Sum along dimensions:")
    sum_dim0_id = ops.sum(dataset_name, tensor_ids["matrix_a"], dim=0)
    sum_dim1_id = ops.sum(dataset_name, tensor_ids["matrix_a"], dim=1)
    
    sum_dim0_result = storage.get_tensor_by_id(dataset_name, sum_dim0_id)
    sum_dim1_result = storage.get_tensor_by_id(dataset_name, sum_dim1_id)
    
    print(f"   Sum along dim=0: {sum_dim0_result['tensor']}")
    print(f"   Sum along dim=1: {sum_dim1_result['tensor']}")
    
    # Reshaping
    print("\n3. Reshaping operations:")
    # Flatten the matrix
    flatten_id = ops.reshape(dataset_name, tensor_ids["matrix_a"], (9,))
    flatten_result = storage.get_tensor_by_id(dataset_name, flatten_id)
    print(f"   Flattened shape: {flatten_result['tensor'].shape}")
    
    # Reshape to different matrix size
    reshape_id = ops.reshape(dataset_name, flatten_id, (1, 9))
    reshape_result = storage.get_tensor_by_id(dataset_name, reshape_id)
    print(f"   Reshaped to: {reshape_result['tensor'].shape}")
    
    # Transpose
    transpose_id = ops.transpose(dataset_name, tensor_ids["matrix_a"], 0, 1)
    transpose_result = storage.get_tensor_by_id(dataset_name, transpose_id)
    print(f"   Transposed: {transpose_result['tensor']}")


def demonstrate_caching(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate caching functionality and performance benefits."""
    print(f"\n=== Caching and Performance ===")
    
    ops = StorageConnectedTensorOps(storage)
    
    # Enable caching
    ops.enable_caching(max_size=50)
    print("Caching enabled with max size 50")
    
    # First computation (no cache)
    print("\n1. First computation (uncached):")
    start_time = time.time()
    result1 = ops.add(dataset_name, tensor_ids["large_matrix"], tensor_ids["large_matrix"], 
                     store_result=False)
    first_time = time.time() - start_time
    print(f"   Time: {first_time:.4f}s")
    print(f"   Cached: {result1.cached}")
    
    # Second computation (cached)
    print("\n2. Second computation (cached):")
    start_time = time.time()
    result2 = ops.add(dataset_name, tensor_ids["large_matrix"], tensor_ids["large_matrix"], 
                     store_result=False)
    second_time = time.time() - start_time
    print(f"   Time: {second_time:.4f}s")
    print(f"   Cached: {result2.cached}")
    print(f"   Speedup: {first_time/second_time:.1f}x")
    
    # Cache statistics
    cache_stats = ops.get_cache_stats()
    print(f"\n3. Cache Statistics:")
    print(f"   Enabled: {cache_stats['enabled']}")
    print(f"   Size: {cache_stats['size']}")
    print(f"   Max Size: {cache_stats['max_size']}")
    
    # Clear cache
    ops.clear_cache()
    print("   Cache cleared")


def demonstrate_batch_operations(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate batch operations on multiple tensors."""
    print(f"\n=== Batch Operations ===")
    
    ops = StorageConnectedTensorOps(storage)
    
    # Create a list of tensor IDs for batch processing
    matrix_ids = [tensor_ids["matrix_a"], tensor_ids["matrix_b"]]
    vector_ids = [tensor_ids["vector_x"], tensor_ids["vector_y"]]
    
    print("1. Batch scalar multiplication:")
    batch_results = ops.batch_operation(
        dataset_name, "multiply", matrix_ids, tensor_id2=0.5, store_result=True
    )
    
    print(f"   Processed {len(batch_results)} tensors")
    for i, result_id in enumerate(batch_results):
        result = storage.get_tensor_by_id(dataset_name, result_id)
        print(f"   Result {i+1} shape: {result['tensor'].shape}")
    
    print("\n2. Batch addition with scalars:")
    vector_batch_results = ops.batch_operation(
        dataset_name, "add", vector_ids, tensor_id2=10.0, store_result=True
    )
    
    for i, result_id in enumerate(vector_batch_results):
        result = storage.get_tensor_by_id(dataset_name, result_id)
        print(f"   Vector {i+1} result: {result['tensor']}")


def demonstrate_direct_storage_integration(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate direct integration with TensorStorage."""
    print(f"\n=== Direct Storage Integration ===")
    
    # Using TensorStorage methods directly
    print("1. Using TensorStorage.tensor_* methods:")
    
    # Direct addition
    direct_add_id = storage.tensor_add(dataset_name, tensor_ids["matrix_a"], tensor_ids["matrix_b"])
    print(f"   Direct addition result ID: {direct_add_id[:8]}...")
    
    # Direct matrix multiplication
    direct_matmul_id = storage.tensor_matmul(dataset_name, tensor_ids["matrix_a"], tensor_ids["matrix_b"])
    print(f"   Direct matmul result ID: {direct_matmul_id[:8]}...")
    
    # Direct sum with custom metadata
    custom_metadata = {"experiment": "demo", "version": "1.0", "notes": "Direct integration test"}
    direct_sum_id = storage.tensor_sum(dataset_name, tensor_ids["large_matrix"], 
                                      result_metadata=custom_metadata)
    
    sum_result = storage.get_tensor_by_id(dataset_name, direct_sum_id)
    print(f"   Sum result: {sum_result['tensor'].item():.2f}")
    print(f"   Custom metadata preserved: {sum_result['metadata']['experiment']}")
    
    # Enable and test caching
    print("\n2. Caching integration:")
    storage.enable_operation_caching(max_size=25)
    cache_stats = storage.get_operation_cache_stats()
    print(f"   Cache enabled: {cache_stats['enabled']}")
    print(f"   Max cache size: {cache_stats['max_size']}")


def demonstrate_benchmarking(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate operation benchmarking."""
    print(f"\n=== Operation Benchmarking ===")
    
    ops = StorageConnectedTensorOps(storage)
    
    # Disable caching for fair benchmarking
    ops.disable_caching()
    
    operations_to_benchmark = [
        ("add", {"tensor_id2": tensor_ids["matrix_b"]}),
        ("multiply", {"tensor_id2": 2.0}),
        ("sum", {}),
        ("transpose", {"dim0": 0, "dim1": 1})
    ]
    
    for op_name, kwargs in operations_to_benchmark:
        print(f"\nBenchmarking {op_name}:")
        benchmark = ops.benchmark_operation(
            dataset_name, op_name, tensor_ids["matrix_a"], 
            iterations=10, **kwargs
        )
        
        print(f"   Mean time: {benchmark['mean_time']*1000:.2f} ms")
        print(f"   Min time:  {benchmark['min_time']*1000:.2f} ms")
        print(f"   Max time:  {benchmark['max_time']*1000:.2f} ms")


def demonstrate_advanced_features(storage, dataset_name: str, tensor_ids: dict):
    """Demonstrate advanced features and use cases."""
    print(f"\n=== Advanced Features ===")
    
    # Using TensorStorageWithOps convenience class
    print("1. TensorStorageWithOps convenience class:")
    storage_with_ops = TensorStorageWithOps(storage)
    
    # Chain of operations
    print("   Chaining operations: (A + B) * 0.5 -> transpose -> sum")
    
    # Step 1: Add
    step1_id = storage_with_ops.tensor_add(dataset_name, tensor_ids["matrix_a"], tensor_ids["matrix_b"])
    
    # Step 2: Multiply by scalar
    step2_id = storage_with_ops.ops.multiply(dataset_name, step1_id, 0.5)
    
    # Step 3: Transpose
    step3_id = storage_with_ops.tensor_reshape(dataset_name, step2_id, (3, 3))
    step3_trans_id = storage_with_ops.ops.transpose(dataset_name, step3_id, 0, 1)
    
    # Step 4: Sum
    final_result_id = storage_with_ops.ops.sum(dataset_name, step3_trans_id)
    
    final_result = storage_with_ops.get_tensor_by_id(dataset_name, final_result_id)
    print(f"   Final result: {final_result['tensor'].item():.2f}")
    
    # Query operations by metadata
    print("\n2. Querying operations by metadata:")
    all_data = storage.get_dataset_with_metadata(dataset_name)
    
    operation_counts = {}
    for record in all_data:
        op = record["metadata"].get("operation", "original_data")
        operation_counts[op] = operation_counts.get(op, 0) + 1
    
    print("   Operations performed:")
    for op, count in sorted(operation_counts.items()):
        print(f"     {op}: {count} times")


def main():
    """Run the complete storage-connected operations demonstration."""
    print("TENSORUS STORAGE-CONNECTED OPERATIONS DEMO")
    print("Addressing GAP 5: Operations Disconnected from Storage")
    print("=" * 60)
    
    # Create temporary storage
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize storage
        print(f"Creating TensorStorage in: {temp_dir}")
        storage = TensorStorage(storage_path=temp_dir)
        
        # Create demo dataset
        dataset_name = "operations_demo"
        tensor_ids = create_demo_dataset(storage, dataset_name)
        
        # Run demonstrations
        demonstrate_basic_operations(storage, dataset_name, tensor_ids)
        demonstrate_linear_algebra(storage, dataset_name, tensor_ids)
        demonstrate_reductions_and_reshaping(storage, dataset_name, tensor_ids)
        demonstrate_caching(storage, dataset_name, tensor_ids)
        demonstrate_batch_operations(storage, dataset_name, tensor_ids)
        demonstrate_direct_storage_integration(storage, dataset_name, tensor_ids)
        demonstrate_benchmarking(storage, dataset_name, tensor_ids)
        demonstrate_advanced_features(storage, dataset_name, tensor_ids)
        
        # Final statistics
        print(f"\n=== Final Statistics ===")
        datasets = storage.list_datasets()
        total_tensors = sum(len(storage.get_dataset(ds)) for ds in datasets)
        print(f"Total datasets: {len(datasets)}")
        print(f"Total tensors stored: {total_tensors}")
        
        # Show some sample operation metadata
        sample_data = storage.get_dataset_with_metadata(dataset_name)
        operations_with_metadata = [
            record for record in sample_data 
            if "operation" in record["metadata"]
        ]
        
        if operations_with_metadata:
            print(f"\nSample operation metadata:")
            sample = operations_with_metadata[0]["metadata"]
            for key, value in sample.items():
                if key.startswith("operation") or key in ["computation_time", "input_tensors"]:
                    print(f"  {key}: {value}")
        
        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✅ Storage-connected operations implemented")
        print("✅ Direct operations on tensor IDs without manual retrieval")
        print("✅ Automatic result caching for performance optimization")
        print("✅ Batch operations for multiple tensors")
        print("✅ Comprehensive metadata tracking for all operations")
        print("✅ Direct TensorStorage integration methods")
        print("✅ Benchmarking and performance analysis")
        print("✅ Advanced operation chaining and querying")
        print("\nGAP 5 has been successfully addressed!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    exit(main())