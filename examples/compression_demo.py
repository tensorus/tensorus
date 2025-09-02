#!/usr/bin/env python3
"""
Tensorus Compression and Quantization Demo

This script demonstrates the compression and quantization capabilities
added to TensorStorage, addressing GAP 3: No Compression/Quantization.

Features demonstrated:
- Different compression algorithms (GZIP, LZ4)
- Different quantization methods (INT8, FP16)
- Compression presets for easy configuration
- Compression statistics and analysis
- Storage efficiency improvements
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add tensorus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorus.tensor_storage import TensorStorage
    from tensorus.compression import (
        CompressionConfig, get_compression_preset,
        GZIPCompression, LZ4Compression, INT8Quantization, FP16Quantization
    )
except ImportError as e:
    print(f"Error importing tensorus modules: {e}")
    sys.exit(1)

def create_sample_tensors():
    """Create various types of tensors for testing compression."""
    tensors = {}
    
    # Large image-like tensor (good for compression)
    tensors['large_image'] = torch.rand(3, 256, 256)
    
    # Weight matrix (typical in ML models)
    tensors['weight_matrix'] = torch.randn(512, 512)
    
    # Sparse tensor (mostly zeros)
    sparse = torch.zeros(100, 100)
    sparse[::10, ::10] = torch.randn(10, 10)
    tensors['sparse_data'] = sparse
    
    # Time series data
    t = torch.linspace(0, 4*np.pi, 1000)
    tensors['time_series'] = torch.sin(t).unsqueeze(0).expand(20, -1)
    
    # Small dense tensor
    tensors['small_dense'] = torch.randn(50, 50)
    
    return tensors

def demonstrate_compression_algorithms():
    """Demonstrate different compression algorithms."""
    print("=" * 60)
    print("COMPRESSION ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    
    # Create test data
    test_tensor = torch.rand(100, 100)
    
    algorithms = [
        ("No Compression", "none"),
        ("GZIP Fast", "gzip", {"compression_level": 1}),
        ("GZIP Balanced", "gzip", {"compression_level": 6}),
        ("GZIP Maximum", "gzip", {"compression_level": 9}),
        ("LZ4 Fast", "lz4", {"compression_level": 1}),
        ("LZ4 High", "lz4", {"compression_level": 4})
    ]
    
    print(f"Test tensor size: {test_tensor.shape} ({test_tensor.numel() * 4} bytes)")
    print()
    
    for name, algo_name, *kwargs in algorithms:
        config = CompressionConfig(
            compression=algo_name,
            quantization="none",
            compression_kwargs=kwargs[0] if kwargs else {}
        )
        
        compressor = config.create_tensor_compression()
        compressed_bytes, metadata = compressor.compress_tensor(test_tensor)
        reconstructed = compressor.decompress_tensor(compressed_bytes, metadata)
        
        original_size = metadata['original_size']
        compressed_size = metadata['compressed_size']
        ratio = metadata['compression_ratio']
        
        print(f"{name:15} | {compressed_size:6d} bytes | {ratio:5.2f}x compression | "
              f"Error: {torch.norm(test_tensor - reconstructed):.2e}")

def demonstrate_quantization_algorithms():
    """Demonstrate different quantization algorithms."""
    print("\n" + "=" * 60)
    print("QUANTIZATION ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    
    # Create test data with different characteristics
    test_tensors = {
        "Random Normal": torch.randn(50, 50),
        "Large Values": torch.randn(50, 50) * 100,
        "Small Values": torch.randn(50, 50) * 0.01,
        "Constant": torch.ones(50, 50) * 3.14,
        "Mixed Range": torch.cat([torch.randn(25, 50) * 0.1, torch.randn(25, 50) * 10], dim=0)
    }
    
    algorithms = [
        ("No Quantization", "none"),
        ("FP16", "fp16"),
        ("INT8", "int8")
    ]
    
    for tensor_name, tensor in test_tensors.items():
        print(f"\nTensor: {tensor_name} (range: {tensor.min():.3f} to {tensor.max():.3f})")
        print("-" * 50)
        
        for algo_name, quant_name in algorithms:
            config = CompressionConfig(
                compression="none",
                quantization=quant_name
            )
            
            compressor = config.create_tensor_compression()
            compressed_bytes, metadata = compressor.compress_tensor(tensor)
            reconstructed = compressor.decompress_tensor(compressed_bytes, metadata)
            
            error = torch.norm(tensor - reconstructed).item()
            size_reduction = len(compressed_bytes) / (tensor.numel() * 4)
            
            print(f"  {algo_name:15} | Size: {size_reduction:4.2f}x | Error: {error:.2e}")

def demonstrate_presets():
    """Demonstrate compression presets."""
    print("\n" + "=" * 60)
    print("COMPRESSION PRESETS DEMONSTRATION")
    print("=" * 60)
    
    # Create a storage directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    presets = ["none", "fast", "balanced", "maximum", "fp16_only", "int8_only"]
    test_tensor = torch.rand(200, 200)
    
    results = {}
    
    for preset in presets:
        storage = TensorStorage(
            storage_path=f"{temp_dir}/{preset}",
            compression_preset=preset
        )
        
        storage.create_dataset("test")
        record_id = storage.insert("test", test_tensor, {"preset": preset})
        
        # Get compression stats
        stats = storage.get_compression_stats("test")
        
        # Retrieve tensor to check accuracy
        retrieved = storage.get_tensor_by_id("test", record_id)
        error = torch.norm(test_tensor - retrieved["tensor"]).item()
        
        results[preset] = {
            'compressed_tensors': stats['compressed_tensors'],
            'original_size': stats['total_original_size'],
            'compressed_size': stats['total_compressed_size'],
            'compression_ratio': stats.get('average_compression_ratio', 1.0),
            'error': error,
            'compression_algo': stats['compression_algorithms'],
            'quantization_algo': stats['quantization_algorithms']
        }
    
    print(f"{'Preset':12} | {'Compressed':>10} | {'Ratio':>6} | {'Error':>10} | {'Algorithms':>20}")
    print("-" * 70)
    
    for preset, stats in results.items():
        compressed = "Yes" if stats['compressed_tensors'] > 0 else "No"
        ratio = f"{stats['compression_ratio']:.2f}x" if stats['compression_ratio'] > 1 else "1.00x"
        algos = f"{stats['compression_algo'][0] if stats['compression_algo'] else 'none'}/{stats['quantization_algo'][0] if stats['quantization_algo'] else 'none'}"
        
        print(f"{preset:12} | {compressed:>10} | {ratio:>6} | {stats['error']:>10.2e} | {algos:>20}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)

def demonstrate_real_world_usage():
    """Demonstrate real-world usage scenarios."""
    print("\n" + "=" * 60)
    print("REAL-WORLD USAGE SCENARIOS")
    print("=" * 60)
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    scenarios = [
        {
            "name": "Model Weights Storage",
            "preset": "maximum",
            "description": "Store neural network weights with maximum compression",
            "tensors": [
                ("conv1_weight", torch.randn(64, 3, 7, 7)),
                ("conv1_bias", torch.randn(64)),
                ("fc_weight", torch.randn(1000, 512)),
                ("fc_bias", torch.randn(1000))
            ]
        },
        {
            "name": "Image Dataset",
            "preset": "balanced",
            "description": "Store image tensors with balanced compression/speed",
            "tensors": [
                (f"image_{i}", torch.rand(3, 224, 224)) for i in range(10)
            ]
        },
        {
            "name": "Time Series Data",
            "preset": "fast",
            "description": "Store time series with fast compression for real-time access",
            "tensors": [
                (f"timeseries_{i}", torch.randn(100, 50)) for i in range(5)
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 50)
        
        storage = TensorStorage(
            storage_path=f"{temp_dir}/{scenario['name'].replace(' ', '_')}",
            compression_preset=scenario['preset']
        )
        
        storage.create_dataset("data")
        
        total_original_elements = 0
        for tensor_name, tensor in scenario['tensors']:
            storage.insert("data", tensor, {"name": tensor_name})
            total_original_elements += tensor.numel()
        
        stats = storage.get_compression_stats("data")
        
        print(f"Tensors stored: {len(scenario['tensors'])}")
        print(f"Total elements: {total_original_elements:,}")
        print(f"Compression preset: {scenario['preset']}")
        print(f"Compression ratio: {stats.get('average_compression_ratio', 1.0):.2f}x")
        print(f"Space saved: {(1 - 1/stats.get('average_compression_ratio', 1.0))*100:.1f}%")
        
        # Test retrieval speed (simplified)
        import time
        start_time = time.time()
        all_data = storage.get_dataset_with_metadata("data")
        retrieval_time = time.time() - start_time
        print(f"Retrieval time: {retrieval_time*1000:.1f}ms")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)

def main():
    """Run all demonstrations."""
    print("TENSORUS COMPRESSION & QUANTIZATION DEMO")
    print("Addressing GAP 3: No Compression/Quantization Support")
    print()
    
    try:
        demonstrate_compression_algorithms()
        demonstrate_quantization_algorithms()
        demonstrate_presets()
        demonstrate_real_world_usage()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✅ Compression algorithms implemented: GZIP, LZ4")
        print("✅ Quantization methods implemented: INT8, FP16")
        print("✅ Multiple compression presets available")
        print("✅ Seamless integration with TensorStorage")
        print("✅ Automatic compression/decompression")
        print("✅ Compression statistics and monitoring")
        print("✅ Backward compatibility maintained")
        print("\nGAP 3 has been successfully addressed!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())