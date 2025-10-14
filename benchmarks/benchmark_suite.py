"""
Tensorus Comprehensive Benchmarking Suite

Provides real performance measurements across all core components.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorus import Tensorus
from tensorus.tensor_storage import TensorStorage
from tensorus.tensor_ops import TensorOps


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    duration_ms: float
    operations_per_second: float
    memory_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TensorusBenchmark:
    """Main benchmarking class."""
    
    def __init__(self, output_file: str = "benchmark_results.json"):
        self.results: List[BenchmarkResult] = []
        self.output_file = output_file
        
    def measure_time(self, func: Callable, iterations: int = 1) -> float:
        """Measure execution time in milliseconds."""
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        end = time.perf_counter()
        return (end - start) * 1000 / iterations
    
    def estimate_memory_mb(self, tensor: torch.Tensor) -> float:
        """Estimate tensor memory usage in MB."""
        return tensor.element_size() * tensor.numel() / (1024 ** 2)
    
    def run_benchmark(self, name: str, func: Callable, iterations: int = 1,
                     metadata: Dict[str, Any] = None) -> BenchmarkResult:
        """Run a benchmark and record results."""
        print(f"Running: {name}...", end=" ", flush=True)
        
        duration_ms = self.measure_time(func, iterations)
        ops_per_sec = (iterations / duration_ms) * 1000 if duration_ms > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            duration_ms=duration_ms,
            operations_per_second=ops_per_sec,
            memory_mb=0.0,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        print(f"✓ {duration_ms:.2f}ms ({ops_per_sec:.0f} ops/sec)")
        return result
    
    def save_results(self):
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__
            },
            "results": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "ops_per_sec": r.operations_per_second,
                    "memory_mb": r.memory_mb,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to {self.output_file}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("TENSORUS BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Benchmark':<40} {'Duration':<12} {'Ops/Sec':<12}")
        print("-"*70)
        
        for result in self.results:
            print(f"{result.name:<40} {result.duration_ms:>8.2f}ms {result.operations_per_second:>10.0f}")
        
        print("="*70)


def benchmark_tensor_storage(bench: TensorusBenchmark):
    """Benchmark tensor storage operations."""
    print("\n### Tensor Storage Benchmarks ###")
    
    ts = Tensorus()
    
    # Tensor insertion
    def insert_tensors():
        for i in range(100):
            tensor = np.random.rand(10, 10)
            ts.create_tensor(tensor, name=f"tensor_{i}")
    
    bench.run_benchmark(
        "Storage: Insert 100 tensors (10x10)",
        insert_tensors,
        iterations=1,
        metadata={"tensor_shape": [10, 10], "count": 100}
    )
    
    # Tensor retrieval
    tensor_ids = [t["id"] for t in ts.list_tensors()]
    
    def retrieve_tensors():
        for tid in tensor_ids[:50]:
            ts.get_tensor(tid)
    
    bench.run_benchmark(
        "Storage: Retrieve 50 tensors",
        retrieve_tensors,
        iterations=10,
        metadata={"count": 50}
    )
    
    # Metadata search
    bench.run_benchmark(
        "Storage: Metadata search (tags)",
        lambda: ts.search_metadata({"name": "tensor_10"}),
        iterations=100,
        metadata={"filter_type": "exact_match"}
    )


def benchmark_tensor_operations(bench: TensorusBenchmark):
    """Benchmark tensor mathematical operations."""
    print("\n### Tensor Operations Benchmarks ###")
    
    # Matrix multiplication - various sizes
    sizes = [
        (64, 64),
        (256, 256),
        (1024, 1024),
    ]
    
    for size in sizes:
        a = torch.randn(size)
        b = torch.randn(size)
        
        bench.run_benchmark(
            f"TensorOps: Matrix multiply {size[0]}x{size[1]}",
            lambda: TensorOps.matmul(a, b),
            iterations=10,
            metadata={"shape": size, "device": str(a.device)}
        )
    
    # SVD decomposition
    matrix = torch.randn(100, 100)
    bench.run_benchmark(
        "TensorOps: SVD decomposition (100x100)",
        lambda: TensorOps.svd(matrix),
        iterations=5,
        metadata={"shape": [100, 100]}
    )
    
    # Convolution
    input_tensor = torch.randn(1, 3, 32, 32)
    kernel = torch.randn(3, 3)
    
    bench.run_benchmark(
        "TensorOps: 2D Convolution (32x32x3)",
        lambda: TensorOps.convolve_2d(input_tensor[0, 0], kernel),
        iterations=10,
        metadata={"input_shape": [32, 32], "kernel_shape": [3, 3]}
    )


def benchmark_vector_database(bench: TensorusBenchmark):
    """Benchmark vector similarity search."""
    print("\n### Vector Database Benchmarks ###")
    
    ts = Tensorus(enable_vector_search=True, enable_embeddings=True)
    
    # Index creation
    dimensions = 384
    bench.run_benchmark(
        f"VectorDB: Create index ({dimensions}D)",
        lambda: ts.create_index("bench_index", dimensions=dimensions),
        iterations=1,
        metadata={"dimensions": dimensions}
    )
    
    # Vector insertion - batched
    num_vectors = 1000
    embeddings = np.random.rand(num_vectors, dimensions).astype(np.float32)
    vector_ids = [f"vec_{i}" for i in range(num_vectors)]
    
    bench.run_benchmark(
        f"VectorDB: Add {num_vectors} vectors (batched)",
        lambda: ts.add_vectors("bench_index", vector_ids, embeddings),
        iterations=1,
        metadata={"count": num_vectors, "dimensions": dimensions}
    )
    
    # Vector search - various k values
    query = np.random.rand(dimensions).astype(np.float32)
    
    for k in [1, 10, 50]:
        bench.run_benchmark(
            f"VectorDB: Search (k={k})",
            lambda: ts.search_vectors("bench_index", query, k=k),
            iterations=100,
            metadata={"k": k, "index_size": num_vectors}
        )


def benchmark_embedding_generation(bench: TensorusBenchmark):
    """Benchmark embedding generation."""
    print("\n### Embedding Generation Benchmarks ###")
    
    try:
        ts = Tensorus(enable_embeddings=True, embedding_model="all-MiniLM-L6-v2")
        
        # Single text embedding
        text = "This is a test sentence for embedding generation."
        bench.run_benchmark(
            "Embeddings: Generate single embedding",
            lambda: ts.generate_embeddings(text),
            iterations=10,
            metadata={"text_length": len(text)}
        )
        
        # Batch embedding generation
        texts = ["Test sentence " + str(i) for i in range(32)]
        bench.run_benchmark(
            "Embeddings: Generate batch (32 texts)",
            lambda: ts.generate_embeddings(texts),
            iterations=5,
            metadata={"batch_size": len(texts)}
        )
        
    except Exception as e:
        print(f"⚠ Embedding benchmarks skipped: {e}")


def benchmark_nql_queries(bench: TensorusBenchmark):
    """Benchmark Natural Query Language processing."""
    print("\n### NQL Query Benchmarks ###")
    
    ts = Tensorus(enable_nql=True)
    
    # Create test dataset
    ts.create_dataset("test_nql")
    for i in range(100):
        tensor = np.random.rand(5, 5)
        ts.create_tensor(
            tensor,
            dataset="test_nql",
            metadata={"id": i, "value": np.random.rand()}
        )
    
    # Simple queries
    queries = [
        "get all from test_nql",
        "count test_nql",
        "list datasets"
    ]
    
    for query in queries:
        try:
            bench.run_benchmark(
                f"NQL: '{query}'",
                lambda q=query: ts.query(q),
                iterations=10,
                metadata={"query": query}
            )
        except Exception as e:
            print(f"⚠ Query failed: {e}")


def benchmark_compression(bench: TensorusBenchmark):
    """Benchmark compression performance."""
    print("\n### Compression Benchmarks ###")
    
    from tensorus.compression import CompressionManager
    
    # Large tensor for compression
    large_tensor = torch.randn(1000, 1000)
    tensor_bytes = large_tensor.numpy().tobytes()
    
    compressor = CompressionManager()
    
    # LZ4 compression
    bench.run_benchmark(
        "Compression: LZ4 compress (1000x1000 tensor)",
        lambda: compressor.compress(tensor_bytes, algorithm="lz4"),
        iterations=10,
        metadata={"algorithm": "lz4", "size_mb": large_tensor.numel() * 4 / (1024**2)}
    )
    
    # GZIP compression
    bench.run_benchmark(
        "Compression: GZIP compress (1000x1000 tensor)",
        lambda: compressor.compress(tensor_bytes, algorithm="gzip"),
        iterations=5,
        metadata={"algorithm": "gzip", "size_mb": large_tensor.numel() * 4 / (1024**2)}
    )


def benchmark_end_to_end_workflow(bench: TensorusBenchmark):
    """Benchmark complete end-to-end workflows."""
    print("\n### End-to-End Workflow Benchmarks ###")
    
    ts = Tensorus(enable_vector_search=True, enable_embeddings=True)
    
    # Complete ML experiment workflow
    def ml_experiment_workflow():
        # Create dataset
        ts.create_dataset("ml_exp")
        
        # Generate and store experiment results
        for i in range(10):
            weights = np.random.rand(10, 10)
            ts.create_tensor(
                weights,
                dataset="ml_exp",
                name=f"model_{i}",
                metadata={"accuracy": np.random.rand(), "epoch": i}
            )
        
        # Query best results
        results = ts.search_metadata({"accuracy": 0.5})
    
    bench.run_benchmark(
        "E2E: ML experiment tracking (10 models)",
        ml_experiment_workflow,
        iterations=5,
        metadata={"workflow": "ml_experiment", "models": 10}
    )
    
    # Semantic search workflow
    def semantic_search_workflow():
        docs = ["Document " + str(i) for i in range(20)]
        ts.embed_and_index(docs, "docs", ids=[f"doc_{i}" for i in range(20)])
        query = "Document 5"
        ts.search_vectors("docs", ts.generate_embeddings(query)[0], k=5)
    
    try:
        bench.run_benchmark(
            "E2E: Semantic search (20 docs)",
            semantic_search_workflow,
            iterations=3,
            metadata={"workflow": "semantic_search", "docs": 20}
        )
    except Exception as e:
        print(f"⚠ Workflow skipped: {e}")


def benchmark_comparison_with_traditional(bench: TensorusBenchmark):
    """Compare Tensorus with traditional file-based storage."""
    print("\n### Traditional Storage Comparison ###")
    
    import tempfile
    import pickle
    import os
    
    # Tensorus storage
    ts = Tensorus()
    tensors = [np.random.rand(100, 100) for _ in range(100)]
    
    def tensorus_store():
        for i, tensor in enumerate(tensors):
            ts.create_tensor(tensor, name=f"comp_tensor_{i}")
    
    bench.run_benchmark(
        "Comparison: Tensorus store 100 tensors",
        tensorus_store,
        iterations=1,
        metadata={"method": "tensorus", "count": 100}
    )
    
    # Traditional pickle storage
    temp_dir = tempfile.mkdtemp()
    
    def traditional_store():
        for i, tensor in enumerate(tensors):
            filepath = os.path.join(temp_dir, f"tensor_{i}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(tensor, f)
    
    bench.run_benchmark(
        "Comparison: Pickle store 100 tensors",
        traditional_store,
        iterations=1,
        metadata={"method": "pickle", "count": 100}
    )
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all benchmarks."""
    print("="*70)
    print("TENSORUS COMPREHENSIVE BENCHMARK SUITE")
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    bench = TensorusBenchmark()
    
    try:
        benchmark_tensor_storage(bench)
        benchmark_tensor_operations(bench)
        benchmark_vector_database(bench)
        benchmark_embedding_generation(bench)
        benchmark_nql_queries(bench)
        benchmark_compression(bench)
        benchmark_end_to_end_workflow(bench)
        benchmark_comparison_with_traditional(bench)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmarks interrupted by user")
    except Exception as e:
        print(f"\n\n⚠ Benchmark error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        bench.print_summary()
        bench.save_results()


if __name__ == "__main__":
    main()
