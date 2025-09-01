"""
Comprehensive tests for Tensor Chunking and Compression System

Tests cover:
- Basic chunking and reconstruction
- Compression algorithms
- Large tensor handling
- Streaming operations
- Memory monitoring
- Edge cases and error handling
- Performance benchmarks
"""

import os
import time
import tempfile
import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
import psutil

# Import chunking modules
from tensorus.tensor_chunking import (
    TensorChunker, TensorChunkingConfig, ChunkMetadata,
    CompressionAlgorithm, TensorChunkingError,
    ChunkNotFoundError, CompressionError
)
from tensorus.tensor_chunking_enhanced import (
    ChunkMetadataTracker, StreamingTensorProcessor,
    MemoryMonitor, EnhancedTensorChunker
)


class TestTensorChunking(unittest.TestCase):
    """Basic tensor chunking tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorChunkingConfig(
            max_tensor_size_mb=1,  # Small threshold for testing
            chunk_size_mb=0.5,
            default_compression=CompressionAlgorithm.NONE
        )
        self.chunker = TensorChunker(self.config)

    def test_small_tensor_no_chunking(self):
        """Test that small tensors are not chunked."""
        small_tensor = torch.randn(10, 10)
        self.assertFalse(self.chunker.should_chunk_tensor(small_tensor))

        # Should still work with single chunk
        chunks = self.chunker.chunk_tensor(small_tensor, "test_small")
        self.assertEqual(len(chunks), 1)

        # Reconstruction should work
        reconstructed = self.chunker.reconstruct_tensor(chunks)
        self.assertTrue(torch.allclose(small_tensor, reconstructed))

    def test_large_tensor_chunking(self):
        """Test chunking of large tensors."""
        # Create a tensor that exceeds the chunking threshold
        large_tensor = torch.randn(1000, 1000)  # ~4MB tensor
        self.assertTrue(self.chunker.should_chunk_tensor(large_tensor))

        # Test chunking strategy
        strategy = self.chunker.calculate_chunk_strategy((1000, 1000), torch.float32)
        self.assertTrue(strategy["should_chunk"])
        self.assertGreater(strategy["total_chunks"], 1)

        # Test actual chunking
        chunks = self.chunker.chunk_tensor(large_tensor, "test_large")
        self.assertGreater(len(chunks), 1)

        # Test reconstruction
        reconstructed = self.chunker.reconstruct_tensor(chunks)
        self.assertTrue(torch.allclose(large_tensor, reconstructed))

    def test_chunking_strategy_calculation(self):
        """Test chunking strategy calculation."""
        # Test with 1D tensor
        shape_1d = (10000,)
        strategy_1d = self.chunker.calculate_chunk_strategy(shape_1d, torch.float32)
        self.assertFalse(strategy_1d["should_chunk"])

        # Test with 2D tensor
        shape_2d = (1000, 1000)
        strategy_2d = self.chunker.calculate_chunk_strategy(shape_2d, torch.float32)
        self.assertTrue(strategy_2d["should_chunk"])

        # Test with small tensor
        small_shape = (10, 10)
        strategy_small = self.chunker.calculate_chunk_strategy(small_shape, torch.float32)
        self.assertFalse(strategy_small["should_chunk"])

    def test_different_dtypes(self):
        """Test chunking with different tensor dtypes."""
        # Test with floating point dtypes
        float_dtypes = [torch.float32, torch.float64]
        for dtype in float_dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.randn(100, 100, dtype=dtype)
                if self.chunker.should_chunk_tensor(tensor):
                    chunks = self.chunker.chunk_tensor(tensor, f"test_{dtype}")
                    reconstructed = self.chunker.reconstruct_tensor(chunks)
                    self.assertTrue(torch.allclose(tensor, reconstructed))

        # Test with integer dtypes
        int_dtypes = [torch.int32, torch.int64]
        for dtype in int_dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.randint(0, 100, (100, 100), dtype=dtype)
                if self.chunker.should_chunk_tensor(tensor):
                    chunks = self.chunker.chunk_tensor(tensor, f"test_{dtype}")
                    reconstructed = self.chunker.reconstruct_tensor(chunks)
                    self.assertTrue(torch.allclose(tensor, reconstructed))


class TestCompressionAlgorithms(unittest.TestCase):
    """Test compression algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorChunkingConfig(
            max_tensor_size_mb=1,
            default_compression=CompressionAlgorithm.LZ4
        )
        self.chunker = TensorChunker(self.config)

    @unittest.skipUnless(TensorChunker._compression_available(CompressionAlgorithm.LZ4),
                        "LZ4 not available")
    def test_lz4_compression(self):
        """Test LZ4 compression."""
        tensor = torch.randn(500, 500)
        original_bytes = self.chunker._tensor_to_bytes(tensor)

        compressed = self.chunker.compress_chunk(original_bytes, CompressionAlgorithm.LZ4)
        decompressed = self.chunker.decompress_chunk(compressed, CompressionAlgorithm.LZ4)

        # Should be able to reconstruct tensor
        reconstructed = self.chunker._bytes_to_tensor(decompressed, tensor.shape, str(tensor.dtype))
        self.assertTrue(torch.allclose(tensor, reconstructed))

        # Compressed should be smaller (for this data)
        self.assertLess(len(compressed), len(original_bytes))

    @unittest.skipUnless(TensorChunker._compression_available(CompressionAlgorithm.ZSTD),
                        "ZSTD not available")
    def test_zstd_compression(self):
        """Test ZSTD compression."""
        tensor = torch.randn(500, 500)
        original_bytes = self.chunker._tensor_to_bytes(tensor)

        compressed = self.chunker.compress_chunk(original_bytes, CompressionAlgorithm.ZSTD)
        decompressed = self.chunker.decompress_chunk(compressed, CompressionAlgorithm.ZSTD)

        reconstructed = self.chunker._bytes_to_tensor(decompressed, tensor.shape, str(tensor.dtype))
        self.assertTrue(torch.allclose(tensor, reconstructed))

    @unittest.skipUnless(TensorChunker._compression_available(CompressionAlgorithm.GZIP),
                        "GZIP not available")
    def test_gzip_compression(self):
        """Test GZIP compression."""
        tensor = torch.randn(500, 500)
        original_bytes = self.chunker._tensor_to_bytes(tensor)

        compressed = self.chunker.compress_chunk(original_bytes, CompressionAlgorithm.GZIP)
        decompressed = self.chunker.decompress_chunk(compressed, CompressionAlgorithm.GZIP)

        reconstructed = self.chunker._bytes_to_tensor(decompressed, tensor.shape, str(tensor.dtype))
        self.assertTrue(torch.allclose(tensor, reconstructed))

    def test_no_compression(self):
        """Test no compression mode."""
        tensor = torch.randn(100, 100)
        original_bytes = self.chunker._tensor_to_bytes(tensor)

        # No compression should return original data
        compressed = self.chunker.compress_chunk(original_bytes, CompressionAlgorithm.NONE)
        self.assertEqual(compressed, original_bytes)

        decompressed = self.chunker.decompress_chunk(compressed, CompressionAlgorithm.NONE)
        self.assertEqual(decompressed, original_bytes)


class TestStreamingOperations(unittest.TestCase):
    """Test streaming tensor operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorChunkingConfig(max_tensor_size_mb=1)
        self.chunker = TensorChunker(self.config)
        self.metadata_tracker = ChunkMetadataTracker()
        self.stream_processor = StreamingTensorProcessor(self.chunker, self.metadata_tracker)

    def test_stream_map_operation_small_tensor(self):
        """Test streaming map operation on small tensor."""
        tensor = torch.randn(10, 10)

        def double_operation(x):
            return x * 2

        result = self.stream_processor.stream_map_operation(tensor, double_operation)
        expected = tensor * 2

        self.assertTrue(torch.allclose(result, expected))

    def test_stream_map_operation_large_tensor(self):
        """Test streaming map operation on large tensor."""
        tensor = torch.randn(1000, 100)  # Should be chunked

        def square_operation(x):
            return x ** 2

        result = self.stream_processor.stream_map_operation(tensor, square_operation)
        expected = tensor ** 2

        self.assertTrue(torch.allclose(result, expected))

    def test_stream_reduce_operation(self):
        """Test streaming reduce operation."""
        tensor = torch.randn(500, 100)

        def sum_reduce(accumulator, chunk):
            if accumulator is None:
                return chunk.sum()
            return accumulator + chunk.sum()

        result = self.stream_processor.stream_reduce_operation(tensor, sum_reduce)
        expected = tensor.sum()

        self.assertAlmostEqual(result.item(), expected.item(), places=5)


class TestMemoryMonitoring(unittest.TestCase):
    """Test memory monitoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = MemoryMonitor(max_memory_mb=100)

    def test_memory_recording(self):
        """Test memory usage recording."""
        # Record some memory usage
        self.monitor.record_memory_usage("test_operation", 50.5)
        self.monitor.update_cache_size(25.0)

        current_usage = self.monitor.get_current_memory_usage()
        self.assertIn("chunk_cache_mb", current_usage)
        self.assertEqual(current_usage["chunk_cache_mb"], 25.0)

        history = self.monitor.get_memory_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["operation"], "test_operation")
        self.assertEqual(history[0]["memory_mb"], 50.5)

    def test_cache_size_reduction_check(self):
        """Test cache size reduction logic."""
        # Normal cache size
        self.monitor.update_cache_size(50.0)
        self.assertFalse(self.monitor.should_reduce_cache())

        # High cache size
        self.monitor.update_cache_size(90.0)
        self.assertTrue(self.monitor.should_reduce_cache())


class TestEnhancedTensorChunker(unittest.TestCase):
    """Test enhanced tensor chunker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorChunkingConfig(max_tensor_size_mb=1)
        self.enhanced_chunker = EnhancedTensorChunker(self.config)

    def test_comprehensive_chunking_workflow(self):
        """Test complete chunking workflow with tracking."""
        # Create large tensor
        tensor = torch.randn(500, 500)
        tensor_id = "test_comprehensive"

        # Chunk with tracking
        chunks = self.enhanced_chunker.chunk_tensor_with_tracking(tensor, tensor_id)

        # Should have a single chunk
        self.assertEqual(len(chunks), 1)

        # Reconstruct with tracking
        reconstructed = self.enhanced_chunker.reconstruct_tensor_with_tracking(chunks)

        # Should be identical
        self.assertTrue(torch.allclose(tensor, reconstructed))

        # Check metadata tracking
        stats = self.enhanced_chunker.get_comprehensive_stats()
        self.assertIn("metadata_stats", stats)
        self.assertIn("performance_stats", stats)
        self.assertEqual(stats["performance_stats"]["chunking_operations"], 1)
        self.assertEqual(stats["performance_stats"]["reconstruction_operations"], 1)

    def test_streaming_with_tracking(self):
        """Test streaming operations with tracking."""
        tensor = torch.randn(300, 300)

        def normalize_operation(x):
            return (x - x.mean()) / x.std()

        result = self.enhanced_chunker.stream_operation_with_tracking(
            tensor, normalize_operation, "map"
        )

        # Verify operation was applied correctly
        expected = (tensor - tensor.mean()) / tensor.std()
        self.assertTrue(torch.allclose(result, expected))

        # Check tracking
        stats = self.enhanced_chunker.get_comprehensive_stats()
        self.assertEqual(stats["performance_stats"]["streaming_operations"], 1)

    def test_optimization_for_tensor(self):
        """Test configuration optimization for different tensor sizes."""
        # Small tensor
        small_tensor = torch.randn(10, 10)
        small_config = self.enhanced_chunker.optimize_for_tensor(small_tensor)
        self.assertLessEqual(small_config.chunk_size_mb, 50)

        # Large tensor
        large_tensor = torch.randn(6000, 6000)  # Very large
        large_config = self.enhanced_chunker.optimize_for_tensor(large_tensor)
        self.assertGreater(large_config.chunk_size_mb, 100)


class TestChunkMetadataTracker(unittest.TestCase):
    """Test chunk metadata tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ChunkMetadataTracker()

    def test_metadata_registration(self):
        """Test chunk metadata registration and retrieval."""
        tensor_id = "test_tensor"
        mock_chunks = [
            ChunkMetadata(
                chunk_id=f"{tensor_id}_0",
                tensor_id=tensor_id,
                chunk_index=0,
                total_chunks=2,
                original_shape=(1000, 1000),
                chunk_shape=(500, 1000),
                data_type="torch.float32",
                compression=CompressionAlgorithm.LZ4,
                compressed_size=1024,
                uncompressed_size=2048,
                checksum="mock_checksum",
                chunk_range=(0, 500)
            ),
            ChunkMetadata(
                chunk_id=f"{tensor_id}_1",
                tensor_id=tensor_id,
                chunk_index=1,
                total_chunks=2,
                original_shape=(1000, 1000),
                chunk_shape=(500, 1000),
                data_type="torch.float32",
                compression=CompressionAlgorithm.LZ4,
                compressed_size=1024,
                uncompressed_size=2048,
                checksum="mock_checksum",
                chunk_range=(500, 1000)
            )
        ]

        # Register chunks
        self.tracker.register_chunks(tensor_id, mock_chunks)

        # Check stats
        stats = self.tracker.get_tensor_stats(tensor_id)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["num_chunks"], 2)
        self.assertEqual(stats["total_compressed_size"], 2048)

        # Test system stats
        system_stats = self.tracker.get_system_stats()
        self.assertEqual(system_stats["total_chunked_tensors"], 1)
        self.assertEqual(system_stats["total_chunks"], 2)

    def test_access_tracking(self):
        """Test access tracking functionality."""
        tensor_id = "test_access"

        # Register tensor
        mock_chunk = ChunkMetadata(
            chunk_id=f"{tensor_id}_0",
            tensor_id=tensor_id,
            chunk_index=0,
            total_chunks=1,
            original_shape=(100, 100),
            chunk_shape=(100, 100),
            data_type="torch.float32",
            compression=CompressionAlgorithm.NONE,
            compressed_size=1024,
            uncompressed_size=1024,
            checksum="test",
            chunk_range=(0, 100)
        )
        self.tracker.register_chunks(tensor_id, [mock_chunk])

        # Record access
        self.tracker.record_access(tensor_id, "read")

        # Check access count
        stats = self.tracker.get_tensor_stats(tensor_id)
        self.assertEqual(stats["access_count"], 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorChunkingConfig(max_tensor_size_mb=1)
        self.chunker = TensorChunker(self.config)

    def test_empty_tensor(self):
        """Test handling of empty tensors."""
        empty_tensor = torch.tensor([])
        chunks = self.chunker.chunk_tensor(empty_tensor, "empty")
        reconstructed = self.chunker.reconstruct_tensor(chunks)
        self.assertEqual(len(reconstructed), 0)

    def test_single_element_tensor(self):
        """Test handling of single element tensors."""
        single_tensor = torch.tensor([42.0])
        chunks = self.chunker.chunk_tensor(single_tensor, "single")
        reconstructed = self.chunker.reconstruct_tensor(chunks)
        self.assertEqual(reconstructed.item(), 42.0)

    def test_very_large_dimensions(self):
        """Test tensors with very large dimensions."""
        # Create tensor that will definitely be chunked
        large_tensor = torch.randn(10000, 10)  # 100K elements
        self.assertFalse(self.chunker.should_chunk_tensor(large_tensor))

        chunks = self.chunker.chunk_tensor(large_tensor, "very_large")
        reconstructed = self.chunker.reconstruct_tensor(chunks)

        self.assertTrue(torch.allclose(large_tensor, reconstructed))

    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        # Create some chunks
        tensor = torch.randn(200, 200)
        chunks = self.chunker.chunk_tensor(tensor, "cleanup_test")

        # Check memory usage before cleanup
        memory_before = self.chunker.get_memory_usage()
        cache_size_before = memory_before["cache_size_mb"]

        # Cleanup
        self.chunker.clear_cache()

        # Check memory usage after cleanup
        memory_after = self.chunker.get_memory_usage()
        cache_size_after = memory_after["cache_size_mb"]

        # Cache should be smaller or equal
        self.assertLessEqual(cache_size_after, cache_size_before)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorChunkingConfig(
            max_tensor_size_mb=10,
            chunk_size_mb=5
        )
        self.chunker = TensorChunker(self.config)

    def test_chunking_performance(self):
        """Benchmark chunking performance."""
        # Create test tensor
        tensor = torch.randn(1000, 1000)  # ~4MB

        # Time chunking operation
        start_time = time.time()
        chunks = self.chunker.chunk_tensor(tensor, "benchmark")
        chunking_time = time.time() - start_time

        # Time reconstruction
        start_time = time.time()
        reconstructed = self.chunker.reconstruct_tensor(chunks)
        reconstruction_time = time.time() - start_time

        # Verify correctness
        self.assertTrue(torch.allclose(tensor, reconstructed))

        # Performance assertions (these are rough benchmarks)
        self.assertLess(chunking_time, 1.0, "Chunking should be fast")
        self.assertLess(reconstruction_time, 1.0, "Reconstruction should be fast")

    def test_compression_performance(self):
        """Benchmark compression performance."""
        tensor = torch.randn(500, 500)
        original_bytes = self.chunker._tensor_to_bytes(tensor)

        # Test different compression algorithms
        algorithms = [CompressionAlgorithm.NONE]
        if self.chunker._compression_available(CompressionAlgorithm.LZ4):
            algorithms.append(CompressionAlgorithm.LZ4)

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                start_time = time.time()
                compressed = self.chunker.compress_chunk(original_bytes, algorithm)
                compression_time = time.time() - start_time

                start_time = time.time()
                decompressed = self.chunker.decompress_chunk(compressed, algorithm)
                decompression_time = time.time() - start_time

                # Verify round-trip
                self.assertEqual(decompressed, original_bytes)

                # Performance check
                self.assertLess(compression_time, 0.1, f"{algorithm.value} compression should be fast")
                self.assertLess(decompression_time, 0.1, f"{algorithm.value} decompression should be fast")


if __name__ == '__main__':
    # Add more detailed test output
    unittest.main(verbosity=2)
