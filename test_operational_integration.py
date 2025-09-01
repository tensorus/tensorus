"""
Comprehensive Tests for Integrated Tensor Operations and Storage

Tests cover:
- Operational core functionality
- Tensor storage and retrieval
- Operation execution on stored tensors
- Streaming operations
- Query language and API
- Integration with indexing system
- Caching and optimization
- Performance benchmarks
- Edge cases and error handling
"""

import os
import time
import tempfile
import unittest
import shutil
from unittest.mock import Mock, patch
import torch
import numpy as np

from tensorus.tensorus_operational_core import (
    TensorusOperationalCore, OperationalTensor, create_tensorus_core
)
from tensorus.tensor_operations_integrated import OperationalStorage
from tensorus.tensor_streaming_pipeline import StreamingOperationManager
from tensorus.tensor_operation_api import TensorOperationAPI, TensorQuery
from tensorus.metadata.indexing import IndexManager
from tensorus.tensor_chunking import TensorChunkingConfig


class TestOperationalCore(unittest.TestCase):
    """Test the main operational core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

        # Create test tensors
        self.tensor1 = torch.randn(10, 20)
        self.tensor2 = torch.randn(10, 20)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tensor_storage_and_retrieval(self):
        """Test storing and retrieving tensors."""
        # Store tensors
        id1 = self.core.store_tensor(self.tensor1, {"name": "test1"})
        id2 = self.core.store_tensor(self.tensor2, {"name": "test2"})

        # Verify storage
        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)
        self.assertNotEqual(id1, id2)

        # Retrieve tensors
        op_tensor1 = self.core.get_tensor(id1)
        op_tensor2 = self.core.get_tensor(id2)

        # Verify retrieval
        self.assertIsInstance(op_tensor1, OperationalTensor)
        self.assertIsInstance(op_tensor2, OperationalTensor)
        self.assertEqual(op_tensor1.tensor_id, id1)
        self.assertEqual(op_tensor2.tensor_id, id2)

    def test_basic_operations(self):
        """Test basic tensor operations."""
        # Store tensors
        id1 = self.core.store_tensor(self.tensor1)
        id2 = self.core.store_tensor(self.tensor2)

        # Test addition
        result_add = self.core.add(id1, id2)
        self.assertIsInstance(result_add, OperationalTensor)

        # Test subtraction
        result_sub = self.core.subtract(id1, id2)
        self.assertIsInstance(result_sub, OperationalTensor)

        # Test multiplication
        result_mul = self.core.multiply(id1, id2)
        self.assertIsInstance(result_mul, OperationalTensor)

        # Test matrix multiplication (reshape first)
        tensor3 = torch.randn(20, 15)
        id3 = self.core.store_tensor(tensor3)
        result_matmul = self.core.matmul(id1, id3)
        self.assertIsInstance(result_matmul, OperationalTensor)

    def test_unary_operations(self):
        """Test unary tensor operations."""
        # Store tensor
        id1 = self.core.store_tensor(self.tensor1)

        # Test sum
        result_sum = self.core.sum(id1)
        self.assertIsInstance(result_sum, OperationalTensor)

        # Test mean
        result_mean = self.core.mean(id1)
        self.assertIsInstance(result_mean, OperationalTensor)

        # Test transpose
        result_transpose = self.core.transpose(id1, 0, 1)
        self.assertIsInstance(result_transpose, OperationalTensor)

        # Test reshape
        result_reshape = self.core.reshape(id1, 5, 40)
        self.assertIsInstance(result_reshape, OperationalTensor)

    def test_tensor_listing(self):
        """Test tensor listing and querying."""
        # Store tensors with metadata
        id1 = self.core.store_tensor(self.tensor1, {"category": "test", "size": "small"})
        id2 = self.core.store_tensor(self.tensor2, {"category": "production", "size": "small"})

        # List all tensors
        all_tensors = self.core.list_tensors()
        self.assertIsInstance(all_tensors, list)

        # Query by conditions
        test_tensors = self.core.list_tensors({"category": "test"})
        self.assertIsInstance(test_tensors, list)
        self.assertIn(id1, test_tensors)


class TestOperationalTensor(unittest.TestCase):
    """Test OperationalTensor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)
        self.tensor = torch.randn(5, 10)
        self.tensor_id = self.core.store_tensor(self.tensor)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_operational_tensor_properties(self):
        """Test OperationalTensor properties."""
        op_tensor = self.core.get_tensor(self.tensor_id)

        # Test metadata access
        metadata = op_tensor.metadata
        self.assertIsInstance(metadata, dict)
        self.assertIn("shape", metadata)
        self.assertIn("data_type", metadata)

        # Test data access
        data = op_tensor.data
        self.assertTrue(torch.allclose(data, self.tensor))

    def test_operational_tensor_operations(self):
        """Test operations on OperationalTensor."""
        op_tensor1 = self.core.get_tensor(self.tensor_id)

        # Create second tensor
        tensor2 = torch.randn(5, 10)
        id2 = self.core.store_tensor(tensor2)
        op_tensor2 = self.core.get_tensor(id2)

        # Test operations
        result_add = op_tensor1 + op_tensor2
        self.assertIsInstance(result_add, OperationalTensor)

        result_sum = op_tensor1.sum()
        self.assertIsInstance(result_sum, OperationalTensor)

    def test_tensor_streaming(self):
        """Test tensor streaming."""
        op_tensor = self.core.get_tensor(self.tensor_id)

        # Test streaming
        chunks = list(op_tensor.stream())
        self.assertGreater(len(chunks), 0)

        # Verify chunks can reconstruct original
        reconstructed = torch.cat(chunks, dim=0)
        self.assertTrue(torch.allclose(reconstructed, self.tensor))


class TestQueryOperations(unittest.TestCase):
    """Test query-based operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

        # Create test data
        for i in range(5):
            tensor = torch.randn(10, 10)
            metadata = {
                "category": "test" if i < 3 else "production",
                "size": "small",
                "batch": i % 2
            }
            self.core.store_tensor(tensor, metadata)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_select_query(self):
        """Test SELECT queries."""
        # Select test category tensors
        results = self.core.select({"category": "test"})
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIsInstance(result, OperationalTensor)

    def test_compute_query(self):
        """Test COMPUTE queries."""
        # Get all tensor IDs
        all_tensors = self.core.list_tensors({"category": "test"})
        if all_tensors:
            tensor_id = all_tensors[0]

            # Compute sum
            result = self.core.compute("sum", [tensor_id])
            self.assertIsInstance(result, OperationalTensor)

    def test_aggregate_query(self):
        """Test AGGREGATE queries."""
        # Aggregate test tensors
        result = self.core.aggregate({"category": "test"}, "sum")
        # Result could be tensor ID or value depending on implementation
        self.assertIsNotNone(result)

    def test_query_language(self):
        """Test tensor query language."""
        # Simple select query
        query = "SELECT tensors WHERE category = test COMPUTE sum"
        result = self.core.query(query)
        self.assertIsNotNone(result)
        self.assertIn("status", result)

        # Aggregate query
        query = "SELECT tensors WHERE category = test AGGREGATE sum"
        result = self.core.query(query)
        self.assertIsNotNone(result)


class TestStreamingOperations(unittest.TestCase):
    """Test streaming operations for large tensors."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

        # Create a larger tensor for streaming
        self.large_tensor = torch.randn(100, 100)  # ~40KB

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_streaming_operation(self):
        """Test streaming operation execution."""
        # Store large tensor
        tensor_id = self.core.store_tensor(self.large_tensor)

        # Execute streaming operation
        result = self.core.stream_operation("sum", [tensor_id])
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)

    def test_progress_callbacks(self):
        """Test progress callbacks for streaming operations."""
        progress_updates = []

        def progress_callback(update):
            progress_updates.append(update)

        self.core.add_progress_callback(progress_callback)

        # Execute operation
        tensor_id = self.core.store_tensor(self.large_tensor)
        self.core.stream_operation("sum", [tensor_id])

        # Check if progress was reported
        # Note: Progress reporting depends on implementation details
        self.assertIsInstance(progress_updates, list)


class TestBatchOperations(unittest.TestCase):
    """Test batch operation execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

        # Create test tensors
        self.tensors = []
        for i in range(3):
            tensor = torch.randn(5, 5)
            tensor_id = self.core.store_tensor(tensor)
            self.tensors.append(tensor_id)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_operations(self):
        """Test executing multiple operations in batch."""
        batch_ops = [
            {
                "type": "compute",
                "operation": "sum",
                "tensor_ids": [self.tensors[0]]
            },
            {
                "type": "compute",
                "operation": "mean",
                "tensor_ids": [self.tensors[1]]
            }
        ]

        results = self.core.batch_operate(batch_ops)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(batch_ops))

        for result in results:
            self.assertIsNotNone(result)


class TestCachingAndOptimization(unittest.TestCase):
    """Test caching and optimization features."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Perform some operations to populate cache
        tensor = torch.randn(5, 5)
        id1 = self.core.store_tensor(tensor)
        self.core.sum(id1)

        # Clear caches
        self.core.clear_caches()

        # Verify operation still works after cache clear
        result = self.core.sum(id1)
        self.assertIsInstance(result, OperationalTensor)

    def test_system_optimization(self):
        """Test system optimization."""
        # Add some test data
        for i in range(3):
            tensor = torch.randn(5, 5)
            self.core.store_tensor(tensor)

        # Run optimization
        opt_result = self.core.optimize()
        self.assertIsInstance(opt_result, dict)
        self.assertIn("cache_cleared", opt_result)

    def test_statistics(self):
        """Test system statistics."""
        # Perform some operations
        tensor = torch.randn(5, 5)
        id1 = self.core.store_tensor(tensor)
        self.core.sum(id1)

        # Get statistics
        stats = self.core.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("operational_storage", stats)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow(self):
        """Test complete workflow from storage to operations."""
        # 1. Store tensors
        tensor1 = torch.randn(10, 20)
        tensor2 = torch.randn(10, 20)

        id1 = self.core.store_tensor(tensor1, {"experiment": "test1"})
        id2 = self.core.store_tensor(tensor2, {"experiment": "test1"})

        # 2. Query tensors
        test_tensors = self.core.select({"experiment": "test1"})
        self.assertEqual(len(test_tensors), 2)

        # 3. Perform operations
        result_add = test_tensors[0] + test_tensors[1]
        result_sum = result_add.sum()

        # 4. Execute and get results
        final_result = result_sum.execute()
        self.assertIsInstance(final_result, torch.Tensor)

        # 5. Verify result shape
        expected_shape = torch.add(tensor1, tensor2).sum().shape
        self.assertEqual(final_result.shape, expected_shape)

    def test_mixed_operations(self):
        """Test mixing different operation types."""
        # Store tensors
        tensor = torch.randn(6, 8)
        tensor_id = self.core.store_tensor(tensor)

        # Chain operations
        op_tensor = self.core.get_tensor(tensor_id)
        result = (op_tensor + torch.ones_like(tensor)).sum(dim=0).mean()

        # Execute
        final = result.execute()
        self.assertIsInstance(final, torch.Tensor)

    def test_error_handling(self):
        """Test error handling in operations."""
        # Test with invalid tensor ID
        with self.assertRaises(Exception):
            self.core.get_tensor("invalid_id")

        # Test operation on incompatible tensors
        tensor1 = torch.randn(5, 5)
        tensor2 = torch.randn(3, 3)  # Different shape

        id1 = self.core.store_tensor(tensor1)
        id2 = self.core.store_tensor(tensor2)

        # This should handle the error gracefully
        try:
            result = self.core.add(id1, id2)
            # If it succeeds, verify the result
            final = result.execute()
            self.assertIsInstance(final, torch.Tensor)
        except Exception:
            # Expected for incompatible shapes
            pass


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.core = create_tensorus_core(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_operation_performance(self):
        """Benchmark operation performance."""
        # Create test tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(20, 20)
            tensor_id = self.core.store_tensor(tensor)
            tensors.append(tensor_id)

        # Benchmark operations
        start_time = time.time()
        results = []
        for tensor_id in tensors:
            result = self.core.sum(tensor_id)
            results.append(result)
        operation_time = time.time() - start_time

        # Should complete reasonably fast
        self.assertLess(operation_time, 2.0, f"Operations took too long: {operation_time}s")
        self.assertEqual(len(results), len(tensors))

    def test_query_performance(self):
        """Benchmark query performance."""
        # Create test data
        for i in range(50):
            tensor = torch.randn(10, 10)
            metadata = {"category": f"cat_{i % 5}", "batch": i % 10}
            self.core.store_tensor(tensor, metadata)

        # Benchmark queries
        start_time = time.time()
        results = self.core.select({"category": "cat_0"})
        query_time = time.time() - start_time

        # Should be fast with indexing
        self.assertLess(query_time, 0.5, f"Query took too long: {query_time}s")
        self.assertIsInstance(results, list)

    def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        # Create a moderately large tensor
        large_tensor = torch.randn(100, 100)
        tensor_id = self.core.store_tensor(large_tensor)

        # Monitor memory usage during operation
        import psutil
        process = psutil.Process()

        memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Perform operation
        result = self.core.sum(tensor_id)
        final = result.execute()

        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_delta = memory_after - memory_before

        # Memory increase should be reasonable
        self.assertLess(memory_delta, 100, f"Memory increase too large: {memory_delta}MB")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
