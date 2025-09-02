#!/usr/bin/env python3
"""
Test suite for storage-connected tensor operations.

Tests the integration between TensorOps and TensorStorage,
ensuring operations work correctly on database-resident tensors.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any

# Import the modules to test
try:
    from tensorus.storage_ops import StorageConnectedTensorOps, OperationResult, TensorStorageWithOps
    from tensorus.tensor_storage import TensorStorage
    STORAGE_OPS_AVAILABLE = True
except ImportError:
    STORAGE_OPS_AVAILABLE = False

# Skip all tests if storage ops module is not available
pytestmark = pytest.mark.skipif(not STORAGE_OPS_AVAILABLE, reason="Storage ops module not available")


class TestStorageConnectedTensorOps:
    """Test StorageConnectedTensorOps functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.storage = TensorStorage(storage_path=str(self.temp_path))
        self.ops = StorageConnectedTensorOps(self.storage)
        
        # Create a test dataset
        self.storage.create_dataset("test_ops")
        
        # Add some test tensors
        self.tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.tensor_b = torch.tensor([[2.0, 0.0], [1.0, 2.0]]) 
        self.tensor_c = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        self.id_a = self.storage.insert("test_ops", self.tensor_a, {"name": "matrix_a"})
        self.id_b = self.storage.insert("test_ops", self.tensor_b, {"name": "matrix_b"})
        self.id_c = self.storage.insert("test_ops", self.tensor_c, {"name": "vector_c"})
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ops_initialization(self):
        """Test StorageConnectedTensorOps initialization."""
        assert self.ops.storage == self.storage
        assert self.ops._cache_enabled == True
        assert self.ops._max_cache_size == 100
        assert len(self.ops._result_cache) == 0
    
    def test_caching_control(self):
        """Test enabling/disabling caching."""
        # Test disable
        self.ops.disable_caching()
        assert self.ops._cache_enabled == False
        assert len(self.ops._result_cache) == 0
        
        # Test enable
        self.ops.enable_caching(50)
        assert self.ops._cache_enabled == True
        assert self.ops._max_cache_size == 50
        
        # Test clear cache
        self.ops._result_cache["test"] = "dummy"
        self.ops.clear_cache()
        assert len(self.ops._result_cache) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.ops.get_cache_stats()
        assert "enabled" in stats
        assert "size" in stats
        assert "max_size" in stats
        assert "operations_cached" in stats
        assert stats["enabled"] == True
        assert stats["size"] == 0
        assert stats["max_size"] == 100
    
    def test_add_operation(self):
        """Test addition operation on stored tensors."""
        # Test tensor + tensor
        result_id = self.ops.add("test_ops", self.id_a, self.id_b, store_result=True)
        assert isinstance(result_id, str)
        
        # Verify result
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor([[3.0, 2.0], [4.0, 6.0]])
        assert torch.allclose(result["tensor"], expected)
        assert result["metadata"]["operation"] == "add"
        assert result["metadata"]["input_tensors"] == [self.id_a, self.id_b]
        
        # Test tensor + scalar
        scalar_result_id = self.ops.add("test_ops", self.id_a, 5.0, store_result=True)
        scalar_result = self.storage.get_tensor_by_id("test_ops", scalar_result_id)
        expected_scalar = torch.tensor([[6.0, 7.0], [8.0, 9.0]])
        assert torch.allclose(scalar_result["tensor"], expected_scalar)
    
    def test_add_operation_no_store(self):
        """Test addition operation without storing result."""
        result = self.ops.add("test_ops", self.id_a, self.id_b, store_result=False)
        assert isinstance(result, OperationResult)
        assert result.operation == "add"
        assert result.inputs == [self.id_a, self.id_b]
        expected = torch.tensor([[3.0, 2.0], [4.0, 6.0]])
        assert torch.allclose(result.tensor, expected)
        assert result.computation_time > 0
        assert result.cached == False
    
    def test_subtract_operation(self):
        """Test subtraction operation."""
        result_id = self.ops.subtract("test_ops", self.id_a, self.id_b)
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor([[-1.0, 2.0], [2.0, 2.0]])
        assert torch.allclose(result["tensor"], expected)
        assert result["metadata"]["operation"] == "subtract"
    
    def test_multiply_operation(self):
        """Test multiplication operation."""
        result_id = self.ops.multiply("test_ops", self.id_a, self.id_b)
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor([[2.0, 0.0], [3.0, 8.0]])
        assert torch.allclose(result["tensor"], expected)
        assert result["metadata"]["operation"] == "multiply"
    
    def test_matmul_operation(self):
        """Test matrix multiplication operation."""
        result_id = self.ops.matmul("test_ops", self.id_a, self.id_b)
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
        assert torch.allclose(result["tensor"], expected)
        assert result["metadata"]["operation"] == "matmul"
    
    def test_sum_operation(self):
        """Test sum reduction operation."""
        # Sum all elements
        result_id = self.ops.sum("test_ops", self.id_a)
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor(10.0)  # 1+2+3+4
        assert torch.allclose(result["tensor"], expected)
        
        # Sum along dimension
        result_id_dim = self.ops.sum("test_ops", self.id_a, dim=0)
        result_dim = self.storage.get_tensor_by_id("test_ops", result_id_dim)
        expected_dim = torch.tensor([4.0, 6.0])  # sum along rows
        assert torch.allclose(result_dim["tensor"], expected_dim)
    
    def test_mean_operation(self):
        """Test mean reduction operation."""
        result_id = self.ops.mean("test_ops", self.id_a)
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor(2.5)  # (1+2+3+4)/4
        assert torch.allclose(result["tensor"], expected)
    
    def test_reshape_operation(self):
        """Test reshape operation."""
        result_id = self.ops.reshape("test_ops", self.id_c, (2, 2))
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(result["tensor"], expected)
        assert result["tensor"].shape == (2, 2)
    
    def test_transpose_operation(self):
        """Test transpose operation."""
        result_id = self.ops.transpose("test_ops", self.id_a, 0, 1)
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        assert torch.allclose(result["tensor"], expected)
    
    def test_svd_operation(self):
        """Test SVD operation."""
        u_id, s_id, vt_id = self.ops.svd("test_ops", self.id_a)
        
        # Verify all three components are stored
        u_result = self.storage.get_tensor_by_id("test_ops", u_id)
        s_result = self.storage.get_tensor_by_id("test_ops", s_id)
        vt_result = self.storage.get_tensor_by_id("test_ops", vt_id)
        
        # Reconstruct original matrix
        reconstructed = u_result["tensor"] @ torch.diag(s_result["tensor"]) @ vt_result["tensor"]
        assert torch.allclose(reconstructed, self.tensor_a, atol=1e-5)
        
        # Check metadata
        assert u_result["metadata"]["operation"] == "svd_u"
        assert s_result["metadata"]["operation"] == "svd_s"
        assert vt_result["metadata"]["operation"] == "svd_vt"
    
    def test_caching_functionality(self):
        """Test that caching works correctly."""
        # First operation should not be cached
        result1 = self.ops.add("test_ops", self.id_a, self.id_b, store_result=False)
        assert result1.cached == False
        
        # Second identical operation should be cached
        result2 = self.ops.add("test_ops", self.id_a, self.id_b, store_result=False)
        assert result2.cached == True  # Should be retrieved from cache
        
        # Results should be identical
        assert torch.allclose(result1.tensor, result2.tensor)
        
        # Cache stats should show the operation
        stats = self.ops.get_cache_stats()
        assert stats["size"] == 1
    
    def test_cache_key_generation(self):
        """Test cache key generation with different parameters."""
        # Different tensor IDs should have different keys
        key1 = self.ops._generate_cache_key("add", [self.id_a, self.id_b])
        key2 = self.ops._generate_cache_key("add", [self.id_b, self.id_a])
        assert key1 == key2  # Should be order-independent due to sorting
        
        # Different operations should have different keys
        key3 = self.ops._generate_cache_key("subtract", [self.id_a, self.id_b])
        assert key1 != key3
        
        # Different parameters should have different keys
        key4 = self.ops._generate_cache_key("sum", [self.id_a], dim=0)
        key5 = self.ops._generate_cache_key("sum", [self.id_a], dim=1)
        assert key4 != key5
    
    def test_batch_operations(self):
        """Test batch operations on multiple tensors."""
        tensor_ids = [self.id_a, self.id_b, self.id_c]
        
        # Batch add with scalar
        result_ids = self.ops.batch_operation("test_ops", "add", tensor_ids, 
                                            tensor_id2=10.0, store_result=True)
        assert len(result_ids) == 3
        
        # Verify first result
        result = self.storage.get_tensor_by_id("test_ops", result_ids[0])
        expected = torch.tensor([[11.0, 12.0], [13.0, 14.0]])
        assert torch.allclose(result["tensor"], expected)
    
    def test_metadata_preservation(self):
        """Test that operation metadata is properly stored."""
        custom_metadata = {"experiment": "test_run", "custom_version": "1.0"}
        result_id = self.ops.add("test_ops", self.id_a, self.id_b, 
                                result_metadata=custom_metadata)
        
        result = self.storage.get_tensor_by_id("test_ops", result_id)
        metadata = result["metadata"]
        
        # Check custom metadata
        assert metadata["experiment"] == "test_run"
        assert metadata["custom_version"] == "1.0"
        
        # Check operation metadata
        assert metadata["operation"] == "add"
        assert metadata["input_tensors"] == [self.id_a, self.id_b]
        assert "computation_time" in metadata
        assert "operation_timestamp" in metadata
    
    def test_benchmark_operation(self):
        """Test operation benchmarking."""
        benchmark_result = self.ops.benchmark_operation(
            "test_ops", "add", self.id_a, iterations=5, tensor_id2=self.id_b
        )
        
        assert "mean_time" in benchmark_result
        assert "min_time" in benchmark_result
        assert "max_time" in benchmark_result
        assert "total_time" in benchmark_result
        assert "iterations" in benchmark_result
        assert benchmark_result["iterations"] == 5
        assert benchmark_result["mean_time"] > 0
        assert benchmark_result["min_time"] <= benchmark_result["mean_time"]
        assert benchmark_result["mean_time"] <= benchmark_result["max_time"]


class TestTensorStorageWithOps:
    """Test TensorStorageWithOps convenience class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = TensorStorage(storage_path=self.temp_dir)
        self.storage_with_ops = TensorStorageWithOps(self.storage)
        
        # Create test dataset and tensors
        self.storage_with_ops.create_dataset("test")
        self.tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.tensor_b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
        
        self.id_a = self.storage_with_ops.insert("test", self.tensor_a, {"name": "a"})
        self.id_b = self.storage_with_ops.insert("test", self.tensor_b, {"name": "b"})
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_delegation(self):
        """Test that storage methods are properly delegated."""
        # Test that normal storage methods work
        datasets = self.storage_with_ops.list_datasets()
        assert "test" in datasets
        
        # Test tensor retrieval
        result = self.storage_with_ops.get_tensor_by_id("test", self.id_a)
        assert torch.allclose(result["tensor"], self.tensor_a)
    
    def test_convenience_methods(self):
        """Test convenience methods for operations."""
        # Test tensor_add
        result_id = self.storage_with_ops.tensor_add("test", self.id_a, self.id_b)
        result = self.storage_with_ops.get_tensor_by_id("test", result_id)
        expected = torch.tensor([[3.0, 2.0], [4.0, 6.0]])
        assert torch.allclose(result["tensor"], expected)
        
        # Test tensor_matmul
        matmul_id = self.storage_with_ops.tensor_matmul("test", self.id_a, self.id_b)
        matmul_result = self.storage_with_ops.get_tensor_by_id("test", matmul_id)
        expected_matmul = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
        assert torch.allclose(matmul_result["tensor"], expected_matmul)
        
        # Test tensor_reshape
        reshape_id = self.storage_with_ops.tensor_reshape("test", self.id_a, (4,))
        reshape_result = self.storage_with_ops.get_tensor_by_id("test", reshape_id)
        expected_reshape = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(reshape_result["tensor"], expected_reshape)
    
    def test_caching_methods(self):
        """Test caching control methods."""
        # Test enable caching
        self.storage_with_ops.enable_operation_caching(50)
        stats = self.storage_with_ops.get_operation_cache_stats()
        assert stats["enabled"] == True
        assert stats["max_size"] == 50
        
        # Test disable caching
        self.storage_with_ops.disable_operation_caching()
        stats = self.storage_with_ops.get_operation_cache_stats()
        assert stats["enabled"] == False


class TestTensorStorageIntegration:
    """Test direct TensorStorage integration methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = TensorStorage(storage_path=self.temp_dir)
        
        # Create test dataset and tensors
        self.storage.create_dataset("integration_test")
        self.tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.tensor_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        self.id_a = self.storage.insert("integration_test", self.tensor_a, {"name": "a"})
        self.id_b = self.storage.insert("integration_test", self.tensor_b, {"name": "b"})
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_direct_tensor_operations(self):
        """Test direct tensor operations on storage."""
        # Test tensor_add
        result_id = self.storage.tensor_add("integration_test", self.id_a, self.id_b)
        result = self.storage.get_tensor_by_id("integration_test", result_id)
        expected = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
        assert torch.allclose(result["tensor"], expected)
        
        # Test tensor_subtract
        sub_id = self.storage.tensor_subtract("integration_test", self.id_b, self.id_a)
        sub_result = self.storage.get_tensor_by_id("integration_test", sub_id)
        expected_sub = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
        assert torch.allclose(sub_result["tensor"], expected_sub)
        
        # Test tensor_multiply
        mul_id = self.storage.tensor_multiply("integration_test", self.id_a, 2.0)
        mul_result = self.storage.get_tensor_by_id("integration_test", mul_id)
        expected_mul = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        assert torch.allclose(mul_result["tensor"], expected_mul)
    
    def test_reduction_operations(self):
        """Test reduction operations."""
        # Test tensor_sum
        sum_id = self.storage.tensor_sum("integration_test", self.id_a)
        sum_result = self.storage.get_tensor_by_id("integration_test", sum_id)
        expected_sum = torch.tensor(10.0)  # 1+2+3+4
        assert torch.allclose(sum_result["tensor"], expected_sum)
        
        # Test tensor_mean with dimension
        mean_id = self.storage.tensor_mean("integration_test", self.id_a, dim=0)
        mean_result = self.storage.get_tensor_by_id("integration_test", mean_id)
        expected_mean = torch.tensor([2.0, 3.0])  # mean along dim 0
        assert torch.allclose(mean_result["tensor"], expected_mean)
    
    def test_shape_operations(self):
        """Test shape manipulation operations."""
        # Test tensor_reshape
        reshape_id = self.storage.tensor_reshape("integration_test", self.id_a, (4,))
        reshape_result = self.storage.get_tensor_by_id("integration_test", reshape_id)
        expected_reshape = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(reshape_result["tensor"], expected_reshape)
        
        # Test tensor_transpose
        transpose_id = self.storage.tensor_transpose("integration_test", self.id_a, 0, 1)
        transpose_result = self.storage.get_tensor_by_id("integration_test", transpose_id)
        expected_transpose = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        assert torch.allclose(transpose_result["tensor"], expected_transpose)
    
    def test_linear_algebra_operations(self):
        """Test linear algebra operations."""
        # Test tensor_matmul
        matmul_id = self.storage.tensor_matmul("integration_test", self.id_a, self.id_b)
        matmul_result = self.storage.get_tensor_by_id("integration_test", matmul_id)
        expected_matmul = torch.matmul(self.tensor_a, self.tensor_b)
        assert torch.allclose(matmul_result["tensor"], expected_matmul)
        
        # Test tensor_svd
        u_id, s_id, vt_id = self.storage.tensor_svd("integration_test", self.id_a)
        
        # Verify reconstruction
        u_result = self.storage.get_tensor_by_id("integration_test", u_id)
        s_result = self.storage.get_tensor_by_id("integration_test", s_id)
        vt_result = self.storage.get_tensor_by_id("integration_test", vt_id)
        
        reconstructed = u_result["tensor"] @ torch.diag(s_result["tensor"]) @ vt_result["tensor"]
        assert torch.allclose(reconstructed, self.tensor_a, atol=1e-5)
    
    def test_batch_operations(self):
        """Test batch operations."""
        tensor_ids = [self.id_a, self.id_b]
        
        result_ids = self.storage.batch_tensor_operation(
            "integration_test", "add", tensor_ids, tensor_id2=1.0
        )
        
        assert len(result_ids) == 2
        
        # Check first result
        result1 = self.storage.get_tensor_by_id("integration_test", result_ids[0])
        expected1 = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        assert torch.allclose(result1["tensor"], expected1)
    
    def test_caching_integration(self):
        """Test caching functionality integration."""
        # Enable caching
        self.storage.enable_operation_caching(10)
        
        # Perform operation twice
        result_id1 = self.storage.tensor_add("integration_test", self.id_a, self.id_b)
        result_id2 = self.storage.tensor_add("integration_test", self.id_a, self.id_b)
        
        # Both should have same result but different IDs (since stored)
        result1 = self.storage.get_tensor_by_id("integration_test", result_id1)
        result2 = self.storage.get_tensor_by_id("integration_test", result_id2)
        assert torch.allclose(result1["tensor"], result2["tensor"])
        assert result_id1 != result_id2  # Different storage IDs
        
        # Check cache stats
        stats = self.storage.get_operation_cache_stats()
        assert stats["enabled"] == True
        assert stats["size"] >= 1
    
    def test_benchmarking(self):
        """Test operation benchmarking."""
        benchmark = self.storage.benchmark_tensor_operation(
            "integration_test", "add", self.id_a, iterations=3, tensor_id2=self.id_b
        )
        
        assert "mean_time" in benchmark
        assert "iterations" in benchmark
        assert benchmark["iterations"] == 3
        assert benchmark["mean_time"] > 0


class TestErrorHandling:
    """Test error handling in storage operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = TensorStorage(storage_path=self.temp_dir)
        self.ops = StorageConnectedTensorOps(self.storage)
        
        self.storage.create_dataset("error_test")
        self.tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.tensor_id = self.storage.insert("error_test", self.tensor, {"name": "test"})
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_tensor_id(self):
        """Test handling of invalid tensor IDs."""
        with pytest.raises(Exception):  # Should raise TensorNotFoundError
            self.ops.add("error_test", "invalid_id", self.tensor_id)
    
    def test_invalid_dataset(self):
        """Test handling of invalid dataset names."""
        with pytest.raises(Exception):  # Should raise DatasetNotFoundError
            self.ops.add("invalid_dataset", self.tensor_id, 5.0)
    
    def test_shape_mismatch_operations(self):
        """Test handling of shape mismatches in operations."""
        # Create incompatible tensor
        incompatible_tensor = torch.tensor([1.0, 2.0, 3.0])  # 1D tensor
        incompatible_id = self.storage.insert("error_test", incompatible_tensor, {"name": "incompatible"})
        
        with pytest.raises(Exception):  # Should raise RuntimeError for matmul
            self.ops.matmul("error_test", self.tensor_id, incompatible_id)


if __name__ == "__main__":
    pytest.main([__file__])