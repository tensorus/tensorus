#!/usr/bin/env python3
"""
Storage-connected tensor operations for Tensorus.

This module provides integration between TensorOps and TensorStorage,
allowing operations to be performed directly on database-resident tensors
without requiring manual retrieval and storage.

Features:
- Direct operations on tensor IDs stored in TensorStorage
- Result caching to avoid redundant computations
- Batch operations for multiple tensors
- Automatic result storage with metadata tracking
- Lazy evaluation and operation queuing
"""

import torch
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

from .tensor_ops import TensorOps


class OperationResult:
    """Container for operation results with metadata."""
    
    def __init__(self, tensor: torch.Tensor, operation: str, inputs: List[str], 
                 computation_time: float, cached: bool = False):
        self.tensor = tensor
        self.operation = operation
        self.inputs = inputs  # Input tensor IDs or values
        self.computation_time = computation_time
        self.cached = cached
        self.timestamp = time.time()


class StorageConnectedTensorOps:
    """
    Tensor operations that are connected to TensorStorage.
    
    This class extends TensorOps functionality to work directly with
    stored tensors, providing caching, batch operations, and automatic
    result management.
    """
    
    def __init__(self, storage):
        """
        Initialize with a TensorStorage instance.
        
        Args:
            storage: TensorStorage instance to operate on
        """
        self.storage = storage
        self._result_cache = {}  # Cache for computed results
        self._cache_enabled = True
        self._max_cache_size = 100
        self._operation_history = []
        
        logger = logging.getLogger(__name__)
        logger.info("StorageConnectedTensorOps initialized with caching enabled")
    
    def enable_caching(self, max_size: int = 100) -> None:
        """Enable result caching with specified maximum cache size."""
        self._cache_enabled = True
        self._max_cache_size = max_size
        logging.info(f"Result caching enabled with max size {max_size}")
    
    def disable_caching(self) -> None:
        """Disable result caching and clear existing cache."""
        self._cache_enabled = False
        self._result_cache.clear()
        logging.info("Result caching disabled")
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._result_cache.clear()
        logging.info("Result cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self._cache_enabled,
            "size": len(self._result_cache),
            "max_size": self._max_cache_size,
            "operations_cached": list(self._result_cache.keys())
        }
    
    def _generate_cache_key(self, operation: str, inputs: List[str], **kwargs) -> str:
        """Generate a unique cache key for an operation."""
        # Create deterministic hash from operation, inputs, and parameters
        key_data = f"{operation}:{':'.join(sorted(inputs))}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[OperationResult]:
        """Retrieve cached result if available."""
        if not self._cache_enabled:
            return None
        return self._result_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: OperationResult) -> None:
        """Cache an operation result."""
        if not self._cache_enabled:
            return
        
        # Implement LRU eviction if cache is full
        if len(self._result_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._result_cache.keys(), 
                           key=lambda k: self._result_cache[k].timestamp)
            del self._result_cache[oldest_key]
        
        self._result_cache[cache_key] = result
    
    def _retrieve_tensor(self, dataset_name: str, tensor_id: str) -> torch.Tensor:
        """Retrieve a tensor from storage."""
        result = self.storage.get_tensor_by_id(dataset_name, tensor_id)
        return result["tensor"]
    
    def _store_result(self, dataset_name: str, result: OperationResult, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store operation result in the storage system."""
        if metadata is None:
            metadata = {}
        
        # Add operation metadata
        metadata.update({
            "operation": result.operation,
            "input_tensors": result.inputs,
            "computation_time": result.computation_time,
            "cached": result.cached,
            "operation_timestamp": result.timestamp
        })
        
        return self.storage.insert(dataset_name, result.tensor, metadata)
    
    # === Binary Operations ===
    
    def add(self, dataset_name: str, tensor_id1: str, tensor_id2: Union[str, float, int],
            store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """
        Add two tensors or a tensor and scalar, with automatic caching.
        
        Args:
            dataset_name: Name of the dataset containing the tensors
            tensor_id1: ID of the first tensor
            tensor_id2: ID of the second tensor or a scalar value
            store_result: Whether to store the result in the dataset
            result_metadata: Additional metadata for the result
            
        Returns:
            str: Record ID of stored result if store_result=True
            OperationResult: Operation result if store_result=False
        """
        inputs = [tensor_id1, str(tensor_id2)]
        cache_key = self._generate_cache_key("add", inputs)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logging.debug(f"Cache hit for add operation: {cache_key}")
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        # Perform computation
        start_time = time.time()
        tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
        
        if isinstance(tensor_id2, str):
            tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
        else:
            tensor2 = tensor_id2
        
        result_tensor = TensorOps.add(tensor1, tensor2)
        computation_time = time.time() - start_time
        
        # Create result object
        result = OperationResult(
            tensor=result_tensor,
            operation="add",
            inputs=inputs,
            computation_time=computation_time,
            cached=False
        )
        
        # Cache the result
        self._cache_result(cache_key, result)
        
        # Store or return result
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    def subtract(self, dataset_name: str, tensor_id1: str, tensor_id2: Union[str, float, int],
                store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Subtract two tensors or a tensor and scalar."""
        inputs = [tensor_id1, str(tensor_id2)]
        cache_key = self._generate_cache_key("subtract", inputs)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
        
        if isinstance(tensor_id2, str):
            tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
        else:
            tensor2 = tensor_id2
        
        result_tensor = TensorOps.subtract(tensor1, tensor2)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="subtract",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    def multiply(self, dataset_name: str, tensor_id1: str, tensor_id2: Union[str, float, int],
                store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Multiply two tensors or a tensor and scalar."""
        inputs = [tensor_id1, str(tensor_id2)]
        cache_key = self._generate_cache_key("multiply", inputs)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
        
        if isinstance(tensor_id2, str):
            tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
        else:
            tensor2 = tensor_id2
        
        result_tensor = TensorOps.multiply(tensor1, tensor2)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="multiply",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    def matmul(self, dataset_name: str, tensor_id1: str, tensor_id2: str,
              store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Matrix multiplication of two stored tensors."""
        inputs = [tensor_id1, tensor_id2]
        cache_key = self._generate_cache_key("matmul", inputs)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
        tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
        
        result_tensor = TensorOps.matmul(tensor1, tensor2)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="matmul",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    # === Reduction Operations ===
    
    def sum(self, dataset_name: str, tensor_id: str, dim: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdim: bool = False, store_result: bool = True, 
            result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Sum of tensor elements over given dimensions."""
        inputs = [tensor_id]
        cache_key = self._generate_cache_key("sum", inputs, dim=dim, keepdim=keepdim)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor = self._retrieve_tensor(dataset_name, tensor_id)
        result_tensor = TensorOps.sum(tensor, dim=dim, keepdim=keepdim)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="sum",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    def mean(self, dataset_name: str, tensor_id: str, dim: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdim: bool = False, store_result: bool = True,
             result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Mean of tensor elements over given dimensions."""
        inputs = [tensor_id]
        cache_key = self._generate_cache_key("mean", inputs, dim=dim, keepdim=keepdim)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor = self._retrieve_tensor(dataset_name, tensor_id)
        result_tensor = TensorOps.mean(tensor, dim=dim, keepdim=keepdim)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="mean",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    # === Reshaping Operations ===
    
    def reshape(self, dataset_name: str, tensor_id: str, shape: Tuple[int, ...],
               store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Reshape a stored tensor."""
        inputs = [tensor_id]
        cache_key = self._generate_cache_key("reshape", inputs, shape=shape)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor = self._retrieve_tensor(dataset_name, tensor_id)
        result_tensor = TensorOps.reshape(tensor, shape)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="reshape",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    def transpose(self, dataset_name: str, tensor_id: str, dim0: int, dim1: int,
                 store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None) -> Union[str, OperationResult]:
        """Transpose dimensions of a stored tensor."""
        inputs = [tensor_id]
        cache_key = self._generate_cache_key("transpose", inputs, dim0=dim0, dim1=dim1)
        
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            # Create a new result marking it as cached
            cache_hit_result = OperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True
            )
            if store_result:
                return self._store_result(dataset_name, cache_hit_result, result_metadata)
            return cache_hit_result
        
        start_time = time.time()
        tensor = self._retrieve_tensor(dataset_name, tensor_id)
        result_tensor = TensorOps.transpose(tensor, dim0, dim1)
        computation_time = time.time() - start_time
        
        result = OperationResult(
            tensor=result_tensor,
            operation="transpose",
            inputs=inputs,
            computation_time=computation_time
        )
        
        self._cache_result(cache_key, result)
        
        if store_result:
            return self._store_result(dataset_name, result, result_metadata)
        return result
    
    # === Linear Algebra Operations ===
    
    def svd(self, dataset_name: str, tensor_id: str, store_result: bool = True,
            result_metadata: Optional[Dict[str, Any]] = None) -> Union[Tuple[str, str, str], Tuple[OperationResult, OperationResult, OperationResult]]:
        """Singular Value Decomposition of a stored matrix."""
        inputs = [tensor_id]
        cache_key = self._generate_cache_key("svd", inputs)
        
        # For SVD, we need to handle three result tensors
        cached_result = self._get_cached_result(cache_key)
        if cached_result and hasattr(cached_result, 'tensors'):  # Multi-tensor result
            if store_result:
                u_id = self._store_result(dataset_name, 
                    OperationResult(cached_result.tensors[0], "svd_u", inputs, 0, True), result_metadata)
                s_id = self._store_result(dataset_name, 
                    OperationResult(cached_result.tensors[1], "svd_s", inputs, 0, True), result_metadata)
                vt_id = self._store_result(dataset_name, 
                    OperationResult(cached_result.tensors[2], "svd_vt", inputs, 0, True), result_metadata)
                return u_id, s_id, vt_id
            return cached_result
        
        start_time = time.time()
        tensor = self._retrieve_tensor(dataset_name, tensor_id)
        u, s, vt = TensorOps.svd(tensor)
        computation_time = time.time() - start_time
        
        # Create multi-tensor result
        class MultiTensorResult:
            def __init__(self, tensors, operation, inputs, computation_time):
                self.tensors = tensors
                self.operation = operation
                self.inputs = inputs
                self.computation_time = computation_time
                self.cached = False
                self.timestamp = time.time()
        
        result = MultiTensorResult([u, s, vt], "svd", inputs, computation_time)
        self._cache_result(cache_key, result)
        
        if store_result:
            u_id = self._store_result(dataset_name, 
                OperationResult(u, "svd_u", inputs, computation_time), result_metadata)
            s_id = self._store_result(dataset_name, 
                OperationResult(s, "svd_s", inputs, computation_time), result_metadata)
            vt_id = self._store_result(dataset_name, 
                OperationResult(vt, "svd_vt", inputs, computation_time), result_metadata)
            return u_id, s_id, vt_id
        
        return (OperationResult(u, "svd_u", inputs, computation_time),
                OperationResult(s, "svd_s", inputs, computation_time), 
                OperationResult(vt, "svd_vt", inputs, computation_time))
    
    # === Batch Operations ===
    
    def batch_operation(self, dataset_name: str, operation: str, tensor_ids: List[str],
                       **operation_kwargs) -> List[str]:
        """
        Apply an operation to multiple tensors in batch.
        
        Args:
            dataset_name: Name of the dataset
            operation: Name of the operation to perform
            tensor_ids: List of tensor IDs to operate on
            **operation_kwargs: Additional arguments for the operation
            
        Returns:
            List[str]: List of result tensor IDs
        """
        results = []
        
        # Get the operation method
        if not hasattr(self, operation):
            raise ValueError(f"Unknown operation: {operation}")
        
        op_method = getattr(self, operation)
        
        for tensor_id in tensor_ids:
            if operation in ["add", "subtract", "multiply"]:
                # Binary operations require second operand
                result_id = op_method(dataset_name, tensor_id, **operation_kwargs)
            else:
                # Unary operations
                result_id = op_method(dataset_name, tensor_id, **operation_kwargs)
            results.append(result_id)
        
        return results
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of all performed operations."""
        return self._operation_history.copy()
    
    def benchmark_operation(self, dataset_name: str, operation: str, tensor_id: str,
                          iterations: int = 10, **kwargs) -> Dict[str, float]:
        """
        Benchmark an operation over multiple iterations.
        
        Returns:
            Dict with timing statistics
        """
        times = []
        
        # Disable caching for benchmarking
        original_cache_state = self._cache_enabled
        self.disable_caching()
        
        try:
            for _ in range(iterations):
                start_time = time.time()
                getattr(self, operation)(dataset_name, tensor_id, store_result=False, **kwargs)
                times.append(time.time() - start_time)
        finally:
            # Restore cache state
            if original_cache_state:
                self.enable_caching()
        
        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "iterations": iterations
        }


class TensorStorageWithOps:
    """
    Extension of TensorStorage with integrated operations.
    
    This class provides a convenient interface for performing operations
    directly on stored tensors without manual management.
    """
    
    def __init__(self, storage):
        """Initialize with a TensorStorage instance."""
        self.storage = storage
        self.ops = StorageConnectedTensorOps(storage)
    
    def __getattr__(self, name):
        """Delegate to storage for unknown attributes."""
        return getattr(self.storage, name)
    
    # Convenience methods for common operations
    
    def tensor_add(self, dataset_name: str, tensor_id1: str, tensor_id2: Union[str, float, int],
                  store_result: bool = True) -> Union[str, OperationResult]:
        """Add two tensors or tensor and scalar."""
        return self.ops.add(dataset_name, tensor_id1, tensor_id2, store_result)
    
    def tensor_matmul(self, dataset_name: str, tensor_id1: str, tensor_id2: str,
                     store_result: bool = True) -> Union[str, OperationResult]:
        """Matrix multiply two tensors."""
        return self.ops.matmul(dataset_name, tensor_id1, tensor_id2, store_result)
    
    def tensor_reshape(self, dataset_name: str, tensor_id: str, shape: Tuple[int, ...],
                      store_result: bool = True) -> Union[str, OperationResult]:
        """Reshape a tensor."""
        return self.ops.reshape(dataset_name, tensor_id, shape, store_result)
    
    def enable_operation_caching(self, max_size: int = 100):
        """Enable result caching for operations."""
        self.ops.enable_caching(max_size)
    
    def disable_operation_caching(self):
        """Disable result caching for operations."""
        self.ops.disable_caching()
    
    def get_operation_cache_stats(self) -> Dict[str, Any]:
        """Get operation cache statistics."""
        return self.ops.get_cache_stats()


def create_storage_with_ops(storage_path: str, **storage_kwargs):
    """
    Factory function to create TensorStorage with integrated operations.
    
    Args:
        storage_path: Path to storage directory
        **storage_kwargs: Additional arguments for TensorStorage
        
    Returns:
        TensorStorageWithOps instance
    """
    from .tensor_storage import TensorStorage
    
    storage = TensorStorage(storage_path, **storage_kwargs)
    return TensorStorageWithOps(storage)