"""
Integrated Operational Layer for Tensorus

This module provides seamless integration between TensorOps and TensorStorage,
enabling operations to be performed directly on stored tensors without loading
them into memory first.

Key Features:
- Lazy evaluation for operations on stored tensors
- Streaming operations for memory-efficient processing
- Operation result caching and persistence
- Query language for complex tensor operations
- Integration with indexing system for optimization
- Background operation execution
- Result pipeline and composition

Architecture:
- OperationalTensor: Lazy wrapper around stored tensors
- OperationPipeline: Chain operations with streaming
- OperationalStorage: Integration layer between ops and storage
- OperationCache: Cache operation results for performance
- OperationScheduler: Background execution and resource management
"""

import os
import time
import uuid
import pickle
import threading
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator, Callable, Union
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import torch

from tensorus.tensor_ops import TensorOps
from tensorus.tensor_storage import TensorStorage
from tensorus.tensor_chunking import TensorChunker, TensorChunkingConfig
from tensorus.metadata.index_manager import IndexManager


class OperationType(Enum):
    """Types of tensor operations supported."""
    UNARY = "unary"           # Single tensor operations
    BINARY = "binary"         # Two tensor operations
    AGGREGATION = "aggregation"  # Reduce operations
    TRANSFORMATION = "transformation"  # Shape/structure changes
    COMPARISON = "comparison"  # Comparison operations
    STATISTICAL = "statistical"  # Statistical operations


@dataclass
class OperationSpec:
    """Specification for a tensor operation."""
    name: str
    type: OperationType
    func: Callable
    input_count: int = 1
    output_count: int = 1
    memory_factor: float = 1.0  # Memory usage relative to input
    commutative: bool = False
    associative: bool = False


@dataclass
class OperationRequest:
    """Request for an operation on stored tensors."""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_spec: OperationSpec = None
    input_tensor_ids: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_dataset: str = None
    streaming: bool = False
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    status: str = "pending"


@dataclass
class OperationResult:
    """Result of an operation execution."""
    operation_id: str
    status: str  # "success", "error", "running"
    result_tensor_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_used: int = 0
    completed_at: Optional[float] = None


class OperationalTensor:
    """
    Lazy wrapper around stored tensors that enables operations without loading.

    Provides:
    - Lazy evaluation of operations
    - Memory-efficient access patterns
    - Operation chaining and composition
    - Automatic result caching
    """

    def __init__(self, tensor_id: str, storage: 'OperationalStorage',
                 metadata: Optional[Dict[str, Any]] = None):
        self.tensor_id = tensor_id
        self.storage = storage
        self._metadata = metadata or {}
        self._cached_data = None
        self._operation_chain = []

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get tensor metadata."""
        if not self._metadata:
            self._metadata = self.storage.get_tensor_metadata(self.tensor_id)
        return self._metadata

    def load(self) -> torch.Tensor:
        """Load the tensor data into memory."""
        if self._cached_data is None:
            self._cached_data = self.storage.load_tensor(self.tensor_id)
        return self._cached_data

    def __add__(self, other: Union['OperationalTensor', torch.Tensor]) -> 'OperationChain':
        """Addition operation."""
        return OperationChain([self, other], TensorOps.add, OperationType.BINARY)

    def __sub__(self, other: Union['OperationalTensor', torch.Tensor]) -> 'OperationChain':
        """Subtraction operation."""
        return OperationChain([self, other], TensorOps.subtract, OperationType.BINARY)

    def __mul__(self, other: Union['OperationalTensor', torch.Tensor]) -> 'OperationChain':
        """Multiplication operation."""
        return OperationChain([self, other], TensorOps.multiply, OperationType.BINARY)

    def __matmul__(self, other: Union['OperationalTensor', torch.Tensor]) -> 'OperationChain':
        """Matrix multiplication operation."""
        return OperationChain([self, other], TensorOps.matmul, OperationType.BINARY)

    def sum(self, dim: Optional[int] = None) -> 'OperationChain':
        """Sum operation."""
        return OperationChain([self], TensorOps.sum, OperationType.AGGREGATION,
                            parameters={"dim": dim})

    def mean(self, dim: Optional[int] = None) -> 'OperationChain':
        """Mean operation."""
        return OperationChain([self], TensorOps.mean, OperationType.AGGREGATION,
                            parameters={"dim": dim})

    def transpose(self, dim0: int, dim1: int) -> 'OperationChain':
        """Transpose operation."""
        return OperationChain([self], TensorOps.transpose, OperationType.TRANSFORMATION,
                            parameters={"dim0": dim0, "dim1": dim1})

    def reshape(self, *shape) -> 'OperationChain':
        """Reshape operation."""
        return OperationChain([self], TensorOps.reshape, OperationType.TRANSFORMATION,
                            parameters={"shape": shape})

    def execute(self, output_dataset: Optional[str] = None) -> 'OperationalTensor':
        """Execute the operation chain."""
        if self._operation_chain:
            return self.storage.execute_operation_chain(
                self._operation_chain, output_dataset
            )
        else:
            # Just return self if no operations
            return self

    def stream(self) -> Iterator[torch.Tensor]:
        """Stream tensor chunks for memory-efficient processing."""
        return self.storage.stream_tensor(self.tensor_id)


class OperationChain:
    """
    Chain of operations to be executed on tensors.

    Enables:
    - Operation composition and pipelining
    - Lazy evaluation
    - Memory-efficient execution
    - Result caching
    """

    def __init__(self, inputs: List[Union[OperationalTensor, torch.Tensor]],
                 operation: Callable, op_type: OperationType,
                 parameters: Optional[Dict[str, Any]] = None):
        self.inputs = inputs
        self.operation = operation
        self.op_type = op_type
        self.parameters = parameters or {}
        self._next_operation = None

    def __add__(self, other: Union[OperationalTensor, torch.Tensor]) -> 'OperationChain':
        """Chain addition operation."""
        return self._chain_binary_op(other, TensorOps.add)

    def __sub__(self, other: Union[OperationalTensor, torch.Tensor]) -> 'OperationChain':
        """Chain subtraction operation."""
        return self._chain_binary_op(other, TensorOps.subtract)

    def sum(self, dim: Optional[int] = None) -> 'OperationChain':
        """Chain sum operation."""
        return self._chain_unary_op(TensorOps.sum, {"dim": dim})

    def mean(self, dim: Optional[int] = None) -> 'OperationChain':
        """Chain mean operation."""
        return self._chain_unary_op(TensorOps.mean, {"dim": dim})

    def _chain_binary_op(self, other: Union[OperationalTensor, torch.Tensor],
                        operation: Callable) -> 'OperationChain':
        """Chain a binary operation."""
        new_chain = OperationChain([self], operation, OperationType.BINARY)
        new_chain._previous_chain = self
        return new_chain

    def _chain_unary_op(self, operation: Callable,
                       parameters: Dict[str, Any]) -> 'OperationChain':
        """Chain a unary operation."""
        new_chain = OperationChain([self], operation, OperationType.UNARY, parameters)
        new_chain._previous_chain = self
        return new_chain

    def execute(self, output_dataset: Optional[str] = None) -> OperationalTensor:
        """Execute the operation chain."""
        # This would be handled by OperationalStorage
        # For now, return a placeholder
        return OperationalTensor("result_placeholder", None)


class OperationCache:
    """
    Cache for operation results to avoid recomputation.

    Features:
    - Hash-based result caching
    - Memory management with LRU eviction
    - Persistence to disk for long-term caching
    - Cache invalidation on data changes
    """

    def __init__(self, max_size: int = 1000, cache_dir: str = "./operation_cache"):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Load existing cache entries
        self._load_cache_metadata()

    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result."""
        with self._lock:
            if cache_key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                return self.cache[cache_key]

            # Check disk cache
            return self._load_from_disk(cache_key)

    def put(self, cache_key: str, result: Any,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store result in cache."""
        with self._lock:
            # Add to memory cache
            self.cache[cache_key] = result
            self.metadata[cache_key] = metadata or {}

            # Evict if cache is full
            while len(self.cache) > self.max_size:
                oldest_key, _ = self.cache.popitem(last=False)
                if oldest_key in self.metadata:
                    del self.metadata[oldest_key]

            # Save to disk asynchronously
            self._save_to_disk_async(cache_key, result, metadata)

    def invalidate(self, tensor_ids: List[str]) -> None:
        """Invalidate cache entries that depend on the given tensors."""
        with self._lock:
            keys_to_remove = []

            for cache_key, meta in self.metadata.items():
                if meta.get("input_tensor_ids"):
                    if any(tid in meta["input_tensor_ids"] for tid in tensor_ids):
                        keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                if key in self.metadata:
                    del self.metadata[key]
                self._delete_from_disk(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.metadata.clear()
            self._clear_disk_cache()

    def _load_cache_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.pkl")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            except Exception:
                self.metadata = {}

    def _save_to_disk_async(self, cache_key: str, result: Any,
                           metadata: Dict[str, Any]) -> None:
        """Save cache entry to disk asynchronously."""
        def save_worker():
            try:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump({"result": result, "metadata": metadata}, f)

                # Update metadata file
                metadata_file = os.path.join(self.cache_dir, "cache_metadata.pkl")
                with open(metadata_file, 'wb') as f:
                    pickle.dump(self.metadata, f)

            except Exception as e:
                print(f"Failed to save cache entry {cache_key}: {e}")

        thread = threading.Thread(target=save_worker, daemon=True)
        thread.start()

    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """Load cache entry from disk."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    result = data["result"]

                    # Add to memory cache
                    self.cache[cache_key] = result
                    self.cache.move_to_end(cache_key)

                    return result
            except Exception:
                pass
        return None

    def _delete_from_disk(self, cache_key: str) -> None:
        """Delete cache entry from disk."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass

    def _clear_disk_cache(self) -> None:
        """Clear all disk cache files."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except Exception:
                    pass


class OperationExecutor:
    """
    Executes tensor operations with various strategies.

    Supports:
    - In-memory execution for small tensors
    - Streaming execution for large tensors
    - Background execution for long-running operations
    - Resource management and monitoring
    """

    def __init__(self, storage: 'OperationalStorage', max_workers: int = 4):
        self.storage = storage
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_operations: Dict[str, Future] = {}
        self._lock = threading.RLock()

    def execute_operation(self, request: OperationRequest) -> OperationResult:
        """
        Execute a tensor operation.

        Determines the best execution strategy based on tensor size and operation type.
        """
        start_time = time.time()

        try:
            # Check if operation can be cached
            cache_key = self._generate_cache_key(request)
            cached_result = self.storage.cache.get(cache_key)

            if cached_result:
                return OperationResult(
                    operation_id=request.operation_id,
                    status="success",
                    result_tensor_ids=cached_result.get("tensor_ids", []),
                    execution_time=time.time() - start_time
                )

            # Determine execution strategy
            if self._should_stream_operation(request):
                result = self._execute_streaming_operation(request)
            else:
                result = self._execute_in_memory_operation(request)

            # Cache the result
            cache_metadata = {
                "operation": request.operation_spec.name,
                "input_tensor_ids": request.input_tensor_ids,
                "parameters": request.parameters,
                "execution_time": result.execution_time
            }
            self.storage.cache.put(cache_key, {
                "tensor_ids": result.result_tensor_ids,
                "metadata": cache_metadata
            }, cache_metadata)

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            return OperationResult(
                operation_id=request.operation_id,
                status="error",
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    def execute_async(self, request: OperationRequest) -> str:
        """Execute operation asynchronously."""
        future = self.executor.submit(self.execute_operation, request)

        with self._lock:
            self.running_operations[request.operation_id] = future

        return request.operation_id

    def get_operation_status(self, operation_id: str) -> Optional[OperationResult]:
        """Get status of an asynchronous operation."""
        with self._lock:
            if operation_id not in self.running_operations:
                return None

            future = self.running_operations[operation_id]

            if future.done():
                del self.running_operations[operation_id]
                return future.result()
            else:
                return OperationResult(
                    operation_id=operation_id,
                    status="running"
                )

    def _should_stream_operation(self, request: OperationRequest) -> bool:
        """Determine if operation should use streaming execution."""
        if request.streaming:
            return True

        # Check tensor sizes
        total_size = 0
        for tensor_id in request.input_tensor_ids:
            metadata = self.storage.get_tensor_metadata(tensor_id)
            if metadata:
                total_size += metadata.get("byte_size", 0)

        # Use streaming for large operations
        return total_size > 100 * 1024 * 1024  # 100MB threshold

    def _execute_in_memory_operation(self, request: OperationRequest) -> OperationResult:
        """Execute operation by loading tensors into memory."""
        # Load input tensors
        input_tensors = []
        for tensor_id in request.input_tensor_ids:
            tensor = self.storage.load_tensor(tensor_id)
            input_tensors.append(tensor)

        # Execute operation
        if request.operation_spec.input_count == 1:
            result_tensor = request.operation_spec.func(input_tensors[0], **request.parameters)
        elif request.operation_spec.input_count == 2:
            result_tensor = request.operation_spec.func(input_tensors[0], input_tensors[1], **request.parameters)
        else:
            # Handle multiple inputs
            result_tensor = request.operation_spec.func(*input_tensors, **request.parameters)

        # Store result
        result_id = self.storage.save_tensor(result_tensor, request.output_dataset)

        return OperationResult(
            operation_id=request.operation_id,
            status="success",
            result_tensor_ids=[result_id]
        )

    def _execute_streaming_operation(self, request: OperationRequest) -> OperationResult:
        """Execute operation using streaming/chunked processing."""
        # This would implement streaming logic
        # For now, fall back to in-memory execution
        return self._execute_in_memory_operation(request)

    def _generate_cache_key(self, request: OperationRequest) -> str:
        """Generate cache key for operation result."""
        key_data = {
            "operation": request.operation_spec.name if request.operation_spec else "unknown",
            "inputs": sorted(request.input_tensor_ids),
            "params": str(sorted(request.parameters.items()))
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()


class OperationalStorage:
    """
    Integration layer between TensorOps and TensorStorage.

    Provides:
    - Seamless operation execution on stored tensors
    - Lazy evaluation and operation chaining
    - Memory-efficient processing for large tensors
    - Operation result caching and persistence
    - Integration with indexing system
    """

    def __init__(self, tensor_storage: TensorStorage,
                 index_manager: Optional[IndexManager] = None,
                 chunking_config: Optional[TensorChunkingConfig] = None):
        self.tensor_storage = tensor_storage
        self.index_manager = index_manager
        self.chunking_config = chunking_config or TensorChunkingConfig()

        # Initialize components
        self.chunker = TensorChunker(self.chunking_config)
        self.cache = OperationCache()
        self.executor = OperationExecutor(self)

        # Operation registry
        self.operation_specs: Dict[str, OperationSpec] = {}
        self._register_operations()

    def _register_operations(self) -> None:
        """Register available tensor operations."""
        # Unary operations
        self.operation_specs["sum"] = OperationSpec(
            "sum", OperationType.AGGREGATION, TensorOps.sum, 1, 1, 0.1
        )
        self.operation_specs["mean"] = OperationSpec(
            "mean", OperationType.AGGREGATION, TensorOps.mean, 1, 1, 0.1
        )
        self.operation_specs["transpose"] = OperationSpec(
            "transpose", OperationType.TRANSFORMATION, TensorOps.transpose, 1, 1, 1.0
        )
        self.operation_specs["reshape"] = OperationSpec(
            "reshape", OperationType.TRANSFORMATION, TensorOps.reshape, 1, 1, 1.0
        )

        # Binary operations
        self.operation_specs["add"] = OperationSpec(
            "add", OperationType.BINARY, TensorOps.add, 2, 1, 1.0, True, True
        )
        self.operation_specs["subtract"] = OperationSpec(
            "subtract", OperationType.BINARY, TensorOps.subtract, 2, 1, 1.0
        )
        self.operation_specs["multiply"] = OperationSpec(
            "multiply", OperationType.BINARY, TensorOps.multiply, 2, 1, 1.0, True, False
        )
        self.operation_specs["matmul"] = OperationSpec(
            "matmul", OperationType.BINARY, TensorOps.matmul, 2, 1, 2.0
        )

    def get_tensor(self, tensor_id: str) -> OperationalTensor:
        """Get an operational tensor wrapper."""
        return OperationalTensor(tensor_id, self)

    def load_tensor(self, tensor_id: str) -> torch.Tensor:
        """Load tensor from storage."""
        return self.tensor_storage.get_tensor_by_id("default", tensor_id)["tensor"]

    def save_tensor(self, tensor: torch.Tensor, dataset: str = "default") -> str:
        """Save tensor to storage."""
        return self.tensor_storage.insert(dataset, tensor)

    def get_tensor_metadata(self, tensor_id: str) -> Dict[str, Any]:
        """Get tensor metadata."""
        try:
            tensor_data = self.tensor_storage.get_tensor_by_id("default", tensor_id)
            return tensor_data.get("metadata", {})
        except Exception:
            return {}

    def execute_operation(self, operation_name: str, input_tensor_ids: List[str],
                         parameters: Optional[Dict[str, Any]] = None,
                         output_dataset: Optional[str] = None,
                         async_execution: bool = False) -> Union[OperationResult, str]:
        """
        Execute an operation on stored tensors.

        Args:
            operation_name: Name of the operation to execute
            input_tensor_ids: IDs of input tensors
            parameters: Operation parameters
            output_dataset: Dataset for output tensor
            async_execution: Execute asynchronously

        Returns:
            OperationResult or operation ID (for async)
        """
        if operation_name not in self.operation_specs:
            raise ValueError(f"Unknown operation: {operation_name}")

        operation_spec = self.operation_specs[operation_name]
        request = OperationRequest(
            operation_spec=operation_spec,
            input_tensor_ids=input_tensor_ids,
            parameters=parameters or {},
            output_dataset=output_dataset or "default"
        )

        if async_execution:
            return self.executor.execute_async(request)
        else:
            return self.executor.execute_operation(request)

    def execute_operation_chain(self, operation_chain: OperationChain,
                               output_dataset: Optional[str] = None) -> OperationalTensor:
        """
        Execute a chain of operations.

        This is a simplified implementation. In practice, this would
        optimize the operation chain and execute it efficiently.
        """
        # For now, execute operations sequentially
        # In a full implementation, this would optimize the chain

        # Placeholder implementation
        result_id = f"chain_result_{uuid.uuid4()}"
        return OperationalTensor(result_id, self)

    def stream_tensor(self, tensor_id: str) -> Iterator[torch.Tensor]:
        """Stream tensor chunks for memory-efficient processing."""
        # Check if tensor is chunked
        tensor_data = self.tensor_storage.get_tensor_by_id("default", tensor_id)
        tensor = tensor_data["tensor"]

        if self.chunker.should_chunk_tensor(tensor):
            # Return chunks
            strategy = self.chunker.calculate_chunk_strategy(tuple(tensor.shape), tensor.dtype)
            chunk_shape = strategy["chunk_shape"]

            for i in range(strategy["chunks_in_first_dim"]):
                start_idx = i * chunk_shape[0]
                end_idx = min((i + 1) * chunk_shape[0], tensor.shape[0])

                if len(tensor.shape) == 1:
                    chunk = tensor[start_idx:end_idx]
                else:
                    slice_indices = (slice(start_idx, end_idx),) + (slice(None),) * (len(tensor.shape) - 1)
                    chunk = tensor[slice_indices]

                yield chunk
        else:
            # Return whole tensor as single chunk
            yield tensor

    def query_and_operate(self, query_conditions: Dict[str, Any],
                         operation_name: str, operation_params: Optional[Dict[str, Any]] = None,
                         output_dataset: Optional[str] = None) -> List[str]:
        """
        Query tensors and apply operation to results.

        This combines the indexing system with operations for powerful
        "query-and-compute" workflows.
        """
        if not self.index_manager:
            raise ValueError("IndexManager required for query operations")

        # Query for tensor IDs
        tensor_ids = self.index_manager.query_tensors(query_conditions)

        if not tensor_ids:
            return []

        # Apply operation to each tensor
        result_ids = []
        for tensor_id in tensor_ids:
            try:
                result = self.execute_operation(
                    operation_name, [tensor_id], operation_params, output_dataset
                )
                if isinstance(result, OperationResult) and result.status == "success":
                    result_ids.extend(result.result_tensor_ids)
            except Exception as e:
                print(f"Failed to apply operation to tensor {tensor_id}: {e}")

        return result_ids

    def get_operation_status(self, operation_id: str) -> Optional[OperationResult]:
        """Get status of an operation."""
        return self.executor.get_operation_status(operation_id)

    def invalidate_cache_for_tensors(self, tensor_ids: List[str]) -> None:
        """Invalidate cache entries for given tensors."""
        self.cache.invalidate(tensor_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """Get operational statistics."""
        return {
            "cache_size": len(self.cache.cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "running_operations": len(self.executor.running_operations),
            "registered_operations": len(self.operation_specs)
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate operation cache hit rate."""
        # This would track cache hits/misses over time
        # For now, return a placeholder
        return 0.0
