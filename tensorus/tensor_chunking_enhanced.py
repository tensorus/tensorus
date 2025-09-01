"""
Enhanced Tensor Chunking with Metadata Tracking and Streaming Operations

This module extends the basic tensor chunking functionality with:
- Comprehensive metadata tracking for chunks
- Streaming operations for memory-efficient processing
- Memory usage monitoring and optimization
- Performance metrics and statistics
"""

import os
import time
import threading
import psutil
from typing import Dict, List, Optional, Tuple, Any, Iterator, Callable
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch

from .tensor_chunking import (
    TensorChunker, TensorChunkingConfig, ChunkMetadata,
    CompressionAlgorithm, TensorChunkingError
)


class ChunkMetadataTracker:
    """
    Tracks metadata for all chunked tensors in the system.

    Provides comprehensive statistics and monitoring capabilities.
    """

    def __init__(self):
        self.chunk_metadata: Dict[str, List[ChunkMetadata]] = {}
        self.tensor_stats: Dict[str, Dict[str, Any]] = {}
        self.access_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()

    def register_chunks(self, tensor_id: str, chunks: List[ChunkMetadata]):
        """Register chunk metadata for a tensor."""
        with self._lock:
            self.chunk_metadata[tensor_id] = chunks

            # Calculate tensor statistics
            total_size = sum(chunk.compressed_size for chunk in chunks)
            total_uncompressed = sum(chunk.uncompressed_size for chunk in chunks)
            compression_ratio = total_uncompressed / total_size if total_size > 0 else 1.0

            self.tensor_stats[tensor_id] = {
                "num_chunks": len(chunks),
                "total_compressed_size": total_size,
                "total_uncompressed_size": total_uncompressed,
                "compression_ratio": compression_ratio,
                "compression_algorithm": chunks[0].compression.value if chunks else "none",
                "original_shape": chunks[0].original_shape if chunks else (),
                "data_type": chunks[0].data_type if chunks else "unknown",
                "created_at": datetime.utcnow(),
                "last_accessed": datetime.utcnow(),
                "access_count": 0
            }

    def unregister_chunks(self, tensor_id: str):
        """Remove chunk metadata for a tensor."""
        with self._lock:
            if tensor_id in self.chunk_metadata:
                del self.chunk_metadata[tensor_id]
            if tensor_id in self.tensor_stats:
                del self.tensor_stats[tensor_id]
            if tensor_id in self.access_stats:
                del self.access_stats[tensor_id]

    def record_access(self, tensor_id: str, operation: str = "read"):
        """Record access to a chunked tensor."""
        with self._lock:
            if tensor_id in self.tensor_stats:
                self.tensor_stats[tensor_id]["last_accessed"] = datetime.utcnow()
                self.tensor_stats[tensor_id]["access_count"] += 1

                if tensor_id not in self.access_stats:
                    self.access_stats[tensor_id] = {"reads": 0, "writes": 0, "streams": 0}

                if operation in self.access_stats[tensor_id]:
                    self.access_stats[tensor_id][operation] += 1

    def get_tensor_stats(self, tensor_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific tensor."""
        with self._lock:
            return self.tensor_stats.get(tensor_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide chunking statistics."""
        with self._lock:
            total_tensors = len(self.chunk_metadata)
            total_chunks = sum(len(chunks) for chunks in self.chunk_metadata.values())
            total_compressed_size = sum(
                stats["total_compressed_size"] for stats in self.tensor_stats.values()
            )
            total_uncompressed_size = sum(
                stats["total_uncompressed_size"] for stats in self.tensor_stats.values()
            )

            compression_ratio = (
                total_uncompressed_size / total_compressed_size
                if total_compressed_size > 0 else 1.0
            )

            # Calculate access patterns
            total_accesses = sum(stats["access_count"] for stats in self.tensor_stats.values())

            return {
                "total_chunked_tensors": total_tensors,
                "total_chunks": total_chunks,
                "total_compressed_size_mb": total_compressed_size / (1024 * 1024),
                "total_uncompressed_size_mb": total_uncompressed_size / (1024 * 1024),
                "average_compression_ratio": compression_ratio,
                "total_accesses": total_accesses,
                "average_chunks_per_tensor": total_chunks / total_tensors if total_tensors > 0 else 0,
                "most_used_tensors": sorted(
                    [(tid, stats["access_count"]) for tid, stats in self.tensor_stats.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
            }


class StreamingTensorProcessor:
    """
    Handles streaming operations on large tensors using chunked processing.

    Enables memory-efficient processing of tensors that don't fit in RAM.
    """

    def __init__(self, chunker: TensorChunker, metadata_tracker: ChunkMetadataTracker):
        self.chunker = chunker
        self.metadata_tracker = metadata_tracker
        self._processing_stats = defaultdict(dict)

    def stream_map_operation(self, tensor: torch.Tensor, operation: Callable,
                           chunk_size_mb: Optional[int] = None) -> torch.Tensor:
        """
        Apply an operation to a tensor using streaming/chunked processing.

        Args:
            tensor: Input tensor
            operation: Function to apply to each chunk
            chunk_size_mb: Override default chunk size

        Returns:
            Result tensor with operation applied
        """
        start_time = time.time()

        if not self.chunker.should_chunk_tensor(tensor):
            # Small tensor, process normally
            result = operation(tensor)
            processing_time = time.time() - start_time
            self._processing_stats["small_tensor_operations"] = {
                "count": self._processing_stats.get("small_tensor_operations", {}).get("count", 0) + 1,
                "total_time": self._processing_stats.get("small_tensor_operations", {}).get("total_time", 0) + processing_time
            }
            return result

        # Process large tensor using chunking
        strategy = self.chunker.calculate_chunk_strategy(tuple(tensor.shape), tensor.dtype)

        if chunk_size_mb:
            strategy["chunk_shape"] = (int(chunk_size_mb * 1024 * 1024 / tensor.element_size()),) + tensor.shape[1:]
            strategy["chunks_in_first_dim"] = int(np.ceil(tensor.shape[0] / strategy["chunk_shape"][0]))

        results = []
        total_processed_elements = 0

        for i in range(strategy["chunks_in_first_dim"]):
            start_idx = i * strategy["chunk_shape"][0]
            end_idx = min((i + 1) * strategy["chunk_shape"][0], tensor.shape[0])

            # Extract chunk
            if len(tensor.shape) == 1:
                chunk = tensor[start_idx:end_idx]
            else:
                slice_indices = (slice(start_idx, end_idx),) + (slice(None),) * (len(tensor.shape) - 1)
                chunk = tensor[slice_indices]

            # Apply operation
            chunk_result = operation(chunk)
            results.append(chunk_result)
            total_processed_elements += chunk.numel()

        # Combine results
        if len(tensor.shape) == 1:
            result = torch.cat(results, dim=0)
        else:
            result = torch.cat(results, dim=0)

        processing_time = time.time() - start_time
        self._processing_stats["stream_operations"] = {
            "count": self._processing_stats.get("stream_operations", {}).get("count", 0) + 1,
            "total_time": self._processing_stats.get("stream_operations", {}).get("total_time", 0) + processing_time,
            "total_elements_processed": self._processing_stats.get("stream_operations", {}).get("total_elements_processed", 0) + total_processed_elements
        }

        return result

    def stream_reduce_operation(self, tensor: torch.Tensor, operation: Callable,
                              initial_value: Any = None) -> Any:
        """
        Apply a reduction operation using streaming processing.

        Args:
            tensor: Input tensor
            operation: Reduction function (should take accumulator and chunk)
            initial_value: Initial value for reduction

        Returns:
            Reduced result
        """
        start_time = time.time()

        if not self.chunker.should_chunk_tensor(tensor):
            result = operation(initial_value, tensor)
            processing_time = time.time() - start_time
            self._processing_stats["small_tensor_reductions"] = {
                "count": self._processing_stats.get("small_tensor_reductions", {}).get("count", 0) + 1,
                "total_time": self._processing_stats.get("small_tensor_reductions", {}).get("total_time", 0) + processing_time
            }
            return result

        # Process large tensor using streaming reduction
        strategy = self.chunker.calculate_chunk_strategy(tuple(tensor.shape), tensor.dtype)

        accumulator = initial_value
        total_processed_elements = 0

        for i in range(strategy["chunks_in_first_dim"]):
            start_idx = i * strategy["chunk_shape"][0]
            end_idx = min((i + 1) * strategy["chunk_shape"][0], tensor.shape[0])

            # Extract chunk
            if len(tensor.shape) == 1:
                chunk = tensor[start_idx:end_idx]
            else:
                slice_indices = (slice(start_idx, end_idx),) + (slice(None),) * (len(tensor.shape) - 1)
                chunk = tensor[slice_indices]

            # Apply reduction
            accumulator = operation(accumulator, chunk)
            total_processed_elements += chunk.numel()

        processing_time = time.time() - start_time
        self._processing_stats["stream_reductions"] = {
            "count": self._processing_stats.get("stream_reductions", {}).get("count", 0) + 1,
            "total_time": self._processing_stats.get("stream_reductions", {}).get("total_time", 0) + processing_time,
            "total_elements_processed": self._processing_stats.get("stream_reductions", {}).get("total_elements_processed", 0) + total_processed_elements
        }

        return accumulator

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return dict(self._processing_stats)


class MemoryMonitor:
    """
    Monitors memory usage for tensor chunking operations.

    Provides insights into memory consumption patterns and optimization opportunities.
    """

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.memory_history = []
        self.chunk_cache_size = 0
        self._lock = threading.Lock()

    def record_memory_usage(self, operation: str, memory_mb: float):
        """Record memory usage for an operation."""
        with self._lock:
            self.memory_history.append({
                "timestamp": datetime.utcnow(),
                "operation": operation,
                "memory_mb": memory_mb,
                "cache_size_mb": self.chunk_cache_size
            })

            # Keep only recent history
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-1000:]

    def update_cache_size(self, cache_size_mb: float):
        """Update the current chunk cache size."""
        with self._lock:
            self.chunk_cache_size = cache_size_mb

    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "process_memory_percent": process.memory_percent(),
            "chunk_cache_mb": self.chunk_cache_size,
            "max_memory_mb": self.max_memory_mb,
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
            "memory_pressure": "high" if self.chunk_cache_size > self.max_memory_mb * 0.8 else "normal"
        }

    def should_reduce_cache(self) -> bool:
        """Check if cache should be reduced to free memory."""
        current_usage = self.get_current_memory_usage()
        return (
            self.chunk_cache_size > self.max_memory_mb * 0.8 or
            current_usage["process_memory_percent"] > 80
        )

    def get_memory_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memory usage history."""
        with self._lock:
            return self.memory_history[-limit:]


class EnhancedTensorChunker:
    """
    Enhanced tensor chunker with comprehensive metadata tracking,
    streaming operations, and memory monitoring.
    """

    def __init__(self, config: Optional[TensorChunkingConfig] = None):
        self.config = config or TensorChunkingConfig()
        self.chunker = TensorChunker(self.config)
        self.metadata_tracker = ChunkMetadataTracker()
        self.stream_processor = StreamingTensorProcessor(self.chunker, self.metadata_tracker)
        self.memory_monitor = MemoryMonitor(self.config.max_memory_usage_mb)

        # Performance metrics
        self.performance_stats = {
            "chunking_operations": 0,
            "reconstruction_operations": 0,
            "streaming_operations": 0,
            "total_chunking_time": 0,
            "total_reconstruction_time": 0,
            "total_streaming_time": 0
        }

    def chunk_tensor_with_tracking(self, tensor: torch.Tensor, tensor_id: str) -> List[Tuple[bytes, ChunkMetadata]]:
        """
        Chunk a tensor and track comprehensive metadata.

        Returns:
            List of (compressed_chunk_data, metadata) tuples
        """
        start_time = time.time()

        # Record memory usage before chunking
        memory_before = self.memory_monitor.get_current_memory_usage()["process_memory_mb"]

        # Perform chunking
        chunks = self.chunker.chunk_tensor(tensor, tensor_id)

        # Track metadata
        chunk_metadata_list = [metadata for _, metadata in chunks]
        self.metadata_tracker.register_chunks(tensor_id, chunk_metadata_list)

        # Record performance metrics
        chunking_time = time.time() - start_time
        self.performance_stats["chunking_operations"] += 1
        self.performance_stats["total_chunking_time"] += chunking_time

        # Record memory usage after chunking
        memory_after = self.memory_monitor.get_current_memory_usage()["process_memory_mb"]
        self.memory_monitor.record_memory_usage("chunking", memory_after - memory_before)

        return chunks

    def reconstruct_tensor_with_tracking(self, chunks: List[Tuple[bytes, ChunkMetadata]]) -> torch.Tensor:
        """
        Reconstruct a tensor with performance tracking.

        Args:
            chunks: List of (compressed_data, metadata) tuples

        Returns:
            Reconstructed tensor
        """
        start_time = time.time()
        tensor_id = chunks[0][1].tensor_id if chunks else "unknown"

        # Record memory usage before reconstruction
        memory_before = self.memory_monitor.get_current_memory_usage()["process_memory_mb"]

        # Perform reconstruction
        tensor = self.chunker.reconstruct_tensor(chunks)

        # Record access
        self.metadata_tracker.record_access(tensor_id, "read")

        # Record performance metrics
        reconstruction_time = time.time() - start_time
        self.performance_stats["reconstruction_operations"] += 1
        self.performance_stats["total_reconstruction_time"] += reconstruction_time

        # Record memory usage after reconstruction
        memory_after = self.memory_monitor.get_current_memory_usage()["process_memory_mb"]
        self.memory_monitor.record_memory_usage("reconstruction", memory_after - memory_before)

        return tensor

    def stream_operation_with_tracking(self, tensor: torch.Tensor, operation: Callable,
                                     operation_type: str = "map") -> Any:
        """
        Perform streaming operation with comprehensive tracking.

        Args:
            tensor: Input tensor
            operation: Operation to perform
            operation_type: Type of operation ("map" or "reduce")

        Returns:
            Operation result
        """
        start_time = time.time()
        tensor_id = f"streaming_{int(time.time())}"

        # Record memory usage before operation
        memory_before = self.memory_monitor.get_current_memory_usage()["process_memory_mb"]

        # Perform streaming operation
        if operation_type == "reduce":
            result = self.stream_processor.stream_reduce_operation(tensor, operation)
        else:
            result = self.stream_processor.stream_map_operation(tensor, operation)

        # Record performance metrics
        streaming_time = time.time() - start_time
        self.performance_stats["streaming_operations"] += 1
        self.performance_stats["total_streaming_time"] += streaming_time

        # Record memory usage after operation
        memory_after = self.memory_monitor.get_current_memory_usage()["process_memory_mb"]
        self.memory_monitor.record_memory_usage(f"streaming_{operation_type}", memory_after - memory_before)

        return result

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the chunking system."""
        return {
            "metadata_stats": self.metadata_tracker.get_system_stats(),
            "processing_stats": self.stream_processor.get_processing_stats(),
            "memory_stats": self.memory_monitor.get_current_memory_usage(),
            "performance_stats": self.performance_stats.copy(),
            "chunker_stats": self.chunker.get_memory_usage(),
            "config": {
                "max_tensor_size_mb": self.config.max_tensor_size_mb,
                "chunk_size_mb": self.config.chunk_size_mb,
                "default_compression": self.config.default_compression.value,
                "max_memory_usage_mb": self.config.max_memory_usage_mb,
                "enable_streaming": self.config.enable_streaming
            }
        }

    def optimize_for_tensor(self, tensor: torch.Tensor) -> TensorChunkingConfig:
        """
        Get optimized configuration for a specific tensor.

        Args:
            tensor: The tensor to optimize for

        Returns:
            Optimized TensorChunkingConfig
        """
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)

        # Create optimized config
        optimized_config = TensorChunkingConfig()

        if tensor_size_mb > 1000:  # Very large tensors
            optimized_config.chunk_size_mb = 200
            optimized_config.max_memory_usage_mb = 2048
            optimized_config.compression_level = 9  # Maximum compression
            optimized_config.default_compression = CompressionAlgorithm.ZSTD
        elif tensor_size_mb > 500:
            optimized_config.chunk_size_mb = 100
            optimized_config.max_memory_usage_mb = 1024
            optimized_config.compression_level = 6
        elif tensor_size_mb > 100:
            optimized_config.chunk_size_mb = 150
            optimized_config.max_memory_usage_mb = 512
            optimized_config.compression_level = 4

        return optimized_config

    def cleanup_resources(self):
        """Clean up resources and caches."""
        self.chunker.clear_cache()
        self.memory_monitor.memory_history.clear()

        # Clean up old metadata (older than 30 days)
        cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - 30)
        to_remove = []

        for tensor_id, stats in self.metadata_tracker.tensor_stats.items():
            if stats["last_accessed"] < cutoff_date and stats["access_count"] < 5:
                to_remove.append(tensor_id)

        for tensor_id in to_remove:
            self.metadata_tracker.unregister_chunks(tensor_id)
