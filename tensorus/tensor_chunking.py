"""
Tensor Chunking and Compression Module

This module provides comprehensive tensor chunking and compression capabilities
for handling large tensors that exceed available memory limits.

Key Features:
- Automatic chunking for tensors above size thresholds
- Multiple compression algorithms (LZ4, ZSTD, gzip)
- Configurable chunk sizes and compression levels
- Memory usage monitoring and cache management
- Streaming operations for large tensors
- Backward compatibility with existing storage
"""

import os
import io
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import gzip
    GZIP_AVAILABLE = True
except ImportError:
    GZIP_AVAILABLE = False


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"


@dataclass
class ChunkMetadata:
    """Metadata for a single tensor chunk."""
    chunk_id: str
    tensor_id: str
    chunk_index: int
    total_chunks: int
    original_shape: Tuple[int, ...]
    chunk_shape: Tuple[int, ...]
    data_type: str
    compression: CompressionAlgorithm
    compressed_size: int
    uncompressed_size: int
    checksum: str
    chunk_range: Tuple[int, ...]  # (start_dim0, end_dim0, start_dim1, end_dim1, ...)


@dataclass
class TensorChunkingConfig:
    """Configuration for tensor chunking behavior."""
    max_tensor_size_mb: int = 100  # Tensors larger than this get chunked
    chunk_size_mb: int = 50        # Target chunk size
    default_compression: CompressionAlgorithm = CompressionAlgorithm.LZ4
    compression_level: int = 6     # Compression level (1-9, higher = better compression)
    max_memory_usage_mb: int = 512 # Maximum memory for chunk cache
    enable_streaming: bool = True  # Enable streaming operations
    num_worker_threads: int = 4    # Number of threads for parallel operations


class TensorChunkingError(Exception):
    """Base exception for tensor chunking operations."""
    pass


class ChunkNotFoundError(TensorChunkingError):
    """Raised when a requested chunk is not found."""
    pass


class CompressionError(TensorChunkingError):
    """Raised when compression/decompression fails."""
    pass


class TensorChunker:
    """
    Handles chunking and compression of large tensors.

    This class provides methods to:
    - Split large tensors into manageable chunks
    - Compress chunks using various algorithms
    - Reconstruct tensors from chunks
    - Stream chunks for memory-efficient processing
    """

    @staticmethod
    def _compression_available(algorithm: CompressionAlgorithm) -> bool:
        """Check if a compression algorithm is available."""
        if algorithm == CompressionAlgorithm.LZ4:
            return LZ4_AVAILABLE
        elif algorithm == CompressionAlgorithm.ZSTD:
            return ZSTD_AVAILABLE
        elif algorithm == CompressionAlgorithm.GZIP:
            return GZIP_AVAILABLE
        elif algorithm == CompressionAlgorithm.NONE:
            return True
        return False

    def __init__(self, config: Optional[TensorChunkingConfig] = None):
        self.config = config or TensorChunkingConfig()
        self._chunk_cache: Dict[str, bytes] = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_worker_threads)

        # Initialize compression contexts
        self._compression_contexts = {}
        if LZ4_AVAILABLE:
            self._compression_contexts[CompressionAlgorithm.LZ4] = lz4.LZ4FrameCompressor()
        if ZSTD_AVAILABLE:
            self._compression_contexts[CompressionAlgorithm.ZSTD] = zstd.ZstdCompressor(level=self.config.compression_level)
        if GZIP_AVAILABLE:
            self._compression_contexts[CompressionAlgorithm.GZIP] = None  # gzip uses simple interface

    def should_chunk_tensor(self, tensor: torch.Tensor) -> bool:
        """Determine if a tensor should be chunked based on size."""
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        return tensor_size_mb > self.config.max_tensor_size_mb

    def calculate_chunk_strategy(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype) -> Dict[str, Any]:
        """
        Calculate optimal chunking strategy for a tensor.

        Returns:
            Dict containing chunk dimensions, number of chunks, etc.
        """
        total_elements = np.prod(tensor_shape)
        element_size = torch.tensor(0, dtype=dtype).element_size()
        total_size_mb = total_elements * element_size / (1024 * 1024)

        if total_size_mb <= self.config.max_tensor_size_mb:
            return {
                "should_chunk": False,
                "total_chunks": 1,
                "chunk_shape": tensor_shape,
                "estimated_chunk_size_mb": total_size_mb
            }

        # Calculate optimal chunk dimensions
        target_elements_per_chunk = int(self.config.chunk_size_mb * 1024 * 1024 / element_size)

        # For simplicity, we'll chunk along the first dimension primarily
        # More sophisticated strategies could chunk along multiple dimensions
        first_dim_size = tensor_shape[0]
        elements_per_chunk_first_dim = min(first_dim_size, max(1, target_elements_per_chunk // np.prod(tensor_shape[1:])))

        chunks_in_first_dim = int(np.ceil(first_dim_size / elements_per_chunk_first_dim))
        actual_elements_per_chunk = elements_per_chunk_first_dim * np.prod(tensor_shape[1:])

        return {
            "should_chunk": True,
            "total_chunks": chunks_in_first_dim,
            "chunk_shape": (elements_per_chunk_first_dim,) + tensor_shape[1:],
            "chunks_in_first_dim": chunks_in_first_dim,
            "elements_per_chunk": actual_elements_per_chunk,
            "estimated_chunk_size_mb": actual_elements_per_chunk * element_size / (1024 * 1024)
        }

    def compress_chunk(self, chunk_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Compress chunk data using specified algorithm."""
        if algorithm == CompressionAlgorithm.NONE:
            return chunk_data

        try:
            if algorithm == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
                return lz4.compress(chunk_data, compression_level=self.config.compression_level)
            elif algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
                compressor = zstd.ZstdCompressor(level=self.config.compression_level)
                return compressor.compress(chunk_data)
            elif algorithm == CompressionAlgorithm.GZIP and GZIP_AVAILABLE:
                return gzip.compress(chunk_data, compresslevel=self.config.compression_level)
            else:
                raise CompressionError(f"Compression algorithm {algorithm.value} not available")
        except Exception as e:
            raise CompressionError(f"Compression failed: {e}")

    def decompress_chunk(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress chunk data using specified algorithm."""
        if algorithm == CompressionAlgorithm.NONE:
            return compressed_data

        try:
            if algorithm == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
                return lz4.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.GZIP and GZIP_AVAILABLE:
                return gzip.decompress(compressed_data)
            else:
                raise CompressionError(f"Decompression algorithm {algorithm.value} not available")
        except Exception as e:
            raise CompressionError(f"Decompression failed: {e}")

    def chunk_tensor(self, tensor: torch.Tensor, tensor_id: str) -> List[Tuple[bytes, ChunkMetadata]]:
        """
        Split a tensor into chunks and compress them.

        Returns:
            List of (compressed_chunk_data, metadata) tuples
        """
        strategy = self.calculate_chunk_strategy(tuple(tensor.shape), tensor.dtype)

        if not strategy["should_chunk"]:
            # Tensor is small enough, don't chunk
            tensor_bytes = self._tensor_to_bytes(tensor)
            compressed_data = self.compress_chunk(tensor_bytes, self.config.default_compression)

            metadata = ChunkMetadata(
                chunk_id=f"{tensor_id}_0",
                tensor_id=tensor_id,
                chunk_index=0,
                total_chunks=1,
                original_shape=tuple(tensor.shape),
                chunk_shape=tuple(tensor.shape),
                data_type=str(tensor.dtype),
                compression=self.config.default_compression,
                compressed_size=len(compressed_data),
                uncompressed_size=len(tensor_bytes),
                checksum=self._calculate_checksum(tensor_bytes),
                chunk_range=(0, tensor.shape[0]) + tuple(0 for _ in tensor.shape[1:]) * 2
            )

            return [(compressed_data, metadata)]

        # Chunk the tensor
        chunks = []
        chunk_shape = strategy["chunk_shape"]
        chunks_in_first_dim = strategy["chunks_in_first_dim"]

        for i in range(chunks_in_first_dim):
            start_idx = i * chunk_shape[0]
            end_idx = min((i + 1) * chunk_shape[0], tensor.shape[0])

            # Extract chunk
            if len(tensor.shape) == 1:
                chunk = tensor[start_idx:end_idx]
            else:
                slice_indices = (slice(start_idx, end_idx),) + (slice(None),) * (len(tensor.shape) - 1)
                chunk = tensor[slice_indices]

            # Convert to bytes
            chunk_bytes = self._tensor_to_bytes(chunk)

            # Compress
            compressed_data = self.compress_chunk(chunk_bytes, self.config.default_compression)

            # Create metadata
            chunk_range = (start_idx, end_idx) + tuple(0 for _ in tensor.shape[1:]) * 2
            if len(tensor.shape) > 1:
                chunk_range = chunk_range + tuple(s for _ in range(len(tensor.shape) - 1) for s in [0, tensor.shape[_ + 1]])

            metadata = ChunkMetadata(
                chunk_id=f"{tensor_id}_{i}",
                tensor_id=tensor_id,
                chunk_index=i,
                total_chunks=chunks_in_first_dim,
                original_shape=tuple(tensor.shape),
                chunk_shape=tuple(chunk.shape),
                data_type=str(tensor.dtype),
                compression=self.config.default_compression,
                compressed_size=len(compressed_data),
                uncompressed_size=len(chunk_bytes),
                checksum=self._calculate_checksum(chunk_bytes),
                chunk_range=chunk_range
            )

            chunks.append((compressed_data, metadata))

        return chunks

    def reconstruct_tensor(self, chunks: List[Tuple[bytes, ChunkMetadata]]) -> torch.Tensor:
        """
        Reconstruct a tensor from its chunks.

        Args:
            chunks: List of (compressed_data, metadata) tuples

        Returns:
            Reconstructed tensor
        """
        if not chunks:
            raise TensorChunkingError("No chunks provided")

        if len(chunks) == 1:
            # Single chunk (not actually chunked)
            compressed_data, metadata = chunks[0]
            decompressed_data = self.decompress_chunk(compressed_data, metadata.compression)
            return self._bytes_to_tensor(decompressed_data, metadata.chunk_shape, metadata.data_type)

        # Sort chunks by index
        chunks.sort(key=lambda x: x[1].chunk_index)

        # Reconstruct tensor
        first_chunk_data, first_metadata = chunks[0]
        decompressed_data = self.decompress_chunk(first_chunk_data, first_metadata.compression)
        first_chunk = self._bytes_to_tensor(decompressed_data, first_metadata.chunk_shape, first_metadata.data_type)

        # Create output tensor with original shape
        result_shape = first_metadata.original_shape
        result = torch.zeros(result_shape, dtype=first_chunk.dtype)

        # Place first chunk
        if len(result_shape) == 1:
            result[:first_chunk.shape[0]] = first_chunk
        else:
            slice_indices = (slice(0, first_chunk.shape[0]),) + (slice(None),) * (len(result_shape) - 1)
            result[slice_indices] = first_chunk

        # Place remaining chunks
        for compressed_data, metadata in chunks[1:]:
            decompressed_data = self.decompress_chunk(compressed_data, metadata.compression)
            chunk = self._bytes_to_tensor(decompressed_data, metadata.chunk_shape, metadata.data_type)

            start_idx = metadata.chunk_range[0]
            end_idx = metadata.chunk_range[1]

            if len(result_shape) == 1:
                result[start_idx:end_idx] = chunk
            else:
                slice_indices = (slice(start_idx, end_idx),) + (slice(None),) * (len(result_shape) - 1)
                result[slice_indices] = chunk

        return result

    def stream_tensor_chunks(self, tensor: torch.Tensor, tensor_id: str) -> Iterator[Tuple[bytes, ChunkMetadata]]:
        """
        Stream tensor chunks for memory-efficient processing.

        Yields:
            (compressed_chunk_data, metadata) tuples
        """
        for compressed_data, metadata in self.chunk_tensor(tensor, tensor_id):
            yield compressed_data, metadata

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes for storage/compression."""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    def _bytes_to_tensor(self, data: bytes, shape: Tuple[int, ...], dtype_str: str) -> torch.Tensor:
        """Convert bytes back to tensor."""
        buffer = io.BytesIO(data)
        tensor = torch.load(buffer)

        # Verify shape and dtype
        if tuple(tensor.shape) != shape:
            raise TensorChunkingError(f"Shape mismatch: expected {shape}, got {tuple(tensor.shape)}")

        return tensor

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity verification."""
        return hashlib.sha256(data).hexdigest()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        cache_size = sum(len(data) for data in self._chunk_cache.values())
        return {
            "cache_size_mb": cache_size / (1024 * 1024),
            "cached_chunks": len(self._chunk_cache),
            "max_memory_mb": self.config.max_memory_usage_mb,
            "memory_usage_percent": (cache_size / (self.config.max_memory_usage_mb * 1024 * 1024)) * 100
        }

    def clear_cache(self):
        """Clear the chunk cache to free memory."""
        with self._cache_lock:
            self._chunk_cache.clear()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
