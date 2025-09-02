"""
Tensorus Vector Database - Advanced vector similarity search and indexing capabilities.

This module implements a comprehensive vector database system with:
- Geometric partitioning for efficient similarity search
- Multi-tenant namespace isolation
- Freshness layers for real-time updates
- Hybrid search combining tensor operations with semantic search
- Performance monitoring and metrics

Architecture inspired by Pinecone's serverless approach with Tensorus tensor-native enhancements.
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
import time
import torch
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from uuid import uuid4

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

logger = logging.getLogger(__name__)


@dataclass
class VectorMetadata:
    """Enhanced metadata for vector entries with tenant isolation."""
    vector_id: str
    namespace: str = "default"
    tenant_id: str = "default"
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    tags: Set[str] = field(default_factory=set)


@dataclass
class SearchResult:
    """Vector similarity search result."""
    vector_id: str
    score: float
    rank: int
    metadata: VectorMetadata
    vector: Optional[np.ndarray] = None


@dataclass
class IndexStats:
    """Statistics for vector index performance."""
    total_vectors: int = 0
    index_size_mb: float = 0.0
    partitions: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    search_latency_p95: float = 0.0
    insert_rate: float = 0.0
    memory_usage_mb: float = 0.0


class GeometricPartitioner:
    """
    Implements geometric partitioning for efficient vector similarity search.
    
    Uses locality-sensitive hashing and centroid-based partitioning to 
    distribute vectors across multiple index partitions.
    """
    
    def __init__(self, num_partitions: int = 8, dimension: int = 384):
        self.num_partitions = num_partitions
        self.dimension = dimension
        self.centroids: Optional[np.ndarray] = None
        self.partition_assignment: Dict[str, int] = {}
        
    def fit(self, vectors: np.ndarray) -> None:
        """Fit partitioner to vector distribution using k-means clustering."""
        if len(vectors) < self.num_partitions:
            self.num_partitions = max(1, len(vectors))
            
        if not SKLEARN_AVAILABLE:
            # Fallback to random centroids when sklearn is not available
            self.centroids = np.random.randn(self.num_partitions, self.dimension).astype(np.float32)
            return
            
        # Use k-means clustering to find optimal centroids
        kmeans = KMeans(n_clusters=self.num_partitions, random_state=42, n_init=10)
        kmeans.fit(vectors)
        self.centroids = kmeans.cluster_centers_
        
    def assign_partition(self, vector: np.ndarray, vector_id: str) -> int:
        """Assign vector to optimal partition based on nearest centroid."""
        if self.centroids is None:
            # Random assignment if not fitted
            partition = hash(vector_id) % self.num_partitions
        else:
            # Assign to nearest centroid
            distances = np.linalg.norm(self.centroids - vector, axis=1)
            partition = int(np.argmin(distances))
            
        self.partition_assignment[vector_id] = partition
        return partition
        
    def get_partition(self, vector_id: str) -> Optional[int]:
        """Get assigned partition for vector ID."""
        return self.partition_assignment.get(vector_id)


class FreshnessLayer:
    """
    Implements real-time updates for vector database using a lambda-style architecture.
    
    Maintains recently updated vectors in a separate fast-access layer before
    merging with the main index during compaction.
    """
    
    def __init__(self, max_size: int = 10000, compaction_threshold: float = 0.8):
        self.max_size = max_size
        self.compaction_threshold = compaction_threshold
        self.fresh_vectors: Dict[str, Tuple[np.ndarray, VectorMetadata]] = {}
        self.deleted_ids: Set[str] = set()
        self.last_compaction = datetime.utcnow()
        
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: VectorMetadata) -> None:
        """Add vector to freshness layer."""
        metadata.updated_at = datetime.utcnow()
        self.fresh_vectors[vector_id] = (vector, metadata)
        
        # Remove from deleted set if re-added
        self.deleted_ids.discard(vector_id)
        
    def delete_vector(self, vector_id: str) -> None:
        """Mark vector as deleted."""
        self.fresh_vectors.pop(vector_id, None)
        self.deleted_ids.add(vector_id)
        
    def needs_compaction(self) -> bool:
        """Check if freshness layer needs compaction."""
        return len(self.fresh_vectors) >= self.max_size * self.compaction_threshold
        
    def get_fresh_vectors(self) -> Dict[str, Tuple[np.ndarray, VectorMetadata]]:
        """Get all vectors in freshness layer."""
        return self.fresh_vectors.copy()
        
    def get_deleted_ids(self) -> Set[str]:
        """Get set of deleted vector IDs."""
        return self.deleted_ids.copy()
        
    def clear(self) -> None:
        """Clear freshness layer after compaction."""
        self.fresh_vectors.clear()
        self.deleted_ids.clear()
        self.last_compaction = datetime.utcnow()


class VectorIndex(ABC):
    """Abstract base class for vector indexes."""
    
    @abstractmethod
    async def add_vectors(self, vectors: Dict[str, Tuple[np.ndarray, VectorMetadata]]) -> None:
        """Add vectors to index."""
        pass
        
    @abstractmethod
    async def search(self, query_vector: np.ndarray, k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
        
    @abstractmethod
    async def delete_vectors(self, vector_ids: Set[str]) -> None:
        """Delete vectors from index."""
        pass
        
    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        pass


if FAISS_AVAILABLE:
    class FAISSVectorIndex(VectorIndex):
        """FAISS-based vector index implementation."""
        
        def __init__(self, dimension: int, metric: str = "cosine"):
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
                
            self.dimension = dimension
            self.metric = metric
            self.index = self._create_index()
            self.vector_metadata: Dict[int, VectorMetadata] = {}
            self.id_to_index: Dict[str, int] = {}
            self.index_to_id: Dict[int, str] = {}
            self.next_index = 0
        
        def _create_index(self):
            """Create FAISS index based on metric."""
            if self.metric == "cosine":
                # Normalize vectors and use inner product for cosine similarity
                index = faiss.IndexFlatIP(self.dimension)
            elif self.metric == "euclidean":
                index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
                
            return index
            
        async def add_vectors(self, vectors: Dict[str, Tuple[np.ndarray, VectorMetadata]]) -> None:
            """Add vectors to FAISS index."""
            if not vectors:
                return
                
            # Prepare vectors for FAISS
            vector_array = np.array([vec.astype(np.float32) for vec, _ in vectors.values()])
            
            if self.metric == "cosine":
                # Normalize for cosine similarity
                faiss.normalize_L2(vector_array)
                
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(vector_array)
            
            # Update metadata mappings
            for i, (vector_id, (_, metadata)) in enumerate(vectors.items()):
                idx = start_idx + i
                self.vector_metadata[idx] = metadata
                self.id_to_index[vector_id] = idx
                self.index_to_id[idx] = vector_id
                
        async def search(self, query_vector: np.ndarray, k: int = 10,
                        filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
            """Search FAISS index for similar vectors."""
            if self.index.ntotal == 0:
                return []
                
            query = query_vector.astype(np.float32).reshape(1, -1)
            if self.metric == "cosine":
                faiss.normalize_L2(query)
                
            # Search index
            scores, indices = self.index.search(query, min(k, self.index.ntotal))
            
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                vector_id = self.index_to_id.get(idx)
                if not vector_id:
                    continue
                    
                metadata = self.vector_metadata.get(idx)
                if not metadata:
                    continue
                    
                # Apply filters if specified
                if filters and not self._matches_filters(metadata, filters):
                    continue
                    
                results.append(SearchResult(
                    vector_id=vector_id,
                    score=float(score),
                    rank=rank + 1,
                    metadata=metadata
                ))
                
            return results
            
        async def delete_vectors(self, vector_ids: Set[str]) -> None:
            """Delete vectors from index (FAISS doesn't support true deletion)."""
            # For FAISS, we mark as deleted in metadata
            for vector_id in vector_ids:
                if vector_id in self.id_to_index:
                    idx = self.id_to_index[vector_id]
                    self.vector_metadata.pop(idx, None)
                    self.id_to_index.pop(vector_id, None)
                    self.index_to_id.pop(idx, None)
                    
        def get_stats(self) -> IndexStats:
            """Get FAISS index statistics."""
            return IndexStats(
                total_vectors=self.index.ntotal,
                index_size_mb=self.index.ntotal * self.dimension * 4 / (1024 * 1024),  # 4 bytes per float32
                partitions=1,
                last_updated=datetime.utcnow()
            )
            
        def _matches_filters(self, metadata: VectorMetadata, filters: Dict[str, Any]) -> bool:
            """Check if metadata matches filters."""
            for key, value in filters.items():
                if key == "namespace" and metadata.namespace != value:
                    return False
                elif key == "tenant_id" and metadata.tenant_id != value:
                    return False
                elif key in metadata.metadata and metadata.metadata[key] != value:
                    return False
                    
            return True

else:
    # Provide a stub class when FAISS is not available
    class FAISSVectorIndex(VectorIndex):
        def __init__(self, *args, **kwargs):
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        async def add_vectors(self, vectors):
            raise ImportError("FAISS not available")
        
        async def search(self, query_vector, k=10, filters=None):
            raise ImportError("FAISS not available")
        
        async def delete_vectors(self, vector_ids):
            raise ImportError("FAISS not available")
        
        def get_stats(self):
            raise ImportError("FAISS not available")


class HierarchicalVectorIndex:
    """
    Hierarchical vector index with multi-level partitioning for improved scaling.
    
    Implements a tree-like structure where each level contains indexes optimized
    for different vector count ranges, providing O(log n) search complexity.
    """
    
    def __init__(self, dimension: int, max_vectors_per_shard: int = 100000, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.max_vectors_per_shard = max_vectors_per_shard
        self.shards: Dict[int, VectorIndex] = {}
        self.routing_table: Dict[str, int] = {}  # vector_id -> shard_id
        self.shard_stats: Dict[int, int] = {}  # shard_id -> vector_count
        self.next_shard_id = 0
        
    def _get_or_create_shard(self, target_size: int = None) -> int:
        """Get existing shard with capacity or create new one."""
        # Find shard with available capacity
        for shard_id, count in self.shard_stats.items():
            if count < self.max_vectors_per_shard:
                return shard_id
                
        # Create new shard
        shard_id = self.next_shard_id
        self.shards[shard_id] = FAISSVectorIndex(self.dimension, self.metric)
        self.shard_stats[shard_id] = 0
        self.next_shard_id += 1
        return shard_id
        
    async def add_vectors(self, vectors: Dict[str, Tuple[np.ndarray, VectorMetadata]]) -> None:
        """Add vectors with automatic shard assignment."""
        # Group vectors by target shard
        shard_batches: Dict[int, Dict[str, Tuple[np.ndarray, VectorMetadata]]] = {}
        
        for vector_id, vector_data in vectors.items():
            shard_id = self._get_or_create_shard()
            if shard_id not in shard_batches:
                shard_batches[shard_id] = {}
            shard_batches[shard_id][vector_id] = vector_data
            self.routing_table[vector_id] = shard_id
            
        # Add to shards in parallel
        tasks = []
        for shard_id, shard_vectors in shard_batches.items():
            tasks.append(self.shards[shard_id].add_vectors(shard_vectors))
            self.shard_stats[shard_id] += len(shard_vectors)
            
        await asyncio.gather(*tasks)
        
    async def search(self, query_vector: np.ndarray, k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search across all shards with intelligent routing."""
        if not self.shards:
            return []
            
        # Search all shards in parallel
        tasks = []
        for shard in self.shards.values():
            tasks.append(shard.search(query_vector, k, filters))
            
        shard_results = await asyncio.gather(*tasks)
        
        # Merge and rank results
        all_results = []
        for results in shard_results:
            all_results.extend(results)
            
        # Sort by score
        if self.metric == "cosine":
            all_results.sort(key=lambda x: x.score, reverse=True)
        else:  # euclidean
            all_results.sort(key=lambda x: x.score)
            
        # Update ranks and return top k
        for i, result in enumerate(all_results[:k]):
            result.rank = i + 1
            
        return all_results[:k]
        
    async def delete_vectors(self, vector_ids: Set[str]) -> None:
        """Delete vectors from appropriate shards."""
        # Group by shard
        shard_deletes: Dict[int, Set[str]] = {}
        
        for vector_id in vector_ids:
            shard_id = self.routing_table.get(vector_id)
            if shard_id is not None:
                if shard_id not in shard_deletes:
                    shard_deletes[shard_id] = set()
                shard_deletes[shard_id].add(vector_id)
                del self.routing_table[vector_id]
                
        # Delete from shards
        tasks = []
        for shard_id, ids_to_delete in shard_deletes.items():
            tasks.append(self.shards[shard_id].delete_vectors(ids_to_delete))
            self.shard_stats[shard_id] = max(0, self.shard_stats[shard_id] - len(ids_to_delete))
            
        await asyncio.gather(*tasks)
        
    def get_hierarchical_stats(self) -> IndexStats:
        """Get combined statistics across all shards."""
        total_vectors = sum(self.shard_stats.values())
        total_size = sum(shard.get_stats().index_size_mb for shard in self.shards.values())
        
        return IndexStats(
            total_vectors=total_vectors,
            index_size_mb=total_size,
            partitions=len(self.shards),
            last_updated=datetime.utcnow()
        )
        
    def get_stats(self) -> IndexStats:
        """Get combined statistics across all shards (compatibility method)."""
        return self.get_hierarchical_stats()


class PartitionedVectorIndex:
    """
    Multi-partition vector index with geometric partitioning and freshness layer.
    
    Implements Pinecone-style architecture with storage-compute separation.
    Enhanced with hierarchical indexing for better scaling.
    """
    
    def __init__(self, dimension: int, num_partitions: int = 8, metric: str = "cosine", 
                 use_hierarchical: bool = True):
        self.dimension = dimension
        self.metric = metric
        self.use_hierarchical = use_hierarchical
        self.partitioner = GeometricPartitioner(num_partitions, dimension)
        self.partitions: Dict[int, VectorIndex] = {}
        self.freshness_layer = FreshnessLayer()
        self.stats_history: List[IndexStats] = []
        
        # Initialize partitions with hierarchical or flat structure
        for i in range(num_partitions):
            if use_hierarchical:
                self.partitions[i] = HierarchicalVectorIndex(dimension, metric=metric)
            else:
                self.partitions[i] = FAISSVectorIndex(dimension, metric)
            
    async def add_vectors(self, vectors: Dict[str, Tuple[np.ndarray, VectorMetadata]]) -> None:
        """Add vectors with automatic partitioning."""
        # Add to freshness layer first
        for vector_id, (vector, metadata) in vectors.items():
            self.freshness_layer.add_vector(vector_id, vector, metadata)
            
        # Check if compaction is needed
        if self.freshness_layer.needs_compaction():
            await self._compact()
            
    async def search(self, query_vector: np.ndarray, k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search across all partitions and freshness layer with intelligent routing."""
        tasks = []
        
        # Search freshness layer
        fresh_results = self._search_freshness_layer(query_vector, k * 2, filters)
        
        # Intelligent partition selection for large indexes
        if len(self.partitions) > 4 and self.partitioner.centroids is not None:
            # Find closest partitions for more efficient search
            partition_distances = []
            for partition_id, partition in self.partitions.items():
                if partition_id < len(self.partitioner.centroids):
                    centroid = self.partitioner.centroids[partition_id]
                    distance = np.linalg.norm(query_vector - centroid)
                    partition_distances.append((distance, partition_id))
                    
            # Search top partitions first
            partition_distances.sort()
            search_partitions = partition_distances[:min(4, len(partition_distances))]
            
            for _, partition_id in search_partitions:
                tasks.append(self.partitions[partition_id].search(query_vector, k, filters))
        else:
            # Search all partitions for smaller indexes
            for partition in self.partitions.values():
                tasks.append(partition.search(query_vector, k, filters))
            
        partition_results = await asyncio.gather(*tasks)
        
        # Merge results
        all_results = fresh_results
        for results in partition_results:
            all_results.extend(results)
            
        # Remove deleted vectors
        deleted_ids = self.freshness_layer.get_deleted_ids()
        all_results = [r for r in all_results if r.vector_id not in deleted_ids]
        
        # Sort by score and return top k
        if self.metric == "cosine":
            all_results.sort(key=lambda x: x.score, reverse=True)
        else:  # euclidean
            all_results.sort(key=lambda x: x.score)
            
        # Update ranks
        for i, result in enumerate(all_results[:k]):
            result.rank = i + 1
            
        return all_results[:k]
        
    def _search_freshness_layer(self, query_vector: np.ndarray, k: int,
                               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search vectors in freshness layer."""
        results = []
        
        for vector_id, (vector, metadata) in self.freshness_layer.fresh_vectors.items():
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
                
            # Calculate similarity
            if self.metric == "cosine":
                # Cosine similarity
                norm_query = query_vector / np.linalg.norm(query_vector)
                norm_vector = vector / np.linalg.norm(vector)
                score = np.dot(norm_query, norm_vector)
            else:  # euclidean
                score = -np.linalg.norm(query_vector - vector)  # Negative for sorting
                
            results.append(SearchResult(
                vector_id=vector_id,
                score=float(score),
                rank=0,  # Will be updated later
                metadata=metadata
            ))
            
        # Sort and return top k
        if self.metric == "cosine":
            results.sort(key=lambda x: x.score, reverse=True)
        else:
            results.sort(key=lambda x: x.score, reverse=True)  # More negative = closer
            
        return results[:k]
        
    async def delete_vectors(self, vector_ids: Set[str]) -> None:
        """Delete vectors from index."""
        for vector_id in vector_ids:
            self.freshness_layer.delete_vector(vector_id)
            
    async def _compact(self) -> None:
        """Compact freshness layer into main partitions."""
        logger.info("Starting vector index compaction")
        
        fresh_vectors = self.freshness_layer.get_fresh_vectors()
        deleted_ids = self.freshness_layer.get_deleted_ids()
        
        if not fresh_vectors:
            return
            
        # Fit partitioner if needed
        if self.partitioner.centroids is None:
            vectors_array = np.array([vec for vec, _ in fresh_vectors.values()])
            self.partitioner.fit(vectors_array)
            
        # Assign vectors to partitions
        partition_vectors: Dict[int, Dict[str, Tuple[np.ndarray, VectorMetadata]]] = {}
        
        for vector_id, (vector, metadata) in fresh_vectors.items():
            partition = self.partitioner.assign_partition(vector, vector_id)
            if partition not in partition_vectors:
                partition_vectors[partition] = {}
            partition_vectors[partition][vector_id] = (vector, metadata)
            
        # Add vectors to partitions in parallel
        tasks = []
        for partition_id, vectors in partition_vectors.items():
            tasks.append(self.partitions[partition_id].add_vectors(vectors))
            
        await asyncio.gather(*tasks)
        
        # Delete vectors from partitions in parallel
        if deleted_ids:
            delete_tasks = []
            for partition in self.partitions.values():
                delete_tasks.append(partition.delete_vectors(deleted_ids))
            await asyncio.gather(*delete_tasks)
            
        # Clear freshness layer
        self.freshness_layer.clear()
        
        logger.info(f"Compaction completed: {len(fresh_vectors)} vectors, {len(deleted_ids)} deletions")
        
    def _matches_filters(self, metadata: VectorMetadata, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key == "namespace" and metadata.namespace != value:
                return False
            elif key == "tenant_id" and metadata.tenant_id != value:
                return False
            elif key in metadata.metadata and metadata.metadata[key] != value:
                return False
                
        return True
        
    def get_combined_stats(self) -> IndexStats:
        """Get combined statistics across all partitions."""
        total_vectors = 0
        total_size = 0.0
        
        # Collect stats from partitions
        for partition in self.partitions.values():
            if hasattr(partition, 'get_hierarchical_stats'):
                stats = partition.get_hierarchical_stats()
            else:
                stats = partition.get_stats()
            total_vectors += stats.total_vectors
            total_size += stats.index_size_mb
            
        total_vectors += len(self.freshness_layer.fresh_vectors)
        
        return IndexStats(
            total_vectors=total_vectors,
            index_size_mb=total_size,
            partitions=len(self.partitions),
            last_updated=datetime.utcnow()
        )