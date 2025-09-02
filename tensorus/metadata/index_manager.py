"""
IndexManager: Central coordinator for all indexing operations in Tensorus.

This module provides:
- Centralized index management and coordination
- Query optimization with automatic index selection
- Index maintenance and statistics
- Integration with metadata storage
- Performance monitoring and caching
"""

import time
import threading
import pickle
from typing import Dict, List, Set, Optional, Any, Tuple, Iterator, Union
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import heapq

from .indexing import (
    BaseIndex, HashIndex, BTreeIndex, SpatialIndex, CompositeIndex,
    IndexType, IndexStructure, IndexMetadata, QueryPlan
)
from .storage_abc import MetadataStorage


@dataclass
class IndexManagerConfig:
    """Configuration for IndexManager."""
    max_cache_size: int = 1000  # Maximum cached query results
    index_update_batch_size: int = 100  # Batch size for bulk index updates
    enable_query_caching: bool = True  # Cache frequent queries
    enable_statistics: bool = True  # Collect performance statistics
    auto_create_indexes: bool = True  # Auto-create indexes for frequent queries
    index_memory_limit_mb: int = 500  # Memory limit for indexes


@dataclass
class CachedQuery:
    """Represents a cached query result."""
    query_hash: str
    conditions: Dict[str, Any]
    results: Set[str]
    timestamp: float
    access_count: int = 1
    last_accessed: float = field(default_factory=time.time)


class IndexManager:
    """
    Central manager for all indexing operations.

    Provides:
    - Index creation, maintenance, and optimization
    - Query planning and execution
    - Performance monitoring and caching
    - Integration with metadata storage
    """

    def __init__(self, config: Optional[IndexManagerConfig] = None):
        self.config = config or IndexManagerConfig()
        self.indexes: Dict[str, BaseIndex] = {}
        self.query_cache: OrderedDict[str, CachedQuery] = OrderedDict()
        self.statistics = defaultdict(int)
        self._lock = threading.RLock()
        self.metadata_storage: Optional[MetadataStorage] = None

        # Initialize default indexes
        if self.config.auto_create_indexes: # Add this check
            self._create_default_indexes()

    def set_metadata_storage(self, storage: MetadataStorage) -> None:
        """Set the metadata storage backend."""
        self.metadata_storage = storage

    def _create_default_indexes(self) -> None:
        """Create default indexes for common query patterns."""
        with self._lock:
            # Create basic indexes for common fields
            self.indexes["tensor_id"] = HashIndex("tensor_id", ["tensor_id"])
            self.indexes["owner"] = HashIndex("owner", ["owner"])
            self.indexes["data_type"] = HashIndex("data_type", ["data_type"])
            self.indexes["created_at"] = BTreeIndex("created_at", ["created_at"])
            self.indexes["size"] = BTreeIndex("size", ["size"])
            self.indexes["category"] = HashIndex("category", ["category"])  # Add index for category field

            # Create composite indexes for common query patterns
            self.indexes["owner_data_type"] = CompositeIndex("owner_data_type", ["owner", "data_type"])
            self.indexes["data_type_shape"] = CompositeIndex("data_type_shape", ["data_type", "shape"])

            # Spatial index for shape queries
            self.indexes["shape"] = SpatialIndex("shape")

            # Composite indexes for common combinations
            self.indexes["owner_data_type"] = CompositeIndex("owner_data_type", ["owner", "data_type"])
            self.indexes["data_type_shape"] = CompositeIndex("data_type_shape", ["data_type", "shape"])

    def create_index(self, name: str, index_type: Union[IndexType, str],
                    indexed_properties: List[str],
                    structure: Union[IndexStructure, str] = IndexStructure.HASH_MAP) -> bool:
        """
        Create a new index.

        Args:
            name: Unique name for the index
            index_type: Type of index to create
            indexed_properties: Properties to index
            structure: Underlying data structure

        Returns:
            True if index was created successfully
        """
        with self._lock:
            if name in self.indexes:
                return False

            # Convert string to IndexType enum if necessary
            if isinstance(index_type, str):
                try:
                    index_type = IndexType(index_type)
                except ValueError:
                    return False

            # Convert string to IndexStructure enum if necessary
            if isinstance(structure, str):
                try:
                    structure = IndexStructure(structure)
                except ValueError:
                    return False

            try:
                if index_type == IndexType.PROPERTY:
                    if structure == IndexStructure.HASH_MAP:
                        index = HashIndex(name, indexed_properties)
                    elif structure == IndexStructure.B_TREE:
                        index = BTreeIndex(name, indexed_properties)
                    else:
                        return False
                elif index_type == IndexType.COMPOSITE:
                    index = CompositeIndex(name, indexed_properties)
                elif index_type == IndexType.SPATIAL:
                    index = SpatialIndex(name)
                elif index_type == IndexType.RANGE:
                    index = BTreeIndex(name, indexed_properties)
                elif index_type == IndexType.TEXT:
                    return False
                elif index_type == IndexType.PRIMARY:
                    return False
                else:
                    return False

                self.indexes[name] = index
                self.statistics["indexes_created"] += 1
                return True

            except Exception as e:
                return False

    def drop_index(self, name: str) -> bool:
        """Drop an index."""
        with self._lock:
            if name not in self.indexes:
                return False

            del self.indexes[name]
            self.statistics["indexes_dropped"] += 1
            return True

    def add_tensor(self, tensor_id: str, tensor_data: Dict[str, Any]) -> None:
        """Add a tensor to all relevant indexes."""
        with self._lock:
            for index in self.indexes.values():
                try:
                    index.insert(tensor_id, tensor_data)
                except Exception:
                    # Log error but continue with other indexes
                    pass

            self.statistics["tensors_indexed"] += 1

    def remove_tensor(self, tensor_id: str) -> None:
        """Remove a tensor from all indexes."""
        with self._lock:
            for index in self.indexes.values():
                try:
                    index.remove(tensor_id)
                except Exception:
                    pass

            self.statistics["tensors_removed"] += 1

    def update_tensor(self, tensor_id: str, old_data: Dict[str, Any],
                     new_data: Dict[str, Any]) -> None:
        """Update tensor data in all indexes."""
        with self._lock:
            for index in self.indexes.values():
                try:
                    index.update(tensor_id, old_data, new_data)
                except Exception:
                    pass

            self.statistics["tensors_updated"] += 1

    def query_tensors(self, conditions: Dict[str, Any],
                     limit: Optional[int] = None,
                     sort_by: Optional[str] = None) -> List[str]:
        """
        Query tensors using optimized index selection.

        Args:
            conditions: Query conditions
            limit: Maximum number of results
            sort_by: Property to sort results by

        Returns:
            List of tensor IDs matching the query
        """
        start_time = time.time()

        # Check query cache first
        query_hash = self._hash_query(conditions, limit, sort_by)
        if self.config.enable_query_caching and query_hash in self.query_cache:
            cached_result = self.query_cache[query_hash]
            cached_result.access_count += 1
            cached_result.last_accessed = time.time()

            # Move to end (most recently used)
            self.query_cache.move_to_end(query_hash)
            self.statistics["cache_hits"] += 1

            print(f"[DEBUG] Query Cache Hit for {conditions}: {len(cached_result.results)} results")
            return list(cached_result.results)[:limit] if limit else list(cached_result.results)

        # Generate query plan
        query_plan = self._generate_query_plan(conditions)
        print(f"[DEBUG] Query Plan for {conditions}: {query_plan.index_name if query_plan else 'No Plan'}")

        if not query_plan:
            # Fallback to full scan if no suitable indexes
            results = self._full_scan_query(conditions)
            print(f"[DEBUG] Full Scan Fallback for {conditions}: {len(results)} results")
        else:
            # Execute query using selected index
            index = self.indexes[query_plan.index_name]
            results = index.search(conditions)

        # Apply sorting if requested
        if sort_by and self.metadata_storage:
            results = self._sort_results(list(results), sort_by)

        # Apply limit
        if limit:
            results = list(results)[:limit]

        # Cache the result
        if self.config.enable_query_caching:
            self._cache_query_result(query_hash, conditions, set(results))

        # Update statistics
        query_time = time.time() - start_time
        self.statistics["queries_executed"] += 1
        self.statistics["total_query_time"] += query_time

        return results

    def _generate_query_plan(self, conditions: Dict[str, Any]) -> Optional[QueryPlan]:
        """
        Generate an optimized query execution plan.

        Analyzes available indexes and selects the most efficient one
        for the given query conditions.
        """
        best_plan = None
        best_cost = float('inf')

        for index_name, index in self.indexes.items():
            plan = self._evaluate_index_for_query(index, conditions)
            if plan and plan.estimated_cost < best_cost:
                best_plan = plan
                best_cost = plan.estimated_cost

        return best_plan

    def _evaluate_index_for_query(self, index: BaseIndex,
                                 conditions: Dict[str, Any]) -> Optional[QueryPlan]:
        """
        Evaluate how well an index can handle a query.

        Returns a QueryPlan if the index is suitable, None otherwise.
        """
        # Check if index can handle the query conditions
        if not self._index_supports_conditions(index, conditions):
            return None

        # Estimate result count and cost
        estimated_results = self._estimate_result_count(index, conditions)
        estimated_cost = self._calculate_query_cost(index, conditions, estimated_results)

        return QueryPlan(
            index_name=index.name,
            index_type=index.index_type,
            estimated_cost=estimated_cost,
            estimated_results=estimated_results,
            filter_conditions=[conditions]
        )

    def _index_supports_conditions(self, index: BaseIndex,
                                  conditions: Dict[str, Any]) -> bool:
        """Check if an index can support the given query conditions."""
        # Check if all required properties are indexed
        required_props = set(conditions.keys())
        indexed_props = set(index.indexed_properties)

        # For composite indexes, all properties must be covered
        if index.index_type == IndexType.COMPOSITE:
            return required_props == indexed_props

        # For property indexes, at least one property should be covered
        if index.index_type == IndexType.PROPERTY:
            return bool(required_props.intersection(indexed_props))

        # For spatial indexes, check spatial query types
        if index.index_type == IndexType.SPATIAL:
            return any(key in conditions for key in ["shape", "dimensionality", "min_size", "max_size"])

        # For range indexes, check for range queries
        if index.index_type == IndexType.RANGE:
            return any(key in conditions for key in ["min", "max"])

        return False

    def _estimate_result_count(self, index: BaseIndex,
                              conditions: Dict[str, Any]) -> int:
        """Estimate the number of results for a query."""
        stats = index.get_statistics()

        # Simple estimation based on index statistics
        if index.index_type == IndexType.PROPERTY:
            avg_bucket_size = stats.get("avg_bucket_size", 1)
            return int(avg_bucket_size)
        elif index.index_type == IndexType.COMPOSITE:
            avg_tensors_per_key = stats.get("avg_tensors_per_key", 1)
            return int(avg_tensors_per_key)
        else:
            return max(1, index.metadata.tensor_count // 10)  # Conservative estimate

    def _calculate_query_cost(self, index: BaseIndex, conditions: Dict[str, Any],
                             estimated_results: int) -> float:
        """Calculate the estimated cost of executing a query with an index."""
        base_cost = 1.0

        # Index type affects cost
        if index.index_type == IndexType.PRIMARY:
            base_cost = 0.1  # Very fast
        elif index.index_type == IndexType.PROPERTY:
            base_cost = 0.5
        elif index.index_type == IndexType.COMPOSITE:
            base_cost = 1.0
        elif index.index_type == IndexType.RANGE:
            base_cost = 2.0  # Range queries are more expensive
        elif index.index_type == IndexType.SPATIAL:
            base_cost = 3.0  # Spatial queries are complex

        # Factor in result count
        result_factor = min(estimated_results / 100.0, 10.0)

        # Factor in index efficiency
        stats = index.get_statistics()
        efficiency = stats.get("index_efficiency", 1.0)

        # Factor in query specificity
        specificity_factor = 1.0
        # Ensure required_props is defined for composite index check
        required_props = set(conditions.keys())
        indexed_props = set(index.indexed_properties)

        if index.index_type == IndexType.PROPERTY and len(conditions) == 1 and list(conditions.keys())[0] in index.indexed_properties:
            specificity_factor = 0.1 # Highly specific, lower cost
        elif index.index_type == IndexType.COMPOSITE and required_props == indexed_props: # Exact match for all composite properties
            specificity_factor = 0.2 # Exact composite match, lower cost
        elif index.index_type == IndexType.COMPOSITE and required_props.issubset(indexed_props):
            specificity_factor = 0.5 # Partial composite match, medium cost

        return base_cost * result_factor / efficiency * specificity_factor

    def _full_scan_query(self, conditions: Dict[str, Any]) -> Set[str]:
        """Fallback query method when no suitable index is available."""
        if not self.metadata_storage:
            return set()

        self.statistics["full_scans"] += 1
        all_tensor_descriptors = self.metadata_storage.list_tensor_descriptors()
        matching_tensor_ids = set()
        for descriptor in all_tensor_descriptors:
            match = True
            for key, value in conditions.items():
                if not hasattr(descriptor, key) or getattr(descriptor, key) != value:
                    match = False
                    break
            if match:
                matching_tensor_ids.add(str(descriptor.tensor_id))
        return matching_tensor_ids

    def _sort_results(self, tensor_ids: List[str], sort_by: str) -> List[str]:
        """Sort tensor results by a property."""
        if not self.metadata_storage:
            return tensor_ids

        # Get tensor data for sorting
        tensor_data = []
        for tensor_id in tensor_ids:
            try:
                # This would need to be implemented based on the storage backend
                # For now, return unsorted
                pass
            except Exception:
                pass

        return tensor_ids  # Return unsorted for now

    def _hash_query(self, conditions: Dict[str, Any], limit: Optional[int],
                   sort_by: Optional[str]) -> str:
        """Generate a hash for query caching."""
        query_str = str(sorted(conditions.items()))
        if limit:
            query_str += f"_limit_{limit}"
        if sort_by:
            query_str += f"_sort_{sort_by}"

        return str(hash(query_str))

    def _cache_query_result(self, query_hash: str, conditions: Dict[str, Any],
                           results: Set[str]) -> None:
        """Cache a query result."""
        cached_query = CachedQuery(
            query_hash=query_hash,
            conditions=conditions,
            results=results,
            timestamp=time.time()
        )

        self.query_cache[query_hash] = cached_query

        # Maintain cache size limit
        while len(self.query_cache) > self.config.max_cache_size:
            # Remove least recently used
            self.query_cache.popitem(last=False)

        self.statistics["cache_misses"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive indexing statistics."""
        with self._lock:
            index_stats = {}
            for name, index in self.indexes.items():
                index_stats[name] = index.get_statistics()

            return {
                "total_indexes": len(self.indexes),
                "index_types": {name: index.index_type.value for name, index in self.indexes.items()},
                "cache_size": len(self.query_cache),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "performance_stats": dict(self.statistics),
                "index_details": index_stats
            }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate query cache hit rate."""
        total_requests = self.statistics.get("cache_hits", 0) + self.statistics.get("cache_misses", 0)
        if total_requests == 0:
            return 0.0
        return self.statistics.get("cache_hits", 0) / total_requests

    def optimize_indexes(self) -> Dict[str, Any]:
        """
        Analyze and optimize index usage patterns.

        Returns recommendations for index creation/removal.
        """
        recommendations = {
            "indexes_to_create": [],
            "indexes_to_drop": [],
            "performance_improvements": []
        }

        # Analyze query patterns and suggest new indexes
        # This would analyze the query cache and statistics to identify
        # frequently queried property combinations

        return recommendations

    def rebuild_index(self, index_name: str) -> bool:
        """Rebuild an index from scratch."""
        with self._lock:
            if index_name not in self.indexes:
                return False

            index = self.indexes[index_name]
            index.clear()

            # Rebuild from metadata storage
            if self.metadata_storage:
                # This would iterate through all tensors and rebuild the index
                # Implementation depends on the storage backend
                pass

            return True

    def clear_cache(self) -> None:
        """Clear the query result cache."""
        with self._lock:
            self.query_cache.clear()
            self.statistics["cache_cleared"] = time.time()

    def get_index_info(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about indexes."""
        with self._lock:
            if index_name:
                if index_name not in self.indexes:
                    return {}
                return {
                    "name": index_name,
                    "type": self.indexes[index_name].index_type.value,
                    "structure": self.indexes[index_name].structure.value,
                    "properties": self.indexes[index_name].indexed_properties,
                    "statistics": self.indexes[index_name].get_statistics(),
                    "metadata": self.indexes[index_name].metadata
                }
            else:
                return {
                    name: {
                        "type": index.index_type.value,
                        "properties": index.indexed_properties,
                        "tensor_count": index.metadata.tensor_count
                    }
                    for name, index in self.indexes.items()
                }
