"""
Tensorus Indexing System

Provides efficient indexing capabilities for tensor storage, addressing
GAP 4: No Efficient Indexing. Supports multiple index types for fast
lookups and queries.
"""

import torch
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import defaultdict
import threading
import bisect
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class IndexingError(Exception):
    """Raised when indexing operations fail."""
    pass

# === Index Interface ===

class Index(ABC):
    """Abstract base class for all index types."""
    
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
    
    @abstractmethod
    def insert(self, record_id: str, value: Any, tensor: Optional[torch.Tensor] = None) -> None:
        """Insert a record into the index."""
        pass
    
    @abstractmethod
    def delete(self, record_id: str, value: Any = None) -> None:
        """Delete a record from the index."""
        pass
    
    @abstractmethod
    def lookup(self, value: Any) -> List[str]:
        """Lookup records by exact value."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the index."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return the number of unique values in the index."""
        pass

# === Basic Indexes ===

class HashIndex(Index):
    """Hash-based index for exact lookups - O(1) average case."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._index: Dict[Any, Set[str]] = defaultdict(set)
    
    def insert(self, record_id: str, value: Any, tensor: Optional[torch.Tensor] = None) -> None:
        """Insert record into hash index."""
        with self._lock:
            # Handle hashable types
            if isinstance(value, (list, dict, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    # Use tensor hash for tensor values
                    key = self._tensor_hash(value)
                elif isinstance(value, list):
                    key = tuple(value) if all(isinstance(x, (int, float, str, bool)) for x in value) else str(value)
                else:
                    key = json.dumps(value, sort_keys=True)
            else:
                key = value
            
            self._index[key].add(record_id)
    
    def delete(self, record_id: str, value: Any = None) -> None:
        """Delete record from hash index."""
        with self._lock:
            if value is not None:
                # Delete specific value
                key = self._make_key(value)
                if key in self._index:
                    self._index[key].discard(record_id)
                    if not self._index[key]:
                        del self._index[key]
            else:
                # Delete record_id from all values (slower)
                keys_to_remove = []
                for key, record_set in self._index.items():
                    record_set.discard(record_id)
                    if not record_set:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del self._index[key]
    
    def lookup(self, value: Any) -> List[str]:
        """Lookup records by exact value."""
        with self._lock:
            key = self._make_key(value)
            return list(self._index.get(key, set()))
    
    def clear(self) -> None:
        """Clear the index."""
        with self._lock:
            self._index.clear()
    
    def size(self) -> int:
        """Return number of unique values."""
        with self._lock:
            return len(self._index)
    
    def _make_key(self, value: Any) -> Any:
        """Convert value to hashable key."""
        if isinstance(value, (list, dict, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                return self._tensor_hash(value)
            elif isinstance(value, list):
                return tuple(value) if all(isinstance(x, (int, float, str, bool)) for x in value) else str(value)
            else:
                return json.dumps(value, sort_keys=True)
        return value
    
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """Create hash of tensor for indexing."""
        # Use shape, dtype, and a sample of values for hash
        shape_str = str(tuple(tensor.shape))
        dtype_str = str(tensor.dtype)
        
        # Sample tensor values for hash (to avoid hashing large tensors)
        if tensor.numel() <= 1000:
            values_hash = hash(tensor.flatten().detach().cpu().numpy().tobytes())
        else:
            # Sample from tensor
            flat = tensor.flatten()
            sample_indices = torch.linspace(0, flat.size(0)-1, min(100, flat.size(0)), dtype=torch.long)
            sample_values = flat[sample_indices]
            values_hash = hash(sample_values.detach().cpu().numpy().tobytes())
        
        return f"tensor_{shape_str}_{dtype_str}_{values_hash}"

class RangeIndex(Index):
    """Range-based index for ordered lookups and range queries."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._values: List[Tuple[Any, str]] = []  # (value, record_id) pairs, sorted
        self._dirty = False
    
    def insert(self, record_id: str, value: Any, tensor: Optional[torch.Tensor] = None) -> None:
        """Insert record into range index."""
        with self._lock:
            if not isinstance(value, (int, float, str)):
                logger.warning(f"RangeIndex '{self.name}': Cannot index non-comparable value {type(value)}")
                return
            
            self._values.append((value, record_id))
            self._dirty = True
    
    def delete(self, record_id: str, value: Any = None) -> None:
        """Delete record from range index."""
        with self._lock:
            if value is not None:
                # Remove specific (value, record_id) pair
                try:
                    self._values.remove((value, record_id))
                except ValueError:
                    pass
            else:
                # Remove all entries for record_id
                self._values = [(v, rid) for v, rid in self._values if rid != record_id]
            self._dirty = True
    
    def lookup(self, value: Any) -> List[str]:
        """Lookup records by exact value."""
        with self._lock:
            self._ensure_sorted()
            result = []
            for v, rid in self._values:
                if v == value:
                    result.append(rid)
                elif v > value:
                    break
            return result
    
    def range_query(self, min_val: Any = None, max_val: Any = None, 
                    include_min: bool = True, include_max: bool = True) -> List[str]:
        """Query records within a range."""
        with self._lock:
            self._ensure_sorted()
            result = []
            
            for value, record_id in self._values:
                # Check lower bound
                if min_val is not None:
                    if include_min:
                        if value < min_val:
                            continue
                    else:
                        if value <= min_val:
                            continue
                
                # Check upper bound
                if max_val is not None:
                    if include_max:
                        if value > max_val:
                            break
                    else:
                        if value >= max_val:
                            break
                
                result.append(record_id)
            
            return result
    
    def clear(self) -> None:
        """Clear the index."""
        with self._lock:
            self._values.clear()
            self._dirty = False
    
    def size(self) -> int:
        """Return number of entries."""
        with self._lock:
            return len(self._values)
    
    def _ensure_sorted(self) -> None:
        """Ensure the values list is sorted."""
        if self._dirty:
            self._values.sort(key=lambda x: x[0])
            self._dirty = False

# === Specialized Indexes ===

class TensorPropertyIndex(Index):
    """Index for tensor properties like shape, dtype, statistics."""
    
    def __init__(self, name: str, property_extractor: Callable[[torch.Tensor], Any]):
        super().__init__(name)
        self.property_extractor = property_extractor
        self._hash_index = HashIndex(f"{name}_hash")
        self._range_index = RangeIndex(f"{name}_range")
    
    def insert(self, record_id: str, value: Any, tensor: Optional[torch.Tensor] = None) -> None:
        """Insert tensor property into index."""
        if tensor is None:
            logger.warning(f"TensorPropertyIndex '{self.name}': No tensor provided")
            return
        
        try:
            prop_value = self.property_extractor(tensor)
            
            # Use appropriate sub-index based on property type
            if isinstance(prop_value, (int, float)):
                self._range_index.insert(record_id, prop_value, tensor)
            else:
                self._hash_index.insert(record_id, prop_value, tensor)
                
        except Exception as e:
            logger.error(f"TensorPropertyIndex '{self.name}': Error extracting property: {e}")
    
    def delete(self, record_id: str, value: Any = None) -> None:
        """Delete record from property index."""
        # Delete from both sub-indexes
        self._hash_index.delete(record_id, None)
        self._range_index.delete(record_id, None)
    
    def lookup(self, value: Any) -> List[str]:
        """Lookup by exact property value."""
        if isinstance(value, (int, float)):
            return self._range_index.lookup(value)
        else:
            return self._hash_index.lookup(value)
    
    def range_query(self, min_val: Any = None, max_val: Any = None,
                    include_min: bool = True, include_max: bool = True) -> List[str]:
        """Range query for numeric properties."""
        return self._range_index.range_query(min_val, max_val, include_min, include_max)
    
    def clear(self) -> None:
        """Clear the index."""
        self._hash_index.clear()
        self._range_index.clear()
    
    def size(self) -> int:
        """Return total entries across sub-indexes."""
        return self._hash_index.size() + self._range_index.size()

class SpatialIndex(Index):
    """Spatial index for tensor shapes and dimensional properties."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._shape_index = HashIndex(f"{name}_shape")
        self._ndim_index = RangeIndex(f"{name}_ndim")
        self._size_index = RangeIndex(f"{name}_size")
    
    def insert(self, record_id: str, value: Any, tensor: Optional[torch.Tensor] = None) -> None:
        """Insert tensor spatial properties."""
        if tensor is None:
            return
            
        shape = tuple(tensor.shape)
        ndim = len(shape)
        size = tensor.numel()
        
        self._shape_index.insert(record_id, shape, tensor)
        self._ndim_index.insert(record_id, ndim, tensor)
        self._size_index.insert(record_id, size, tensor)
    
    def delete(self, record_id: str, value: Any = None) -> None:
        """Delete record from spatial index."""
        self._shape_index.delete(record_id, None)
        self._ndim_index.delete(record_id, None)
        self._size_index.delete(record_id, None)
    
    def lookup(self, value: Any) -> List[str]:
        """Lookup by exact shape."""
        return self._shape_index.lookup(value)
    
    def lookup_by_shape(self, shape: Tuple[int, ...]) -> List[str]:
        """Lookup tensors with exact shape."""
        return self._shape_index.lookup(shape)
    
    def lookup_by_ndim(self, ndim: int) -> List[str]:
        """Lookup tensors with specific number of dimensions."""
        return self._ndim_index.lookup(ndim)
    
    def lookup_by_size_range(self, min_size: int = None, max_size: int = None) -> List[str]:
        """Lookup tensors within size range."""
        return self._size_index.range_query(min_size, max_size)
    
    def clear(self) -> None:
        """Clear the spatial index."""
        self._shape_index.clear()
        self._ndim_index.clear()
        self._size_index.clear()
    
    def size(self) -> int:
        """Return total entries."""
        return self._shape_index.size()

# === Index Manager ===

class IndexManager:
    """Manages multiple indexes for a dataset."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._indexes: Dict[str, Index] = {}
        self._lock = threading.RLock()
        
        # Create default indexes
        self._create_default_indexes()
    
    def _create_default_indexes(self) -> None:
        """Create standard indexes that are always available."""
        # Primary key index for record IDs
        self.add_index("record_id", HashIndex("record_id"))
        
        # Metadata field indexes
        self.add_index("timestamp_utc", RangeIndex("timestamp_utc"))
        self.add_index("dtype", HashIndex("dtype"))
        self.add_index("version", RangeIndex("version"))
        
        # Tensor property indexes
        self.add_index("tensor_shape", TensorPropertyIndex("tensor_shape", lambda t: tuple(t.shape)))
        self.add_index("tensor_ndim", TensorPropertyIndex("tensor_ndim", lambda t: len(t.shape)))
        self.add_index("tensor_size", TensorPropertyIndex("tensor_size", lambda t: t.numel()))
        self.add_index("tensor_dtype", TensorPropertyIndex("tensor_dtype", lambda t: str(t.dtype)))
        self.add_index("tensor_mean", TensorPropertyIndex("tensor_mean", lambda t: float(t.mean())))
        self.add_index("tensor_std", TensorPropertyIndex("tensor_std", lambda t: float(t.std())))
        self.add_index("tensor_min", TensorPropertyIndex("tensor_min", lambda t: float(t.min())))
        self.add_index("tensor_max", TensorPropertyIndex("tensor_max", lambda t: float(t.max())))
        
        # Spatial index
        self.add_index("spatial", SpatialIndex("spatial"))
    
    def add_index(self, name: str, index: Index) -> None:
        """Add an index to the manager."""
        with self._lock:
            self._indexes[name] = index
            logger.debug(f"Added index '{name}' to dataset '{self.dataset_name}'")
    
    def get_index(self, name: str) -> Optional[Index]:
        """Get an index by name."""
        with self._lock:
            return self._indexes.get(name)
    
    def remove_index(self, name: str) -> bool:
        """Remove an index."""
        with self._lock:
            if name in self._indexes:
                del self._indexes[name]
                logger.debug(f"Removed index '{name}' from dataset '{self.dataset_name}'")
                return True
            return False
    
    def list_indexes(self) -> List[str]:
        """List all available indexes."""
        with self._lock:
            return list(self._indexes.keys())
    
    def insert_record(self, record_id: str, metadata: Dict[str, Any], tensor: torch.Tensor) -> None:
        """Insert a record into all applicable indexes."""
        with self._lock:
            # First, create indexes for any new metadata fields we haven't seen before
            for key, value in metadata.items():
                if key not in ["record_id", "timestamp_utc", "dtype", "version"] and key not in self._indexes:
                    # Create index for this metadata field
                    if isinstance(value, (int, float)):
                        self.add_index(key, RangeIndex(key))
                    else:
                        self.add_index(key, HashIndex(key))
            
            # Now insert into all applicable indexes
            for index_name, index in self._indexes.items():
                try:
                    if index_name == "record_id":
                        index.insert(record_id, record_id, tensor)
                    elif index_name == "spatial":
                        index.insert(record_id, None, tensor)
                    elif index_name.startswith("tensor_"):
                        index.insert(record_id, None, tensor)
                    elif index_name in metadata:
                        index.insert(record_id, metadata[index_name], tensor)
                except Exception as e:
                    logger.warning(f"Failed to insert record '{record_id}' into index '{index_name}': {e}")
    
    def delete_record(self, record_id: str, metadata: Optional[Dict[str, Any]] = None, 
                     tensor: Optional[torch.Tensor] = None) -> None:
        """Delete a record from all indexes."""
        with self._lock:
            for index_name, index in self._indexes.items():
                try:
                    if metadata and index_name in metadata:
                        index.delete(record_id, metadata[index_name])
                    else:
                        index.delete(record_id, None)
                except Exception as e:
                    logger.warning(f"Failed to delete record '{record_id}' from index '{index_name}': {e}")
    
    def clear_all_indexes(self) -> None:
        """Clear all indexes."""
        with self._lock:
            for index in self._indexes.values():
                try:
                    index.clear()
                except Exception as e:
                    logger.error(f"Failed to clear index: {e}")
    
    def get_index_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all indexes."""
        with self._lock:
            stats = {}
            for name, index in self._indexes.items():
                try:
                    stats[name] = {
                        "type": type(index).__name__,
                        "size": index.size(),
                        "name": index.name
                    }
                except Exception as e:
                    stats[name] = {"error": str(e)}
            return stats

# === Query Builder ===

class QueryBuilder:
    """Builder for constructing optimized queries using indexes."""
    
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self._conditions: List[Dict[str, Any]] = []
    
    def where(self, field: str, value: Any) -> 'QueryBuilder':
        """Add exact match condition."""
        self._conditions.append({
            "type": "exact",
            "field": field,
            "value": value
        })
        return self
    
    def where_range(self, field: str, min_val: Any = None, max_val: Any = None,
                   include_min: bool = True, include_max: bool = True) -> 'QueryBuilder':
        """Add range condition."""
        self._conditions.append({
            "type": "range",
            "field": field,
            "min_val": min_val,
            "max_val": max_val,
            "include_min": include_min,
            "include_max": include_max
        })
        return self
    
    def where_shape(self, shape: Tuple[int, ...]) -> 'QueryBuilder':
        """Add tensor shape condition."""
        self._conditions.append({
            "type": "shape",
            "shape": shape
        })
        return self
    
    def where_ndim(self, ndim: int) -> 'QueryBuilder':
        """Add tensor dimension condition."""
        self._conditions.append({
            "type": "ndim", 
            "ndim": ndim
        })
        return self
    
    def execute(self) -> List[str]:
        """Execute the query and return matching record IDs."""
        if not self._conditions:
            return []
        
        # Find the most selective condition first
        result_sets = []
        
        for condition in self._conditions:
            record_ids = self._execute_condition(condition)
            result_sets.append(set(record_ids))
        
        # Intersect all result sets
        if result_sets:
            result = result_sets[0]
            for rs in result_sets[1:]:
                result = result.intersection(rs)
            return list(result)
        
        return []
    
    def _execute_condition(self, condition: Dict[str, Any]) -> List[str]:
        """Execute a single condition."""
        cond_type = condition["type"]
        
        if cond_type == "exact":
            field = condition["field"]
            value = condition["value"]
            index = self.index_manager.get_index(field)
            if index:
                return index.lookup(value)
            return []
        
        elif cond_type == "range":
            field = condition["field"]
            index = self.index_manager.get_index(field)
            if index and hasattr(index, 'range_query'):
                return index.range_query(
                    condition.get("min_val"),
                    condition.get("max_val"),
                    condition.get("include_min", True),
                    condition.get("include_max", True)
                )
            return []
        
        elif cond_type == "shape":
            spatial_index = self.index_manager.get_index("spatial")
            if spatial_index and hasattr(spatial_index, 'lookup_by_shape'):
                return spatial_index.lookup_by_shape(condition["shape"])
            return []
        
        elif cond_type == "ndim":
            spatial_index = self.index_manager.get_index("spatial")
            if spatial_index and hasattr(spatial_index, 'lookup_by_ndim'):
                return spatial_index.lookup_by_ndim(condition["ndim"])
            return []
        
        return []

# === Predefined Property Extractors ===

TENSOR_PROPERTIES = {
    "shape": lambda t: tuple(t.shape),
    "ndim": lambda t: len(t.shape),
    "size": lambda t: t.numel(),
    "dtype": lambda t: str(t.dtype),
    "mean": lambda t: float(t.mean()) if t.numel() > 0 else 0.0,
    "std": lambda t: float(t.std()) if t.numel() > 0 else 0.0,
    "min": lambda t: float(t.min()) if t.numel() > 0 else 0.0,
    "max": lambda t: float(t.max()) if t.numel() > 0 else 0.0,
    "norm": lambda t: float(torch.norm(t)) if t.numel() > 0 else 0.0,
    "sparsity": lambda t: float((t == 0).sum()) / t.numel() if t.numel() > 0 else 0.0,
}

def create_tensor_property_index(name: str, property_name: str) -> TensorPropertyIndex:
    """Create a tensor property index using predefined extractors."""
    if property_name not in TENSOR_PROPERTIES:
        raise IndexingError(f"Unknown tensor property: {property_name}")
    
    return TensorPropertyIndex(name, TENSOR_PROPERTIES[property_name])

def create_custom_metadata_index(name: str, field_name: str, index_type: str = "hash") -> Index:
    """Create an index for custom metadata fields."""
    if index_type == "hash":
        return HashIndex(name)
    elif index_type == "range":
        return RangeIndex(name)
    else:
        raise IndexingError(f"Unknown index type: {index_type}")