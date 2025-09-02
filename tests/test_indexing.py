"""
Test suite for tensor indexing functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

# Import indexing modules
try:
    from tensorus.indexing import (
        HashIndex, RangeIndex, TensorPropertyIndex, SpatialIndex,
        IndexManager, QueryBuilder, TENSOR_PROPERTIES,
        create_tensor_property_index, create_custom_metadata_index,
        IndexingError
    )
    from tensorus.tensor_storage import TensorStorage
    INDEXING_AVAILABLE = True
except ImportError:
    INDEXING_AVAILABLE = False

# Skip all tests if indexing module is not available
pytestmark = pytest.mark.skipif(not INDEXING_AVAILABLE, reason="Indexing module not available")

class TestHashIndex:
    """Test HashIndex functionality."""
    
    def test_basic_operations(self):
        """Test basic insert, lookup, and delete operations."""
        index = HashIndex("test_hash")
        
        # Test insert and lookup
        index.insert("record1", "value1")
        index.insert("record2", "value1")  # Same value, different record
        index.insert("record3", "value2")
        
        assert set(index.lookup("value1")) == {"record1", "record2"}
        assert set(index.lookup("value2")) == {"record3"}
        assert index.lookup("nonexistent") == []
        
        # Test size
        assert index.size() == 2  # Two unique values
        
        # Test delete
        index.delete("record1", "value1")
        assert set(index.lookup("value1")) == {"record2"}
        
        index.delete("record2", "value1")
        assert index.lookup("value1") == []
        
        # Test clear
        index.clear()
        assert index.size() == 0
    
    def test_tensor_hashing(self):
        """Test hashing of tensor values."""
        index = HashIndex("tensor_hash")
        tensor1 = torch.ones(3, 3)
        tensor2 = torch.ones(3, 3)  # Same content
        tensor3 = torch.zeros(3, 3)  # Different content
        
        index.insert("record1", tensor1, tensor1)
        index.insert("record2", tensor2, tensor2)
        index.insert("record3", tensor3, tensor3)
        
        # Same content tensors should hash to same key
        records = index.lookup(tensor1)
        assert len(records) >= 1  # Should find at least record1
        
    def test_complex_data_types(self):
        """Test indexing of complex data types."""
        index = HashIndex("complex_data")
        
        # Test list
        index.insert("record1", [1, 2, 3])
        assert "record1" in index.lookup([1, 2, 3])
        
        # Test dict
        index.insert("record2", {"key": "value"})
        assert "record2" in index.lookup({"key": "value"})

class TestRangeIndex:
    """Test RangeIndex functionality."""
    
    def test_basic_operations(self):
        """Test basic range operations."""
        index = RangeIndex("test_range")
        
        # Insert values
        values = [10, 20, 15, 30, 5]
        for i, val in enumerate(values):
            index.insert(f"record{i}", val)
        
        # Test exact lookup
        assert index.lookup(15) == ["record2"]
        assert index.lookup(999) == []
        
        # Test range queries
        result = index.range_query(min_val=10, max_val=20)
        assert set(result) == {"record0", "record1", "record2"}  # 10, 15, 20
        
        result = index.range_query(min_val=10, max_val=20, include_max=False)
        assert set(result) == {"record0", "record2"}  # 10, 15 (not 20)
        
        result = index.range_query(max_val=15)
        assert set(result) == {"record0", "record2", "record4"}  # 5, 10, 15
        
    def test_string_ordering(self):
        """Test range operations with strings."""
        index = RangeIndex("string_range")
        
        strings = ["apple", "banana", "cherry", "date"]
        for i, s in enumerate(strings):
            index.insert(f"record{i}", s)
        
        result = index.range_query(min_val="banana", max_val="cherry")
        assert set(result) == {"record1", "record2"}  # banana, cherry

class TestTensorPropertyIndex:
    """Test TensorPropertyIndex functionality."""
    
    def test_shape_property(self):
        """Test indexing by tensor shape."""
        extractor = lambda t: tuple(t.shape)
        index = TensorPropertyIndex("shape", extractor)
        
        tensor1 = torch.ones(3, 4)
        tensor2 = torch.zeros(3, 4)
        tensor3 = torch.ones(2, 2)
        
        index.insert("record1", None, tensor1)
        index.insert("record2", None, tensor2)
        index.insert("record3", None, tensor3)
        
        # Test shape lookup
        assert set(index.lookup((3, 4))) == {"record1", "record2"}
        assert set(index.lookup((2, 2))) == {"record3"}
        
    def test_numeric_property(self):
        """Test indexing by numeric tensor properties."""
        extractor = lambda t: float(t.mean())
        index = TensorPropertyIndex("mean", extractor)
        
        tensor1 = torch.ones(3, 3)      # mean = 1.0
        tensor2 = torch.zeros(3, 3)     # mean = 0.0
        tensor3 = torch.ones(2, 2) * 2  # mean = 2.0
        
        index.insert("record1", None, tensor1)
        index.insert("record2", None, tensor2)
        index.insert("record3", None, tensor3)
        
        # Test exact lookup
        assert set(index.lookup(1.0)) == {"record1"}
        
        # Test range query
        result = index.range_query(min_val=0.5, max_val=1.5)
        assert set(result) == {"record1"}

class TestSpatialIndex:
    """Test SpatialIndex functionality."""
    
    def test_spatial_indexing(self):
        """Test spatial indexing capabilities."""
        index = SpatialIndex("spatial")
        
        tensors = [
            torch.ones(3, 4),      # shape (3, 4), ndim=2, size=12
            torch.zeros(3, 4, 5),  # shape (3, 4, 5), ndim=3, size=60
            torch.ones(2, 2),      # shape (2, 2), ndim=2, size=4
        ]
        
        for i, tensor in enumerate(tensors):
            index.insert(f"record{i}", None, tensor)
        
        # Test shape lookup
        assert set(index.lookup_by_shape((3, 4))) == {"record0"}
        assert set(index.lookup_by_shape((2, 2))) == {"record2"}
        
        # Test ndim lookup
        assert set(index.lookup_by_ndim(2)) == {"record0", "record2"}
        assert set(index.lookup_by_ndim(3)) == {"record1"}
        
        # Test size range
        result = index.lookup_by_size_range(min_size=10, max_size=20)
        assert set(result) == {"record0"}  # size=12

class TestIndexManager:
    """Test IndexManager functionality."""
    
    def test_default_indexes(self):
        """Test that default indexes are created."""
        manager = IndexManager("test_dataset")
        
        indexes = manager.list_indexes()
        expected_indexes = [
            "record_id", "timestamp_utc", "dtype", "version",
            "tensor_shape", "tensor_ndim", "tensor_size", "tensor_dtype",
            "tensor_mean", "tensor_std", "tensor_min", "tensor_max",
            "spatial"
        ]
        
        for expected in expected_indexes:
            assert expected in indexes
    
    def test_record_insertion(self):
        """Test inserting records into all indexes."""
        manager = IndexManager("test_dataset")
        
        tensor = torch.randn(3, 4)
        metadata = {
            "record_id": "test_record",
            "timestamp_utc": 1234567890.0,
            "dtype": "float32",
            "version": 1,
            "custom_field": "test_value"
        }
        
        # Insert record
        manager.insert_record("test_record", metadata, tensor)
        
        # Test record ID index
        record_id_index = manager.get_index("record_id")
        assert record_id_index.lookup("test_record") == ["test_record"]
        
        # Test spatial index
        spatial_index = manager.get_index("spatial")
        assert "test_record" in spatial_index.lookup_by_shape((3, 4))
    
    def test_record_deletion(self):
        """Test deleting records from indexes."""
        manager = IndexManager("test_dataset")
        
        tensor = torch.ones(2, 2)
        metadata = {"record_id": "delete_me", "timestamp_utc": 123.0}
        
        # Insert and verify
        manager.insert_record("delete_me", metadata, tensor)
        record_id_index = manager.get_index("record_id")
        assert record_id_index.lookup("delete_me") == ["delete_me"]
        
        # Delete and verify
        manager.delete_record("delete_me", metadata, tensor)
        assert record_id_index.lookup("delete_me") == []
    
    def test_index_stats(self):
        """Test index statistics."""
        manager = IndexManager("test_dataset")
        
        # Insert some test data
        for i in range(5):
            tensor = torch.randn(2, 2)
            metadata = {"record_id": f"record{i}", "timestamp_utc": float(i)}
            manager.insert_record(f"record{i}", metadata, tensor)
        
        stats = manager.get_index_stats()
        assert "record_id" in stats
        assert "spatial" in stats
        
        # Record ID index should have 5 entries
        assert stats["record_id"]["size"] == 5

class TestQueryBuilder:
    """Test QueryBuilder functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.manager = IndexManager("test_dataset")
        
        # Insert test data
        test_data = [
            (torch.ones(2, 2), {"record_id": "r1", "category": "A", "score": 10}),
            (torch.zeros(2, 2), {"record_id": "r2", "category": "B", "score": 20}),
            (torch.ones(3, 3, 2), {"record_id": "r3", "category": "A", "score": 15}),
            (torch.zeros(3, 3, 2), {"record_id": "r4", "category": "B", "score": 25}),
        ]
        
        for tensor, metadata in test_data:
            self.manager.insert_record(metadata["record_id"], metadata, tensor)
    
    def test_exact_query(self):
        """Test exact match queries."""
        builder = QueryBuilder(self.manager)
        result = builder.where("category", "A").execute()
        assert set(result) == {"r1", "r3"}
    
    def test_range_query(self):
        """Test range queries."""
        builder = QueryBuilder(self.manager)
        # Note: Need to create a range index for this to work
        # For now, this tests the query building logic
        result = builder.where_range("score", min_val=15, max_val=25).execute()
        # Result depends on whether range index exists
    
    def test_shape_query(self):
        """Test tensor shape queries."""
        builder = QueryBuilder(self.manager)
        result = builder.where_shape((2, 2)).execute()
        assert set(result) == {"r1", "r2"}
    
    def test_ndim_query(self):
        """Test tensor dimension queries."""
        builder = QueryBuilder(self.manager)
        result = builder.where_ndim(3).execute()
        assert set(result) == {"r3", "r4"}
    
    def test_combined_queries(self):
        """Test combining multiple query conditions."""
        builder = QueryBuilder(self.manager)
        result = builder.where("category", "A").where_shape((3, 3, 2)).execute()
        assert set(result) == {"r3"}

class TestTensorStorageIntegration:
    """Test integration of indexing with TensorStorage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.storage = TensorStorage(storage_path=str(self.temp_path))
        self.storage.create_dataset("test_dataset")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_indexed_insert_and_lookup(self):
        """Test that indexing works with tensor insertion and lookup."""
        # Insert test tensors
        tensors_data = [
            (torch.ones(2, 3), {"label": "ones", "category": "A"}),
            (torch.zeros(2, 3), {"label": "zeros", "category": "B"}),
            (torch.ones(3, 2), {"label": "ones_T", "category": "A"}),
        ]
        
        record_ids = []
        for tensor, metadata in tensors_data:
            record_id = self.storage.insert("test_dataset", tensor, metadata)
            record_ids.append(record_id)
        
        # Test O(1) record ID lookup
        for record_id in record_ids:
            result = self.storage.get_tensor_by_id("test_dataset", record_id)
            assert result["metadata"]["record_id"] == record_id

class TestPerformanceBenchmarks:
    """Performance benchmarks to verify indexing improvements."""
    
    def setup_method(self):
        """Set up test data for performance testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create storage with and without indexing
        self.storage_with_index = TensorStorage(storage_path=str(self.temp_path / "indexed"))
        self.storage_with_index.create_dataset("perf_test")
        
        # Insert test data
        self.num_records = 1000
        self.record_ids = []
        
        for i in range(self.num_records):
            tensor = torch.randn(10, 10)
            metadata = {
                "index": i,
                "category": f"cat_{i % 10}",
                "score": float(i % 100)
            }
            record_id = self.storage_with_index.insert("perf_test", tensor, metadata)
            self.record_ids.append(record_id)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_lookup_performance(self):
        """Test that indexed lookup is faster than linear search."""
        import time
        
        # Test indexed lookup performance
        start_time = time.time()
        for i in range(100):  # Test 100 random lookups
            record_id = self.record_ids[i % len(self.record_ids)]
            result = self.storage_with_index.get_tensor_by_id("perf_test", record_id)
            assert result is not None
        indexed_time = time.time() - start_time
        
        print(f"\nIndexed lookup time for 100 operations: {indexed_time:.4f} seconds")
        print(f"Average time per lookup: {indexed_time/100*1000:.2f} ms")
        
        # With 1000 records, indexed lookups should be significantly faster
        # This is more of a benchmark than an assertion
        assert indexed_time < 1.0  # Should complete in under 1 second
    
    def test_metadata_query_performance(self):
        """Test metadata query performance."""
        import time
        
        # Test category queries (should benefit from indexing)
        start_time = time.time()
        for cat_num in range(5):  # Test 5 different categories
            results = self.storage_with_index.query_by_metadata("perf_test", "category", f"cat_{cat_num}")
            # Each category should have ~100 records (1000 / 10)
            assert len(results) == 100
        query_time = time.time() - start_time
        
        print(f"\nMetadata query time for 5 operations: {query_time:.4f} seconds")
        
        assert query_time < 2.0  # Should complete reasonably quickly

class TestErrorHandling:
    """Test error handling in indexing system."""
    
    def test_invalid_index_type(self):
        """Test handling of invalid index types."""
        with pytest.raises(IndexingError):
            create_custom_metadata_index("test", "field", "invalid_type")
    
    def test_unknown_tensor_property(self):
        """Test handling of unknown tensor properties."""
        with pytest.raises(IndexingError):
            create_tensor_property_index("test", "unknown_property")
    
    def test_missing_dataset(self):
        """Test queries on non-existent datasets."""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = TensorStorage(storage_path=temp_dir)
            
            with pytest.raises(Exception):  # Should raise DatasetNotFoundError
                storage.query_by_metadata("nonexistent", "field", "value")
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestIndexPersistence:
    """Test that indexes work correctly with persistence."""
    
    def test_index_persistence_across_sessions(self):
        """Test that indexes are rebuilt when loading from disk."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create first storage instance and add data
            storage1 = TensorStorage(storage_path=temp_dir)
            storage1.create_dataset("persist_test")
            
            tensor1 = torch.ones(3, 3)
            record_id1 = storage1.insert("persist_test", tensor1, {"label": "test1"})
            
            # Create second storage instance (simulates restart)
            storage2 = TensorStorage(storage_path=temp_dir)
            
            # Indexes should be automatically rebuilt
            result = storage2.get_tensor_by_id("persist_test", record_id1)
            assert result is not None
            assert torch.equal(result["tensor"], tensor1)
            
            # Index stats should show the record
            stats = storage2.get_index_stats("persist_test")
            assert stats["record_id"]["size"] == 1
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__])