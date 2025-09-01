"""
Comprehensive tests for Efficient Indexing System

Tests cover:
- Index creation and basic operations
- Query optimization and performance
- Persistence and recovery
- Maintenance and optimization
- Integration with metadata storage
- Performance benchmarks and scalability
"""

import os
import time
import tempfile
import unittest
import shutil
from unittest.mock import Mock, patch
import uuid

from tensorus.metadata.indexing import (
    BaseIndex, HashIndex, BTreeIndex, SpatialIndex, CompositeIndex,
    IndexType, IndexStructure, IndexMetadata
)
from tensorus.metadata.index_manager import IndexManager, IndexManagerConfig
from tensorus.metadata.index_persistence import (
    IndexPersistenceManager, IndexMaintenanceManager, IndexedMetadataStorage
)
from tensorus.metadata.schemas import TensorDescriptor, DataType
from datetime import datetime


class TestIndexTypes(unittest.TestCase):
    """Test basic index operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [
            {
                "tensor_id": "tensor_1",
                "shape": [100, 200],
                "data_type": "float32",
                "owner": "user1",
                "tags": ["test", "sample"],
                "byte_size": 80000
            },
            {
                "tensor_id": "tensor_2",
                "shape": [50, 100],
                "data_type": "int64",
                "owner": "user2",
                "tags": ["production"],
                "byte_size": 40000
            },
            {
                "tensor_id": "tensor_3",
                "shape": [200, 300],
                "data_type": "float32",
                "owner": "user1",
                "tags": ["test", "large"],
                "byte_size": 240000
            }
        ]

    def test_hash_index(self):
        """Test hash index operations."""
        index = HashIndex("test_owner", ["owner"])

        # Insert data
        for data in self.test_data:
            index.insert(data["tensor_id"], data)

        # Test exact match query
        results = index.search({"owner": "user1"})
        self.assertEqual(len(results), 2)
        self.assertIn("tensor_1", results)
        self.assertIn("tensor_3", results)

        # Test non-existent query
        results = index.search({"owner": "nonexistent"})
        self.assertEqual(len(results), 0)

        # Test removal
        self.assertTrue(index.remove("tensor_1"))
        results = index.search({"owner": "user1"})
        self.assertEqual(len(results), 1)

        # Test statistics
        stats = index.get_statistics()
        self.assertIn("total_entries", stats)
        self.assertIn("avg_bucket_size", stats)

    def test_spatial_index(self):
        """Test spatial index operations."""
        index = SpatialIndex("test_shape")

        # Insert data
        for data in self.test_data:
            index.insert(data["tensor_id"], data)

        # Test exact shape query
        results = index.search({"type": "exact", "shape": [100, 200]})
        self.assertEqual(len(results), 1)
        self.assertIn("tensor_1", results)

        # Test dimensionality query
        results = index.search({"type": "dimensionality", "dimensionality": 2})
        self.assertEqual(len(results), 3)

        # Test size range query
        results = index.search({
            "type": "size_range",
            "min_size": 50000,
            "max_size": 200000
        })
        self.assertEqual(len(results), 2)  # tensor_1 and tensor_2

        # Test removal
        self.assertTrue(index.remove("tensor_1"))
        results = index.search({"type": "exact", "shape": [100, 200]})
        self.assertEqual(len(results), 0)

    def test_composite_index(self):
        """Test composite index operations."""
        index = CompositeIndex("test_composite", ["owner", "data_type"])

        # Insert data
        for data in self.test_data:
            index.insert(data["tensor_id"], data)

        # Test exact composite query
        results = index.search({"owner": "user1", "data_type": "float32"})
        self.assertEqual(len(results), 2)

        # Test partial query (should match both user1 tensors)
        results = index.search({"owner": "user1"})
        self.assertEqual(len(results), 2)

        # Test non-matching query
        results = index.search({"owner": "user1", "data_type": "int64"})
        self.assertEqual(len(results), 0)


class TestIndexManager(unittest.TestCase):
    """Test IndexManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        config = IndexManagerConfig(
            max_cache_size=10,
            enable_query_caching=True
        )
        self.manager = IndexManager(config)

        # Add test data
        self.test_tensors = [
            {
                "tensor_id": "t1",
                "shape": [100, 200],
                "data_type": "float32",
                "owner": "alice",
                "tags": ["test"],
                "byte_size": 80000
            },
            {
                "tensor_id": "t2",
                "shape": [50, 100],
                "data_type": "int64",
                "owner": "bob",
                "tags": ["production"],
                "byte_size": 40000
            }
        ]

        for tensor in self.test_tensors:
            self.manager.add_tensor(tensor["tensor_id"], tensor)

    def test_index_creation(self):
        """Test creating new indexes."""
        # Create a custom index
        success = self.manager.create_index(
            "custom_owner_dtype",
            IndexType.COMPOSITE,
            ["owner", "data_type"]
        )
        self.assertTrue(success)
        self.assertIn("custom_owner_dtype", self.manager.indexes)

    def test_query_optimization(self):
        """Test query optimization and execution."""
        # Query by owner
        results = self.manager.query_tensors({"owner": "alice"})
        self.assertEqual(len(results), 1)
        self.assertIn("t1", results)

        # Query by data type
        results = self.manager.query_tensors({"data_type": "int64"})
        self.assertEqual(len(results), 1)
        self.assertIn("t2", results)

        # Query with limit
        results = self.manager.query_tensors({"owner": "alice"}, limit=1)
        self.assertEqual(len(results), 1)

    def test_query_caching(self):
        """Test query result caching."""
        query = {"owner": "alice"}

        # First query
        results1 = self.manager.query_tensors(query)
        stats1 = self.manager.get_statistics()

        # Second query (should use cache)
        results2 = self.manager.query_tensors(query)
        stats2 = self.manager.get_statistics()

        # Results should be identical
        self.assertEqual(results1, results2)

        # Cache hit rate should be > 0 after second query
        hit_rate = stats2["cache_hit_rate"]
        self.assertGreater(hit_rate, 0)

    def test_statistics(self):
        """Test statistics collection."""
        stats = self.manager.get_statistics()

        self.assertIn("total_indexes", stats)
        self.assertIn("cache_size", stats)
        self.assertIn("performance_stats", stats)
        self.assertGreater(stats["total_indexes"], 0)


class TestIndexPersistence(unittest.TestCase):
    """Test index persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = IndexPersistenceManager(self.temp_dir)
        self.manager = IndexManager()
        self.manager.create_index("test_hash", IndexType.PROPERTY, ["owner"])

        # Add test data
        test_data = {
            "tensor_id": "test_tensor",
            "owner": "test_user",
            "shape": [10, 20]
        }
        self.manager.add_tensor("test_tensor", test_data)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_save_load(self):
        """Test saving and loading indexes."""
        # Save index
        success = self.persistence.save_index(self.manager.indexes["test_hash"])
        self.assertTrue(success)

        # Create new manager and load index
        new_manager = IndexManager()
        loaded_index = self.persistence.load_index("test_hash", new_manager)
        self.assertIsNotNone(loaded_index)

        # Test loaded index
        results = loaded_index.search({"owner": "test_user"})
        self.assertEqual(len(results), 1)
        self.assertIn("test_tensor", results)

    def test_backup_restore(self):
        """Test backup and restore functionality."""
        # Create backup
        backup_path = self.persistence.create_backup("test_backup")
        self.assertTrue(os.path.exists(backup_path))

        # Modify original index
        self.manager.remove_tensor("test_tensor")

        # Restore from backup
        success = self.persistence.restore_backup("test_backup", self.manager)
        self.assertTrue(success)

        # Verify restoration
        results = self.manager.query_tensors({"owner": "test_user"})
        self.assertEqual(len(results), 1)

    def test_storage_stats(self):
        """Test storage statistics."""
        # Save some indexes
        for index in self.manager.indexes.values():
            self.persistence.save_index(index)

        stats = self.persistence.get_storage_stats()
        self.assertIn("total_size_bytes", stats)
        self.assertIn("index_count", stats)
        self.assertGreater(stats["index_count"], 0)


class TestIndexMaintenance(unittest.TestCase):
    """Test index maintenance functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = IndexPersistenceManager(self.temp_dir)
        self.manager = IndexManager()
        self.maintenance = IndexMaintenanceManager(
            self.manager, self.persistence, maintenance_interval=1
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.maintenance.stop_maintenance()

    def test_crud_operations(self):
        """Test CRUD operations trigger index updates."""
        # Create tensor
        tensor_data = {
            "tensor_id": "maintenance_test",
            "owner": "test_user",
            "shape": [10, 10]
        }
        self.maintenance.on_tensor_created("maintenance_test", tensor_data)

        # Verify in index
        results = self.manager.query_tensors({"owner": "test_user"})
        self.assertIn("maintenance_test", results)

        # Update tensor
        new_data = tensor_data.copy()
        new_data["owner"] = "updated_user"
        self.maintenance.on_tensor_updated("maintenance_test", tensor_data, new_data)

        # Verify update
        results = self.manager.query_tensors({"owner": "updated_user"})
        self.assertIn("maintenance_test", results)

        # Delete tensor
        self.maintenance.on_tensor_deleted("maintenance_test")

        # Verify deletion
        results = self.manager.query_tensors({"owner": "updated_user"})
        self.assertNotIn("maintenance_test", results)

    def test_maintenance_operations(self):
        """Test maintenance operations."""
        # Add some test data
        for i in range(5):
            tensor_data = {
                "tensor_id": f"test_{i}",
                "owner": f"user_{i % 2}",
                "shape": [10, 10]
            }
            self.manager.add_tensor(f"test_{i}", tensor_data)

        # Perform maintenance
        results = self.maintenance.perform_maintenance()

        self.assertIn("operations", results)
        self.assertIn("index_saving", results["operations"])

    def test_maintenance_status(self):
        """Test maintenance status reporting."""
        status = self.maintenance.get_maintenance_status()
        self.assertIn("maintenance_running", status)
        self.assertIn("last_maintenance", status)


class TestIndexedMetadataStorage(unittest.TestCase):
    """Test integration with metadata storage."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock metadata storage
        self.mock_storage = Mock()

        # Create indexed storage
        self.temp_dir = tempfile.mkdtemp()
        self.indexed_storage = IndexedMetadataStorage(
            self.mock_storage, self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.indexed_storage.maintenance_manager.stop_maintenance()

    def test_tensor_descriptor_operations(self):
        """Test tensor descriptor operations with indexing."""
        # Mock tensor descriptor
        mock_descriptor = Mock()
        mock_descriptor.tensor_id = uuid.uuid4()
        mock_descriptor.shape = [100, 200]
        mock_descriptor.data_type.value = "float32"
        mock_descriptor.owner = "test_user"
        mock_descriptor.tags = ["test"]
        mock_descriptor.byte_size = 80000
        mock_descriptor.creation_timestamp = datetime.utcnow()
        mock_descriptor.dimensionality = 2

        # Add descriptor
        self.indexed_storage.add_tensor_descriptor(mock_descriptor)

        # Verify base storage was called
        self.mock_storage.add_tensor_descriptor.assert_called_once_with(mock_descriptor)

        # Verify indexing
        results = self.indexed_storage.index_manager.query_tensors({"owner": "test_user"})
        self.assertEqual(len(results), 1)

    def test_query_integration(self):
        """Test query integration between indexed and base storage."""
        # Mock base storage query results
        mock_results = [Mock(tensor_id="mock_tensor")]
        self.mock_storage.list_tensor_descriptors.return_value = mock_results

        # Query through indexed storage
        results = self.indexed_storage.query_tensors({"owner": "test_user"})

        # Should return results from indexed query if available
        # or fallback to base storage
        self.assertIsInstance(results, list)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for indexing system."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.manager = IndexManager()

        # Generate test data
        self.test_tensors = []
        for i in range(1000):
            tensor = {
                "tensor_id": f"perf_tensor_{i}",
                "shape": [10 * (i % 10 + 1), 20 * (i % 5 + 1)],
                "data_type": "float32" if i % 2 == 0 else "int64",
                "owner": f"user_{i % 10}",
                "tags": [f"tag_{i % 3}"],
                "byte_size": 1000 * (i + 1)
            }
            self.test_tensors.append(tensor)
            self.manager.add_tensor(tensor["tensor_id"], tensor)

    def test_query_performance(self):
        """Benchmark query performance."""
        # Test different query types
        queries = [
            {"owner": "user_5"},
            {"data_type": "float32"},
            {"tags": ["tag_1"]},
        ]

        for query in queries:
            start_time = time.time()
            results = self.manager.query_tensors(query)
            query_time = time.time() - start_time

            # Should complete in reasonable time
            self.assertLess(query_time, 0.1, f"Query {query} took too long: {query_time}s")
            self.assertIsInstance(results, list)

    def test_index_scaling(self):
        """Test index performance scaling."""
        # Add more tensors
        additional_tensors = []
        for i in range(1000, 2000):
            tensor = {
                "tensor_id": f"scale_tensor_{i}",
                "shape": [5, 5],
                "data_type": "float32",
                "owner": f"scale_user_{i % 20}",
                "tags": [],
                "byte_size": 100
            }
            additional_tensors.append(tensor)
            self.manager.add_tensor(tensor["tensor_id"], tensor)

        # Test query on larger dataset
        start_time = time.time()
        results = self.manager.query_tensors({"data_type": "float32"})
        query_time = time.time() - start_time

        # Should still be fast even with more data
        self.assertLess(query_time, 0.2, f"Scaled query took too long: {query_time}s")
        self.assertGreater(len(results), 500)  # Should find many results

    def test_cache_performance(self):
        """Test query caching performance."""
        query = {"owner": "user_0"}

        # Warm up cache
        self.manager.query_tensors(query)

        # Time cached query
        start_time = time.time()
        for _ in range(100):
            results = self.manager.query_tensors(query)
        cached_time = time.time() - start_time

        # Should be very fast with caching
        avg_cached_time = cached_time / 100
        self.assertLess(avg_cached_time, 0.001, f"Cached query too slow: {avg_cached_time}s")


class TestIndexOptimization(unittest.TestCase):
    """Test index optimization and analysis."""

    def setUp(self):
        """Set up optimization test fixtures."""
        self.manager = IndexManager()

        # Add diverse test data to create optimization opportunities
        owners = ["alice", "bob", "charlie"] * 100
        data_types = ["float32", "int64", "float64"] * 100

        for i in range(300):
            tensor = {
                "tensor_id": f"opt_tensor_{i}",
                "shape": [10, 10],
                "data_type": data_types[i],
                "owner": owners[i],
                "tags": [f"batch_{i % 5}"],
                "byte_size": 400
            }
            self.manager.add_tensor(tensor["tensor_id"], tensor)

    def test_index_analysis(self):
        """Test index usage analysis."""
        # Run some queries to generate usage statistics
        for i in range(10):
            self.manager.query_tensors({"owner": f"alice"})
            self.manager.query_tensors({"data_type": "float32"})

        stats = self.manager.get_statistics()

        # Should have collected usage statistics
        self.assertIn("performance_stats", stats)
        self.assertGreater(stats["performance_stats"]["queries_executed"], 0)

    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        recommendations = self.manager.optimize_indexes()

        # Should provide some recommendations
        self.assertIsInstance(recommendations, dict)
        self.assertIn("indexes_to_create", recommendations)
        self.assertIn("indexes_to_drop", recommendations)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
