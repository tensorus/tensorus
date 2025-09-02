import unittest
import pytest
import logging

pytest.importorskip("torch")
import torch
import shutil
import os
from pathlib import Path
from tensorus.tensor_storage import (
    TensorStorage,
    DatasetNotFoundError,
    TensorNotFoundError,
    SchemaValidationError,
)  # Assuming tensor_storage.py is accessible

class TestTensorStorageInMemory(unittest.TestCase):
    def setUp(self):
        """Set up a TensorStorage instance for in-memory testing."""
        self.storage = TensorStorage(storage_path=None)
        self.tensor1 = torch.rand(3, 10)
        self.meta1 = {"source": "test1", "value": 1}
        self.tensor2 = torch.tensor([1, 2, 3])
        self.meta2 = {"source": "test2", "value": 2}
        self.dataset_name1 = "test_dataset_alpha"
        self.dataset_name2 = "test_dataset_beta"

    def tearDown(self):
        """Clean up (not strictly necessary for in-memory if no side effects)."""
        del self.storage
        del self.tensor1
        del self.meta1
        del self.tensor2
        del self.meta2

    def test_create_dataset(self):
        self.storage.create_dataset(self.dataset_name1)
        self.assertIn(self.dataset_name1, self.storage.datasets)
        self.assertEqual(self.storage.list_datasets(), [self.dataset_name1])
        # Test creating a duplicate dataset
        with self.assertRaises(ValueError):
            self.storage.create_dataset(self.dataset_name1)

    def test_schema_enforcement(self):
        schema = {
            "shape": [3, 10],
            "dtype": str(self.tensor1.dtype),
            "metadata": {"source": "str", "value": "int"},
        }
        self.storage.create_dataset(self.dataset_name1, schema=schema)

        # Valid insert
        self.storage.insert(self.dataset_name1, self.tensor1, {"source": "s", "value": 1})

        # Missing metadata field
        with self.assertRaises(SchemaValidationError):
            self.storage.insert(self.dataset_name1, self.tensor1, {"source": "s"})

        # Wrong dtype
        with self.assertRaises(SchemaValidationError):
            self.storage.insert(self.dataset_name1, self.tensor1.double(), {"source": "s", "value": 1})

        # Wrong shape
        with self.assertRaises(SchemaValidationError):
            self.storage.insert(self.dataset_name1, torch.rand(2, 10), {"source": "s", "value": 1})

    def test_insert_tensor(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        self.assertIsNotNone(record_id1)
        self.assertEqual(len(self.storage.datasets[self.dataset_name1]["tensors"]), 1)
        self.assertEqual(len(self.storage.datasets[self.dataset_name1]["metadata"]), 1)
        
        retrieved_data = self.storage.get_tensor_by_id(self.dataset_name1, record_id1)
        self.assertTrue(torch.equal(retrieved_data["tensor"], self.tensor1))
        self.assertEqual(retrieved_data["metadata"]["source"], self.meta1["source"])
        self.assertIn("record_id", retrieved_data["metadata"]) # Check auto-generated fields
        self.assertIn("timestamp_utc", retrieved_data["metadata"])
        self.assertIn("shape", retrieved_data["metadata"])
        self.assertIn("dtype", retrieved_data["metadata"])

        # Test inserting into non-existent dataset
        with self.assertRaises(DatasetNotFoundError):
            self.storage.insert("non_existent_dataset", self.tensor1)
        
        # Test inserting non-tensor data
        with self.assertRaises(TypeError):
            self.storage.insert(self.dataset_name1, "not a tensor")

    def test_get_dataset(self):
        self.storage.create_dataset(self.dataset_name1)
        self.storage.insert(self.dataset_name1, self.tensor1)
        self.storage.insert(self.dataset_name1, self.tensor2)
        tensors = self.storage.get_dataset(self.dataset_name1)
        self.assertEqual(len(tensors), 2)
        self.assertTrue(torch.equal(tensors[0], self.tensor1))
        self.assertTrue(torch.equal(tensors[1], self.tensor2))

        with self.assertRaises(DatasetNotFoundError):
            self.storage.get_dataset("non_existent_dataset")

    def test_get_dataset_with_metadata(self):
        self.storage.create_dataset(self.dataset_name1)
        self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        self.storage.insert(self.dataset_name1, self.tensor2, self.meta2)
        data_with_meta = self.storage.get_dataset_with_metadata(self.dataset_name1)
        self.assertEqual(len(data_with_meta), 2)
        self.assertTrue(torch.equal(data_with_meta[0]["tensor"], self.tensor1))
        self.assertEqual(data_with_meta[0]["metadata"]["source"], self.meta1["source"])
        self.assertTrue(torch.equal(data_with_meta[1]["tensor"], self.tensor2))
        self.assertEqual(data_with_meta[1]["metadata"]["source"], self.meta2["source"])

        with self.assertRaises(DatasetNotFoundError):
            self.storage.get_dataset_with_metadata("non_existent_dataset")

    def test_get_tensor_by_id(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        retrieved = self.storage.get_tensor_by_id(self.dataset_name1, record_id1)
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(retrieved["tensor"], self.tensor1))

        with self.assertRaises(TensorNotFoundError):
            self.storage.get_tensor_by_id(self.dataset_name1, "non_existent_id")
        with self.assertRaises(DatasetNotFoundError):
            self.storage.get_tensor_by_id("non_existent_dataset", record_id1)

    def test_query(self):
        self.storage.create_dataset(self.dataset_name1)
        self.storage.insert(self.dataset_name1, self.tensor1, self.meta1) # value: 1
        self.storage.insert(self.dataset_name1, self.tensor2, self.meta2) # value: 2
        
        # Query by metadata
        results_meta = self.storage.query(self.dataset_name1, lambda t, m: m.get("value") == 1)
        self.assertEqual(len(results_meta), 1)
        self.assertTrue(torch.equal(results_meta[0]["tensor"], self.tensor1))

        # Query by tensor property (e.g., sum)
        results_tensor = self.storage.query(self.dataset_name1, lambda t, m: t.sum().item() > 5) # tensor1 is rand, tensor2 sum is 6
        if self.tensor1.sum().item() > 5:
             self.assertEqual(len(results_tensor), 2)
        else:
             self.assertEqual(len(results_tensor), 1) # Only tensor2
             self.assertTrue(torch.equal(results_tensor[0]["tensor"], self.tensor2))


        # Query with non-callable
        with self.assertRaises(TypeError):
            self.storage.query(self.dataset_name1, "not a function")
        
        # Query on non-existent dataset
        with self.assertRaises(DatasetNotFoundError):
            self.storage.query("non_existent_dataset", lambda t, m: True)

    def test_list_datasets(self):
        self.assertEqual(self.storage.list_datasets(), [])
        self.storage.create_dataset(self.dataset_name1)
        self.assertEqual(self.storage.list_datasets(), [self.dataset_name1])
        self.storage.create_dataset(self.dataset_name2)
        self.assertCountEqual(self.storage.list_datasets(), [self.dataset_name1, self.dataset_name2])

    def test_update_tensor_metadata(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        
        new_meta = {"status": "processed", "value": 10}
        update_success = self.storage.update_tensor_metadata(self.dataset_name1, record_id, new_meta)
        self.assertTrue(update_success)

        retrieved = self.storage.get_tensor_by_id(self.dataset_name1, record_id)
        self.assertEqual(retrieved["metadata"]["status"], "processed")
        self.assertEqual(retrieved["metadata"]["value"], 10)
        self.assertNotIn("source", retrieved["metadata"])  # Old metadata removed

        # Test updating non-existent record_id
        with self.assertRaises(TensorNotFoundError):
            self.storage.update_tensor_metadata(self.dataset_name1, "fake_id", new_meta)
        # Test updating in non-existent dataset
        with self.assertRaises(DatasetNotFoundError):
            self.storage.update_tensor_metadata("fake_dataset", record_id, new_meta)

        # Test that record_id cannot be changed
        self.storage.update_tensor_metadata(self.dataset_name1, record_id, {"record_id": "new_record_id"})
        retrieved_after_id_change_attempt = self.storage.get_tensor_by_id(self.dataset_name1, record_id)
        self.assertEqual(retrieved_after_id_change_attempt["metadata"]["record_id"], record_id)


    def test_delete_tensor(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        record_id2 = self.storage.insert(self.dataset_name1, self.tensor2, self.meta2)
        self.assertEqual(len(self.storage.get_dataset(self.dataset_name1)), 2)

        delete_success = self.storage.delete_tensor(self.dataset_name1, record_id1)
        self.assertTrue(delete_success)
        self.assertEqual(len(self.storage.get_dataset(self.dataset_name1)), 1)
        with self.assertRaises(TensorNotFoundError):
            self.storage.get_tensor_by_id(self.dataset_name1, record_id1)
        self.assertIsNotNone(self.storage.get_tensor_by_id(self.dataset_name1, record_id2))

        # Test deleting non-existent record_id
        with self.assertRaises(TensorNotFoundError):
            self.storage.delete_tensor(self.dataset_name1, "fake_id")
        # Test deleting from non-existent dataset
        with self.assertRaises(DatasetNotFoundError):
            self.storage.delete_tensor("fake_dataset", record_id2)

    def test_delete_dataset(self):
        self.storage.create_dataset(self.dataset_name1)
        self.storage.insert(self.dataset_name1, self.tensor1)
        self.assertTrue(self.dataset_name1 in self.storage.datasets)
        
        delete_success = self.storage.delete_dataset(self.dataset_name1)
        self.assertTrue(delete_success)
        self.assertNotIn(self.dataset_name1, self.storage.datasets)
        self.assertEqual(self.storage.list_datasets(), [])

        # Test deleting non-existent dataset
        with self.assertRaises(DatasetNotFoundError):
            self.storage.delete_dataset("fake_dataset")

    def test_sample_dataset(self):
        self.storage.create_dataset(self.dataset_name1)
        ids = [self.storage.insert(self.dataset_name1, torch.rand(2), {"i": i}) for i in range(5)]

        # Sample 0
        samples_0 = self.storage.sample_dataset(self.dataset_name1, 0)
        self.assertEqual(len(samples_0), 0)

        # Sample 2
        samples_2 = self.storage.sample_dataset(self.dataset_name1, 2)
        self.assertEqual(len(samples_2), 2)
        # Check if sampled items are from the original dataset
        sample_ids = [s["metadata"]["record_id"] for s in samples_2]
        for sid in sample_ids:
            self.assertIn(sid, ids)

        # Sample more than available
        samples_10 = self.storage.sample_dataset(self.dataset_name1, 10)
        self.assertEqual(len(samples_10), 5) # Should return all items
        self.assertCountEqual([s["metadata"]["record_id"] for s in samples_10], ids)
        
        # Sample from non-existent dataset
        with self.assertRaises(DatasetNotFoundError):
            self.storage.sample_dataset("non_existent_dataset", 1)

    def test_get_records_paginated(self):
        self.storage.create_dataset(self.dataset_name1)
        ids = [self.storage.insert(self.dataset_name1, torch.tensor([i]), {"i": i}) for i in range(5)]

        first_two = self.storage.get_records_paginated(self.dataset_name1, offset=0, limit=2)
        self.assertEqual(len(first_two), 2)
        self.assertEqual(first_two[0]["metadata"]["record_id"], ids[0])

        next_two = self.storage.get_records_paginated(self.dataset_name1, offset=2, limit=2)
        self.assertEqual(len(next_two), 2)
        self.assertEqual(next_two[0]["metadata"]["record_id"], ids[2])

        remaining = self.storage.get_records_paginated(self.dataset_name1, offset=4, limit=2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["metadata"]["record_id"], ids[4])
        
    def test_count_dataset(self):
        self.storage.create_dataset(self.dataset_name1)
        for _ in range(3):
            self.storage.insert(self.dataset_name1, torch.rand(2))

        self.assertEqual(self.storage.count(self.dataset_name1), 3)

        with self.assertRaises(DatasetNotFoundError):
            self.storage.count("non_existent")


class TestTensorStoragePersistent(unittest.TestCase):
    def setUp(self):
        """Set up for persistence testing with a temporary directory."""
        self.test_dir = Path("temp_test_tensor_data_persistent")
        # Clean up if previous run failed
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.storage = TensorStorage(storage_path=str(self.test_dir))
        self.tensor1 = torch.rand(2, 5)
        self.meta1 = {"id": "p_t1", "data_type": "image"}
        self.tensor2 = torch.tensor([10, 20, 30, 40])
        self.meta2 = {"id": "p_t2", "data_type": "audio"}
        self.dataset_name1 = "persistent_ds_one"
        self.dataset_name2 = "persistent_ds_two"

    def tearDown(self):
        """Clean up the temporary directory."""
        del self.storage # Release file handles if any (though torch.save closes)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _get_dataset_file_path(self, dataset_name):
        return self.test_dir / f"{dataset_name}.pt"

    def test_initialization_and_loading(self):
        # 1. Create data with storage1
        self.storage.create_dataset(self.dataset_name1)
        record_id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        
        # Check if file was created
        ds1_file = self._get_dataset_file_path(self.dataset_name1)
        self.assertTrue(ds1_file.exists())

        # 2. Create storage2, pointing to the same path
        storage2 = TensorStorage(storage_path=str(self.test_dir))
        self.assertCountEqual(storage2.list_datasets(), [self.dataset_name1])
        
        retrieved_data = storage2.get_tensor_by_id(self.dataset_name1, record_id1)
        self.assertIsNotNone(retrieved_data)
        self.assertTrue(torch.equal(retrieved_data["tensor"], self.tensor1))
        self.assertEqual(retrieved_data["metadata"]["id"], self.meta1["id"])

    def test_insert_persists(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        ds1_file = self._get_dataset_file_path(self.dataset_name1)
        
        # Load data directly to check file content (simplified check)
        # In a more complex scenario, you might load the file and check its structure
        self.assertTrue(ds1_file.exists())
        
        # Create a new instance and check if data is loaded
        storage2 = TensorStorage(storage_path=str(self.test_dir))
        retrieved = storage2.get_tensor_by_id(self.dataset_name1, record_id1)
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(retrieved["tensor"], self.tensor1))

    def test_update_tensor_metadata_persists(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        
        new_meta = {"status": "processed_persistent", "data_type": "image_updated"}
        self.storage.update_tensor_metadata(self.dataset_name1, record_id, new_meta)

        # New instance to check persistence
        storage2 = TensorStorage(storage_path=str(self.test_dir))
        retrieved = storage2.get_tensor_by_id(self.dataset_name1, record_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["metadata"]["status"], "processed_persistent")
        self.assertEqual(retrieved["metadata"]["data_type"], "image_updated")  # Check overwrite
        self.assertNotIn("id", retrieved["metadata"])  # Old metadata removed

    def test_delete_tensor_persists(self):
        self.storage.create_dataset(self.dataset_name1)
        record_id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)
        record_id2 = self.storage.insert(self.dataset_name1, self.tensor2, self.meta2)

        self.storage.delete_tensor(self.dataset_name1, record_id1)

        # New instance
        storage2 = TensorStorage(storage_path=str(self.test_dir))
        with self.assertRaises(TensorNotFoundError):
            storage2.get_tensor_by_id(self.dataset_name1, record_id1)
        self.assertIsNotNone(storage2.get_tensor_by_id(self.dataset_name1, record_id2))
        self.assertEqual(len(storage2.get_dataset(self.dataset_name1)), 1)


    def test_delete_dataset_persists(self):
        self.storage.create_dataset(self.dataset_name1)
        self.storage.insert(self.dataset_name1, self.tensor1)
        ds1_file = self._get_dataset_file_path(self.dataset_name1)
        self.assertTrue(ds1_file.exists())

        self.storage.delete_dataset(self.dataset_name1)
        self.assertFalse(ds1_file.exists())
        self.assertNotIn(self.dataset_name1, self.storage.list_datasets())

        # New instance
        storage2 = TensorStorage(storage_path=str(self.test_dir))
        self.assertNotIn(self.dataset_name1, storage2.list_datasets())
        self.assertEqual(len(storage2.list_datasets()), 0)


    def test_operations_reflect_across_instances(self):
        # storage1 creates and inserts
        self.storage.create_dataset(self.dataset_name1)
        id1 = self.storage.insert(self.dataset_name1, self.tensor1, self.meta1)

        # storage2 loads and modifies
        storage2 = TensorStorage(storage_path=str(self.test_dir))
        storage2.update_tensor_metadata(self.dataset_name1, id1, {"new_key": "new_value"})
        id2 = storage2.insert(self.dataset_name1, self.tensor2, self.meta2)

        # storage3 (or original storage reloaded/re-checked) should see all changes
        storage3 = TensorStorage(storage_path=str(self.test_dir))
        self.assertCountEqual(storage3.list_datasets(), [self.dataset_name1])
        
        meta_check1 = storage3.get_tensor_by_id(self.dataset_name1, id1)["metadata"]
        self.assertEqual(meta_check1["new_key"], "new_value")
        self.assertNotIn("id", meta_check1)

        meta_check2 = storage3.get_tensor_by_id(self.dataset_name1, id2)["metadata"]
        self.assertEqual(meta_check2["id"], self.meta2["id"])
        
        self.assertEqual(len(storage3.get_dataset(self.dataset_name1)), 2)

    def test_create_dataset_empty_persists(self):
        self.storage.create_dataset(self.dataset_name2)
        ds2_file = self._get_dataset_file_path(self.dataset_name2)
        self.assertTrue(ds2_file.exists())

        storage2 = TensorStorage(storage_path=str(self.test_dir))
        self.assertIn(self.dataset_name2, storage2.list_datasets())
        self.assertEqual(len(storage2.get_dataset(self.dataset_name2)), 0)

    def test_fallback_to_in_memory_on_path_error(self):
        # This is hard to test perfectly as it involves filesystem permissions
        # We can simulate by trying to create a storage with a path that's a file.
        # Create a dummy file
        dummy_file_path = self.test_dir / "i_am_a_file.txt"
        with open(dummy_file_path, "w") as f:
            f.write("hello")
        
        # Suppress logging for this specific test to avoid console noise from expected error
        original_logging_level = logging.getLogger().getEffectiveLevel()
        logging.disable(logging.ERROR)
        
        try:
            storage_bad_path = TensorStorage(storage_path=str(dummy_file_path))
            self.assertIsNone(storage_bad_path.storage_path) # Should have fallen back
            storage_bad_path.create_dataset("wont_persist")
            # Check that no .pt file was created inside the "i_am_a_file.txt" path (which would be an error)
            self.assertFalse((dummy_file_path / "wont_persist.pt").exists())
        finally:
            logging.disable(original_logging_level) # Restore logging
            os.remove(dummy_file_path)


if __name__ == '__main__':
    # Import logging here only if you want to see logs during test runs via `python test_tensor_storage.py`
    # For normal unittest runs (e.g. `python -m unittest test_tensor_storage`), logging is often suppressed by the runner.
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()

# Note: To run these tests, ensure tensor_storage.py is in the same directory or in PYTHONPATH
# Example: python -m unittest test_tensor_storage.py
# Or, if tensor_storage.py is in a parent/src directory:
# PYTHONPATH=. python -m unittest tests.test_tensor_storage (if tests are in a tests/ subdir)
