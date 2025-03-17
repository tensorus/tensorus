import sys
import os
import unittest
import numpy as np
import tempfile
import shutil

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor_data import TensorStorage
from tensor_indexer import TensorIndexer
from tensor_processor import TensorProcessor
from tensor_database import TensorDatabase


class TestTensorStorage(unittest.TestCase):
    """Test the TensorStorage class."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_storage.h5")
        self.storage = TensorStorage(filename=self.storage_path)
        
    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)
        
    def test_save_load_tensor(self):
        """Test saving and loading a tensor."""
        # Create a test tensor
        tensor = np.random.rand(3, 4, 5)
        metadata = {"name": "test_tensor", "tags": ["test", "example"]}
        
        # Save the tensor
        tensor_id = self.storage.save_tensor(tensor, metadata)
        
        # Verify the tensor_id is a string
        self.assertIsInstance(tensor_id, str)
        
        # Load the tensor
        loaded_tensor, loaded_metadata = self.storage.load_tensor(tensor_id)
        
        # Verify the loaded tensor matches the original
        np.testing.assert_array_equal(tensor, loaded_tensor)
        
        # Verify the metadata
        self.assertEqual(loaded_metadata["name"], metadata["name"])
        self.assertEqual(loaded_metadata["tags"], metadata["tags"])
        
    def test_update_tensor(self):
        """Test updating a tensor."""
        # Create and save a test tensor
        tensor = np.random.rand(3, 4, 5)
        metadata = {"name": "original"}
        tensor_id = self.storage.save_tensor(tensor, metadata)
        
        # Update with new tensor and metadata
        new_tensor = np.random.rand(3, 4, 5)
        new_metadata = {"name": "updated", "version": 2}
        success = self.storage.update_tensor(tensor_id, new_tensor, new_metadata)
        
        # Verify update was successful
        self.assertTrue(success)
        
        # Load the updated tensor
        loaded_tensor, loaded_metadata = self.storage.load_tensor(tensor_id)
        
        # Verify the loaded tensor matches the new tensor
        np.testing.assert_array_equal(new_tensor, loaded_tensor)
        
        # Verify the updated metadata
        self.assertEqual(loaded_metadata["name"], new_metadata["name"])
        self.assertEqual(loaded_metadata["version"], new_metadata["version"])
        
    def test_delete_tensor(self):
        """Test deleting a tensor."""
        # Create and save a test tensor
        tensor = np.random.rand(3, 4, 5)
        tensor_id = self.storage.save_tensor(tensor)
        
        # Delete the tensor
        success = self.storage.delete_tensor(tensor_id)
        
        # Verify deletion was successful
        self.assertTrue(success)
        
        # Verify the tensor no longer exists
        with self.assertRaises(KeyError):
            self.storage.load_tensor(tensor_id)
            
    def test_list_tensors(self):
        """Test listing all tensors."""
        # Create and save multiple tensors
        tensors = []
        tensor_ids = []
        
        for i in range(3):
            tensor = np.random.rand(2, 3, 4)
            metadata = {"index": i}
            tensor_id = self.storage.save_tensor(tensor, metadata)
            tensors.append(tensor)
            tensor_ids.append(tensor_id)
            
        # List all tensors
        tensor_list = self.storage.list_tensors()
        
        # Verify the list has the correct length
        self.assertEqual(len(tensor_list), 3)
        
        # Verify each tensor is in the list
        listed_ids = [t["id"] for t in tensor_list]
        for tensor_id in tensor_ids:
            self.assertIn(tensor_id, listed_ids)


class TestTensorIndexer(unittest.TestCase):
    """Test the TensorIndexer class."""
    
    def setUp(self):
        """Initialize the indexer."""
        self.dimension = 12  # 3x4 flattened
        self.indexer = TensorIndexer(dimension=self.dimension)
        
    def test_add_search_tensor(self):
        """Test adding tensors and searching."""
        # Create test tensors
        tensor1 = np.random.rand(3, 4).flatten()
        tensor2 = np.random.rand(3, 4).flatten()
        tensor3 = np.ones((3, 4)).flatten()  # This will be most similar to the query
        
        # Add tensors to the index
        self.indexer.add_tensor(tensor1, "tensor1")
        self.indexer.add_tensor(tensor2, "tensor2")
        self.indexer.add_tensor(tensor3, "tensor3")
        
        # Create a query tensor (similar to tensor3)
        query = np.ones((3, 4)).flatten() + np.random.rand(3, 4).flatten() * 0.1
        
        # Search for similar tensors
        results, distances = self.indexer.search_tensor(query, k=3)
        
        # Verify tensor3 is the most similar
        self.assertEqual(results[0], "tensor3")


class TestTensorProcessor(unittest.TestCase):
    """Test the TensorProcessor class."""
    
    def setUp(self):
        """Initialize the processor."""
        self.processor = TensorProcessor()
        
    def test_basic_operations(self):
        """Test basic tensor operations."""
        # Create test tensors
        t1 = np.array([[1, 2], [3, 4]])
        t2 = np.array([[5, 6], [7, 8]])
        
        # Test addition
        result = self.processor.add(t1, t2)
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(result, expected)
        
        # Test subtraction
        result = self.processor.subtract(t1, t2)
        expected = np.array([[-4, -4], [-4, -4]])
        np.testing.assert_array_equal(result, expected)
        
        # Test multiplication
        result = self.processor.multiply(t1, t2)
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_array_equal(result, expected)
        
        # Test matrix multiplication
        result = self.processor.matmul(t1, t2)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)
        
    def test_reshape(self):
        """Test tensor reshaping."""
        tensor = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Reshape to 3x2
        result = self.processor.reshape(tensor, (3, 2))
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result, expected)
        
    def test_transpose(self):
        """Test tensor transposition."""
        tensor = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Transpose
        result = self.processor.transpose(tensor)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)


class TestTensorDatabase(unittest.TestCase):
    """Test the TensorDatabase class."""
    
    def setUp(self):
        """Create a temporary directory and initialize the database."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.test_dir, "test_db.h5")
        self.index_path = os.path.join(self.test_dir, "test_index.pkl")
        self.db = TensorDatabase(
            storage_path=self.storage_path,
            index_path=self.index_path
        )
        
    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)
        
    def test_save_get(self):
        """Test saving and retrieving a tensor."""
        # Create a test tensor
        tensor = np.random.rand(3, 4, 5)
        metadata = {"name": "test_tensor"}
        
        # Save the tensor
        tensor_id = self.db.save(tensor, metadata)
        
        # Retrieve the tensor
        retrieved_tensor, retrieved_metadata = self.db.get(tensor_id)
        
        # Verify the tensor
        np.testing.assert_array_equal(tensor, retrieved_tensor)
        
        # Verify the metadata
        self.assertEqual(retrieved_metadata["name"], metadata["name"])
        
    def test_search_similar(self):
        """Test searching for similar tensors."""
        # Create and save test tensors
        tensor1 = np.random.rand(3, 4)
        tensor2 = np.random.rand(3, 4)
        tensor3 = np.ones((3, 4))  # This will be most similar to the query
        
        # Save tensors
        id1 = self.db.save(tensor1, {"name": "tensor1"})
        id2 = self.db.save(tensor2, {"name": "tensor2"})
        id3 = self.db.save(tensor3, {"name": "tensor3"})
        
        # Create a query tensor (similar to tensor3)
        query = np.ones((3, 4)) + np.random.rand(3, 4) * 0.1
        
        # Search for similar tensors
        results = self.db.search_similar(query, k=3)
        
        # Verify tensor3 is the most similar
        self.assertEqual(results[0]["tensor_id"], id3)
        
    def test_process(self):
        """Test processing tensors with operations."""
        # Create and save test tensors
        t1 = np.array([[1, 2], [3, 4]])
        t2 = np.array([[5, 6], [7, 8]])
        
        id1 = self.db.save(t1)
        id2 = self.db.save(t2)
        
        # Process with addition
        result = self.db.process("add", [id1, id2])
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(result, expected)
        
        # Process with reshaping
        result = self.db.process("reshape", [id1], new_shape=(4,))
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main() 