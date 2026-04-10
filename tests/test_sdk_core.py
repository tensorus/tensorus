"""
SDK Core Functionality Tests

This module tests the critical SDK functionality to ensure
the unified Tensorus interface works correctly.
"""

import pytest
import numpy as np
import torch
from uuid import UUID

from tensorus import Tensorus, TensorWrapper
from tensorus.tensor_storage import TensorNotFoundError, DatasetNotFoundError


class TestSDKInitialization:
    """Test SDK initialization scenarios."""
    
    def test_basic_initialization(self):
        """Test basic SDK initialization with minimal options."""
        ts = Tensorus(
            enable_nql=False,
            enable_embeddings=False,
            enable_vector_search=False
        )
        assert ts is not None
        assert ts.storage is not None
    
    def test_initialization_with_all_features(self):
        """Test SDK initialization with all features enabled."""
        ts = Tensorus(
            enable_nql=True,
            enable_embeddings=False,  # Requires heavy dependencies
            enable_vector_search=True
        )
        assert ts is not None


class TestDatasetOperations:
    """Test dataset creation and management."""
    
    @pytest.fixture
    def sdk(self):
        """Provide a clean SDK instance."""
        return Tensorus(
            enable_nql=False,
            enable_embeddings=False,
            enable_vector_search=False
        )
    
    def test_create_dataset(self, sdk):
        """Test dataset creation."""
        sdk.create_dataset("test_dataset")
        datasets = sdk.list_datasets()
        assert "test_dataset" in datasets
    
    def test_delete_dataset(self, sdk):
        """Test dataset deletion."""
        sdk.create_dataset("temp_dataset")
        assert "temp_dataset" in sdk.list_datasets()
        
        sdk.delete_dataset("temp_dataset")
        assert "temp_dataset" not in sdk.list_datasets()


class TestTensorOperations:
    """Test tensor CRUD operations."""
    
    @pytest.fixture
    def sdk(self):
        """Provide a clean SDK instance with a test dataset."""
        ts = Tensorus(
            enable_nql=False,
            enable_embeddings=False,
            enable_vector_search=False
        )
        ts.create_dataset("test_dataset")
        return ts
    
    def test_create_tensor_from_list(self, sdk):
        """Test creating a tensor from a Python list."""
        tensor = sdk.create_tensor(
            [[1, 2], [3, 4]],
            name="list_tensor",
            dataset="test_dataset"
        )
        assert isinstance(tensor, TensorWrapper)
        assert tensor.shape == (2, 2)
        assert tensor.name == "list_tensor"
    
    def test_create_tensor_from_numpy(self, sdk):
        """Test creating a tensor from numpy array."""
        np_array = np.random.rand(3, 4)
        tensor = sdk.create_tensor(
            np_array,
            name="numpy_tensor",
            dataset="test_dataset"
        )
        assert tensor.shape == (3, 4)
    
    def test_create_tensor_from_torch(self, sdk):
        """Test creating a tensor from PyTorch tensor."""
        torch_tensor = torch.randn(5, 5)
        tensor = sdk.create_tensor(
            torch_tensor,
            name="torch_tensor",
            dataset="test_dataset"
        )
        assert tensor.shape == (5, 5)
    
    def test_create_tensor_with_metadata(self, sdk):
        """Test creating a tensor with custom metadata."""
        metadata = {"source": "test", "version": 1.0}
        tensor = sdk.create_tensor(
            [[1, 2]],
            name="meta_tensor",
            metadata=metadata,
            description="Test tensor with metadata",
            dataset="test_dataset"
        )
        assert tensor.metadata.get("source") == "test"
        assert tensor.metadata.get("version") == 1.0
        assert tensor.description == "Test tensor with metadata"
    
    def test_get_tensor_by_id(self, sdk):
        """Test retrieving a tensor by its ID."""
        original = sdk.create_tensor(
            [[1, 2, 3]],
            name="original",
            dataset="test_dataset"
        )
        
        retrieved = sdk.get_tensor(original.id, dataset="test_dataset")
        assert retrieved is not None
        assert retrieved.name == "original"
        assert torch.equal(original._data, retrieved._data)
    
    def test_get_tensor_by_string_id(self, sdk):
        """Test retrieving a tensor using string ID."""
        original = sdk.create_tensor(
            [[1, 2, 3]],
            name="string_id_test",
            dataset="test_dataset"
        )
        
        # Convert UUID to string
        string_id = str(original.id)
        retrieved = sdk.get_tensor(string_id, dataset="test_dataset")
        assert retrieved is not None
        assert retrieved.name == "string_id_test"
    
    def test_get_tensor_not_found(self, sdk):
        """Test that getting a non-existent tensor raises error."""
        with pytest.raises(TensorNotFoundError):
            sdk.get_tensor("non-existent-id", dataset="test_dataset")
    
    def test_list_tensors(self, sdk):
        """Test listing all tensors in a dataset."""
        # Create multiple tensors
        sdk.create_tensor([[1]], name="t1", dataset="test_dataset")
        sdk.create_tensor([[2]], name="t2", dataset="test_dataset")
        sdk.create_tensor([[3]], name="t3", dataset="test_dataset")
        
        tensors = sdk.list_tensors("test_dataset")
        assert len(tensors) == 3
    
    def test_delete_tensor(self, sdk):
        """Test deleting a tensor."""
        tensor = sdk.create_tensor(
            [[1, 2]],
            name="to_delete",
            dataset="test_dataset"
        )
        
        # Verify it exists
        assert len(sdk.list_tensors("test_dataset")) == 1
        
        # Delete it
        result = sdk.delete_tensor(tensor.id, dataset="test_dataset")
        assert result is True
        
        # Verify it's gone
        assert len(sdk.list_tensors("test_dataset")) == 0


class TestMetadataSearch:
    """Test metadata search functionality."""
    
    @pytest.fixture
    def sdk_with_data(self):
        """Provide SDK with pre-populated test data."""
        ts = Tensorus(
            enable_nql=False,
            enable_embeddings=False,
            enable_vector_search=False
        )
        ts.create_dataset("search_test")
        
        # Create tensors with various metadata
        ts.create_tensor(
            [[1]], name="tagged1",
            metadata={"tags": ["example", "test"], "category": "A"},
            dataset="search_test"
        )
        ts.create_tensor(
            [[2]], name="tagged2",
            metadata={"tags": ["example", "production"], "category": "B"},
            dataset="search_test"
        )
        ts.create_tensor(
            [[3]], name="untagged",
            metadata={"category": "A"},
            dataset="search_test"
        )
        return ts
    
    def test_search_by_tag(self, sdk_with_data):
        """Test searching tensors by tag."""
        results = sdk_with_data.search_metadata(
            {"tags": "example"},
            dataset="search_test"
        )
        assert len(results) == 2
    
    def test_search_by_exact_match(self, sdk_with_data):
        """Test searching tensors by exact metadata match."""
        results = sdk_with_data.search_metadata(
            {"category": "A"},
            dataset="search_test"
        )
        assert len(results) == 2
    
    def test_search_by_multiple_criteria(self, sdk_with_data):
        """Test searching with multiple metadata criteria."""
        results = sdk_with_data.search_metadata(
            {"tags": "example", "category": "A"},
            dataset="search_test"
        )
        assert len(results) == 1
        assert results[0].name == "tagged1"
    
    def test_search_no_results(self, sdk_with_data):
        """Test search that returns no results."""
        results = sdk_with_data.search_metadata(
            {"nonexistent_key": "value"},
            dataset="search_test"
        )
        assert len(results) == 0
    
    def test_search_nonexistent_dataset(self, sdk_with_data):
        """Test search on non-existent dataset returns empty list."""
        results = sdk_with_data.search_metadata(
            {"tags": "example"},
            dataset="nonexistent_dataset"
        )
        assert len(results) == 0


class TestTensorMathOperations:
    """Test tensor mathematical operations through SDK."""
    
    @pytest.fixture
    def sdk(self):
        """Provide a clean SDK instance."""
        return Tensorus(
            enable_nql=False,
            enable_embeddings=False,
            enable_vector_search=False
        )
    
    def test_matmul(self, sdk):
        """Test matrix multiplication."""
        a = sdk.create_tensor([[1, 2], [3, 4]], name="a")
        b = sdk.create_tensor([[5, 6], [7, 8]], name="b")
        
        result = sdk.matmul(a, b)
        expected = torch.tensor([[19, 22], [43, 50]])
        assert torch.equal(result._data, expected)
    
    def test_matmul_with_torch_tensors(self, sdk):
        """Test matrix multiplication with raw torch tensors."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        result = sdk.matmul(a, b)
        assert result.shape == (2, 2)
    
    def test_transpose(self, sdk):
        """Test tensor transpose."""
        tensor = sdk.create_tensor([[1, 2, 3], [4, 5, 6]], name="to_transpose")
        
        result = sdk.transpose(tensor)
        assert result.shape == (3, 2)
    
    def test_tensor_wrapper_arithmetic(self, sdk):
        """Test TensorWrapper arithmetic operations."""
        a = sdk.create_tensor([[1, 2], [3, 4]], name="a")
        b = sdk.create_tensor([[5, 6], [7, 8]], name="b")
        
        # Addition
        add_result = a + b
        assert add_result.shape == (2, 2)
        
        # Subtraction
        sub_result = a - b
        assert sub_result.shape == (2, 2)
        
        # Multiplication
        mul_result = a * b
        assert mul_result.shape == (2, 2)


class TestVectorDatabaseOperations:
    """Test vector database functionality."""
    
    @pytest.fixture
    def sdk(self):
        """Provide SDK with vector search enabled."""
        return Tensorus(
            enable_nql=False,
            enable_embeddings=False,
            enable_vector_search=True
        )
    
    def test_create_vector_index(self, sdk):
        """Test creating a vector index."""
        sdk.create_index("test_vectors", dimensions=128)
        assert sdk.index_exists("test_vectors")
    
    def test_add_and_search_vectors(self, sdk):
        """Test adding vectors and searching."""
        # Create index
        sdk.create_index("search_test", dimensions=64)
        
        # Add vectors
        vectors = np.random.rand(10, 64).astype(np.float32)
        ids = [f"vec_{i}" for i in range(10)]
        sdk.add_vectors("search_test", ids, vectors)
        
        # Search
        query = np.random.rand(64).astype(np.float32)
        results = sdk.search_vectors("search_test", query, k=3)
        
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
