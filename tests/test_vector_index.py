# test_vector_index.py
"""
Tests for VectorIndex and VectorIndexManager classes.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from tensorus.vector_index import VectorIndex, VectorIndexManager
from tensorus.tensor_storage import TensorStorage


class TestVectorIndex:
    """Test cases for VectorIndex class."""

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        return torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ], dtype=torch.float32)

    @pytest.fixture
    def sample_ids(self):
        """Create sample vector IDs."""
        return ["vec1", "vec2", "vec3", "vec4"]

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return [
            {"text": "first vector", "category": "A"},
            {"text": "second vector", "category": "B"},
            {"text": "third vector", "category": "A"},
            {"text": "fourth vector", "category": "C"}
        ]

    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_flat_index_creation(self, mock_faiss):
        """Test creation of flat index."""
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        index = VectorIndex(dimension=128, index_type="flat", metric="cosine")
        
        assert index.dimension == 128
        assert index.index_type == "flat"
        assert index.metric == "cosine"
        mock_faiss.assert_called_once_with(128)

    @patch('tensorus.vector_index.FAISS_AVAILABLE', False)
    def test_faiss_not_available(self):
        """Test error when FAISS is not available."""
        with pytest.raises(ImportError):
            VectorIndex(dimension=128)

    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    @patch('faiss.normalize_L2')
    def test_add_vectors(self, mock_normalize, mock_faiss, sample_vectors, sample_ids, sample_metadata):
        """Test adding vectors to index."""
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        index = VectorIndex(dimension=3, index_type="flat", metric="cosine")
        index.add_vectors(sample_vectors, sample_ids, sample_metadata)
        
        assert len(index.vector_ids) == 4
        assert index.vector_ids == sample_ids
        mock_normalize.assert_called_once()
        mock_index.add.assert_called_once()

    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_search_vectors(self, mock_faiss, sample_vectors, sample_ids):
        """Test vector search functionality."""
        mock_index = Mock()
        mock_index.search.return_value = (
            [[0.9, 0.5]], [[0, 2]]  # distances, indices
        )
        mock_faiss.return_value = mock_index
        
        index = VectorIndex(dimension=3, index_type="flat", metric="cosine")
        index.add_vectors(sample_vectors, sample_ids)
        
        query = torch.tensor([[1.0, 0.1, 0.0]])
        results = index.search(query, k=2)
        
        assert len(results) == 1  # One query
        assert len(results[0]) == 2  # Two results
        assert results[0][0]["vector_id"] == "vec1"

    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    @patch('faiss.write_index')
    def test_save_index(self, mock_write, mock_faiss, tmp_path):
        """Test saving index to disk."""
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        index = VectorIndex(dimension=3, index_type="flat", metric="cosine")
        save_path = tmp_path / "test_index"
        
        index.save(save_path)
        
        mock_write.assert_called_once()
        assert (save_path.with_suffix('.meta')).exists()

    def test_get_stats(self):
        """Test getting index statistics."""
        with patch('tensorus.vector_index.FAISS_AVAILABLE', True), \
             patch('faiss.IndexFlatIP'):
            
            index = VectorIndex(dimension=128, index_type="flat", metric="cosine")
            stats = index.get_stats()
            
            assert stats["dimension"] == 128
            assert stats["index_type"] == "flat"
            assert stats["metric"] == "cosine"
            assert stats["total_vectors"] == 0


class TestVectorIndexManager:
    """Test cases for VectorIndexManager class."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def tensor_storage(self, temp_storage_path):
        """Create TensorStorage instance for testing."""
        storage = TensorStorage(storage_path=temp_storage_path)
        
        # Add some test embedding data
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        for i, embedding in enumerate(embeddings):
            metadata = {
                "record_id": f"emb_{i}",
                "is_embedding": True,
                "source_text": f"text {i}",
                "embedding_model": "test-model"
            }
            storage.insert("test_embeddings", embedding, metadata)
        
        return storage

    @pytest.fixture
    def index_manager(self, tensor_storage, temp_storage_path):
        """Create VectorIndexManager instance for testing."""
        index_path = Path(temp_storage_path) / "indexes"
        return VectorIndexManager(tensor_storage, str(index_path))

    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_build_index(self, mock_faiss, index_manager):
        """Test building an index from dataset."""
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        with patch.object(index_manager, 'tensor_storage') as mock_storage:
            mock_storage.get_dataset_with_metadata.return_value = [
                {
                    "tensor": torch.tensor([1.0, 0.0, 0.0]),
                    "metadata": {"record_id": "1", "is_embedding": True}
                },
                {
                    "tensor": torch.tensor([0.0, 1.0, 0.0]),
                    "metadata": {"record_id": "2", "is_embedding": True}
                }
            ]
            
            index = index_manager.build_index("test_dataset", index_type="flat")
            
            assert "test_dataset_flat" in index_manager.indexes
            mock_storage.get_dataset_with_metadata.assert_called_once_with("test_dataset")

    def test_list_indexes_empty(self, index_manager):
        """Test listing indexes when none exist."""
        indexes = index_manager.list_indexes()
        assert indexes == []

    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_delete_index(self, mock_faiss, index_manager, tmp_path):
        """Test deleting an index."""
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        # Create a mock index file
        index_path = index_manager.index_storage_path / "test_index"
        index_path.with_suffix('.faiss').touch()
        index_path.with_suffix('.meta').touch()
        
        success = index_manager.delete_index("test_index")
        
        assert success is True
        assert not index_path.with_suffix('.faiss').exists()
        assert not index_path.with_suffix('.meta').exists()

    def test_search_nonexistent_index(self, index_manager):
        """Test searching a non-existent index."""
        with pytest.raises(FileNotFoundError):
            index_manager.search_index("nonexistent", "query text")


if __name__ == "__main__":
    pytest.main([__file__])
