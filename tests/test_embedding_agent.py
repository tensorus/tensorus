# test_embedding_agent.py
"""
Tests for EmbeddingAgent class and embedding functionality.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from tensorus.embedding_agent import EmbeddingAgent
from tensorus.tensor_storage import TensorStorage


class TestEmbeddingAgent:
    """Test cases for EmbeddingAgent class."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def tensor_storage(self, temp_storage_path):
        """Create TensorStorage instance for testing."""
        return TensorStorage(storage_path=temp_storage_path)

    @pytest.fixture
    def embedding_agent(self, tensor_storage):
        """Create EmbeddingAgent instance for testing."""
        return EmbeddingAgent(tensor_storage)

    def test_initialization(self, embedding_agent):
        """Test EmbeddingAgent initialization."""
        assert embedding_agent.tensor_storage is not None
        assert embedding_agent.model_cache == {}
        assert embedding_agent.embedding_cache == {}

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_text_sentence_transformers(self, mock_st, embedding_agent):
        """Test text encoding with sentence-transformers."""
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        
        text = "Hello world"
        embeddings = embedding_agent.encode_text(text, model_name="all-MiniLM-L6-v2")
        
        assert embeddings.shape == (1, 3)
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")
        mock_model.encode.assert_called_once()

    @patch('openai.embeddings.create')
    def test_encode_text_openai(self, mock_openai, embedding_agent):
        """Test text encoding with OpenAI."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value = mock_response
        
        text = "Hello world"
        embeddings = embedding_agent.encode_text(text, model_name="text-embedding-ada-002")
        
        assert embeddings.shape == (1, 3)

    def test_store_embeddings(self, embedding_agent, tensor_storage):
        """Test storing embeddings in tensor storage."""
        embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        texts = ["Hello", "World"]
        
        result = embedding_agent.store_embeddings(
            embeddings=embeddings,
            texts=texts,
            dataset_name="test_embeddings",
            model_name="test-model"
        )
        
        assert result["success"] is True
        assert result["stored_count"] == 2
        
        # Verify data was stored
        records = tensor_storage.get_dataset_with_metadata("test_embeddings")
        assert len(records) == 2

    def test_similarity_search(self, embedding_agent, tensor_storage):
        """Test similarity search functionality."""
        # First store some embeddings
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        texts = ["first", "second", "third"]
        
        embedding_agent.store_embeddings(
            embeddings=embeddings,
            texts=texts,
            dataset_name="search_test",
            model_name="test-model"
        )
        
        # Search for similar vectors
        query_vector = torch.tensor([1.0, 0.1, 0.0])
        results = embedding_agent.similarity_search(
            query_vector=query_vector,
            dataset_name="search_test",
            k=2
        )
        
        assert len(results) <= 2
        assert all("similarity" in result for result in results)
        assert all("metadata" in result for result in results)

    def test_model_caching(self, embedding_agent):
        """Test that models are cached properly."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_model
            
            # First call should create model
            embedding_agent.encode_text("test", model_name="test-model")
            assert "test-model" in embedding_agent.model_cache
            
            # Second call should use cached model
            embedding_agent.encode_text("test2", model_name="test-model")
            mock_st.assert_called_once()  # Should only be called once

    def test_embedding_caching(self, embedding_agent):
        """Test that embeddings are cached properly."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_model
            
            text = "test text"
            model_name = "test-model"
            
            # First call should compute embedding
            embedding1 = embedding_agent.encode_text(text, model_name=model_name)
            
            # Second call should use cached embedding
            embedding2 = embedding_agent.encode_text(text, model_name=model_name)
            
            assert torch.equal(embedding1, embedding2)
            mock_model.encode.assert_called_once()  # Should only compute once

    def test_get_embedding_stats(self, embedding_agent, tensor_storage):
        """Test embedding statistics retrieval."""
        # Store some test embeddings
        embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        texts = ["a", "b", "c"]
        
        embedding_agent.store_embeddings(
            embeddings=embeddings,
            texts=texts,
            dataset_name="stats_test",
            model_name="test-model"
        )
        
        stats = embedding_agent.get_embedding_stats("stats_test")
        
        assert stats["total_embeddings"] == 3
        assert stats["dimension"] == 2
        assert "models_used" in stats
        assert "test-model" in stats["models_used"]

    def test_invalid_model_name(self, embedding_agent):
        """Test handling of invalid model names."""
        with pytest.raises(ValueError):
            embedding_agent.encode_text("test", model_name="invalid-model-name")

    def test_empty_text_list(self, embedding_agent):
        """Test handling of empty text list."""
        result = embedding_agent.store_embeddings(
            embeddings=torch.tensor([]).reshape(0, 3),
            texts=[],
            dataset_name="empty_test",
            model_name="test-model"
        )
        
        assert result["success"] is True
        assert result["stored_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
