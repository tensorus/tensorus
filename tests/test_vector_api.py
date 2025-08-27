# test_vector_api.py
"""
Tests for vector database API endpoints.
"""

import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import shutil
from tensorus.api import app
from tensorus.tensor_storage import TensorStorage


class TestVectorAPI:
    """Test cases for vector database API endpoints."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def api_key(self):
        """Mock API key for testing."""
        return "test-api-key"

    @pytest.fixture
    def auth_headers(self, api_key):
        """Create authorization headers."""
        return {"Authorization": f"Bearer {api_key}"}

    @patch('tensorus.api.security.verify_api_key')
    def test_embed_text_endpoint(self, mock_verify, client, auth_headers):
        """Test text embedding endpoint."""
        mock_verify.return_value = "test-key"
        
        with patch('tensorus.api.routers.vector.get_embedding_agent') as mock_get_agent:
            mock_agent = Mock()
            mock_agent.encode_text.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_agent.store_embeddings.return_value = {
                "success": True,
                "stored_count": 1,
                "dataset_name": "test_embeddings"
            }
            mock_get_agent.return_value = mock_agent
            
            response = client.post(
                "/vector/embed",
                json={
                    "texts": ["Hello world"],
                    "dataset_name": "test_embeddings",
                    "model_name": "all-MiniLM-L6-v2",
                    "store_embeddings": True
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["stored_count"] == 1

    @patch('tensorus.api.security.verify_api_key')
    def test_similarity_search_endpoint(self, mock_verify, client, auth_headers):
        """Test similarity search endpoint."""
        mock_verify.return_value = "test-key"
        
        with patch('tensorus.api.routers.vector.get_embedding_agent') as mock_get_agent:
            mock_agent = Mock()
            mock_agent.encode_text.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_agent.similarity_search.return_value = [
                {
                    "similarity": 0.95,
                    "metadata": {"source_text": "Hello world", "record_id": "1"}
                }
            ]
            mock_get_agent.return_value = mock_agent
            
            response = client.post(
                "/vector/search",
                json={
                    "query": "Hello",
                    "dataset_name": "test_embeddings",
                    "k": 5,
                    "model_name": "all-MiniLM-L6-v2"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 1
            assert data["results"][0]["similarity"] == 0.95

    @patch('tensorus.api.security.verify_api_key')
    def test_hybrid_search_endpoint(self, mock_verify, client, auth_headers):
        """Test hybrid search endpoint."""
        mock_verify.return_value = "test-key"
        
        with patch('tensorus.api.routers.vector.get_embedding_agent') as mock_get_agent, \
             patch('tensorus.api.routers.vector.get_tensor_storage') as mock_get_storage:
            
            mock_agent = Mock()
            mock_agent.encode_text.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_agent.similarity_search.return_value = [
                {
                    "similarity": 0.95,
                    "metadata": {"source_text": "Hello world", "category": "greeting"}
                }
            ]
            mock_get_agent.return_value = mock_agent
            
            mock_storage = Mock()
            mock_storage.query.return_value = [
                {"metadata": {"source_text": "Hello world", "category": "greeting"}}
            ]
            mock_get_storage.return_value = mock_storage
            
            response = client.post(
                "/vector/hybrid-search",
                json={
                    "query": "Hello",
                    "dataset_name": "test_embeddings",
                    "k": 5,
                    "metadata_filters": {"category": "greeting"},
                    "vector_weight": 0.7,
                    "metadata_weight": 0.3
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @patch('tensorus.api.security.verify_api_key')
    @patch('tensorus.vector_index.FAISS_AVAILABLE', True)
    @patch('faiss.IndexFlatIP')
    def test_build_index_endpoint(self, mock_faiss, mock_verify, client, auth_headers):
        """Test index building endpoint."""
        mock_verify.return_value = "test-key"
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        with patch('tensorus.api.routers.vector.get_index_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_built_index = Mock()
            mock_built_index.get_stats.return_value = {
                "dimension": 384,
                "index_type": "flat",
                "total_vectors": 100
            }
            mock_manager.build_index.return_value = mock_built_index
            mock_get_manager.return_value = mock_manager
            
            response = client.post(
                "/vector/index/build",
                json={
                    "dataset_name": "test_embeddings",
                    "index_type": "flat",
                    "metric": "cosine"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "stats" in data

    @patch('tensorus.api.security.verify_api_key')
    def test_index_search_endpoint(self, mock_verify, client, auth_headers):
        """Test index search endpoint."""
        mock_verify.return_value = "test-key"
        
        with patch('tensorus.api.routers.vector.get_index_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.search_index.return_value = [
                {
                    "vector_id": "vec1",
                    "similarity": 0.95,
                    "metadata": {"source_text": "Hello world"}
                }
            ]
            mock_get_manager.return_value = mock_manager
            
            response = client.post(
                "/vector/index/search",
                json={
                    "index_name": "test_index",
                    "query": "Hello",
                    "k": 5
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["results"]) == 1

    @patch('tensorus.api.security.verify_api_key')
    def test_list_indexes_endpoint(self, mock_verify, client, auth_headers):
        """Test list indexes endpoint."""
        mock_verify.return_value = "test-key"
        
        with patch('tensorus.api.routers.vector.get_index_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.list_indexes.return_value = [
                {
                    "name": "test_index",
                    "dimension": 384,
                    "index_type": "flat",
                    "total_vectors": 100
                }
            ]
            mock_get_manager.return_value = mock_manager
            
            response = client.get("/vector/index/list", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["indexes"]) == 1

    @patch('tensorus.api.security.verify_api_key')
    def test_delete_index_endpoint(self, mock_verify, client, auth_headers):
        """Test delete index endpoint."""
        mock_verify.return_value = "test-key"
        
        with patch('tensorus.api.routers.vector.get_index_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.delete_index.return_value = True
            mock_get_manager.return_value = mock_manager
            
            response = client.delete("/vector/index/test_index", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @patch('tensorus.api.security.verify_api_key')
    def test_list_models_endpoint(self, mock_verify, client, auth_headers):
        """Test list embedding models endpoint."""
        mock_verify.return_value = "test-key"
        
        response = client.get("/vector/models", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(model["name"] == "all-MiniLM-L6-v2" for model in data)

    def test_unauthorized_access(self, client):
        """Test that endpoints require authentication."""
        response = client.post("/vector/embed", json={"texts": ["test"]})
        assert response.status_code == 401

    @patch('tensorus.api.security.verify_api_key')
    def test_invalid_request_data(self, mock_verify, client, auth_headers):
        """Test handling of invalid request data."""
        mock_verify.return_value = "test-key"
        
        # Missing required fields
        response = client.post("/vector/embed", json={}, headers=auth_headers)
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])
