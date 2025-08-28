"""
Test suite for Tensorus Vector Database API endpoints.

Tests the comprehensive REST API for vector database capabilities including:
- Embedding generation and storage
- Similarity search with multi-tenancy
- Hybrid search combining semantic and computational relevance
- Tensor workflow execution
- Performance monitoring and metrics
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from tensorus.api import app
from tensorus.tensor_storage import TensorStorage
from tensorus.embedding_agent import EmbeddingAgent
from tensorus.hybrid_search import HybridSearchEngine


# Use client fixture from conftest.py which has proper auth setup


class TestEmbeddingEndpoints:
    """Test embedding generation and storage endpoints."""
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_embed_text_single(self, mock_get_agent, client, auth_headers):
        """Test embedding single text."""
        
        # Mock embedding agent
        mock_agent = Mock()
        mock_agent.store_embeddings = AsyncMock(return_value=["record_1"])
        mock_agent.default_model = "all-MiniLM-L6-v2"
        mock_agent.get_model_info.return_value = Mock(
            name="all-MiniLM-L6-v2",
            provider="sentence-transformers",
            dimension=384,
            description="Test model"
        )
        mock_get_agent.return_value = mock_agent
        
        response = client.post(
            "/api/v1/vector/embed",
            json={
                "texts": "Hello world",
                "dataset_name": "test_dataset",
                "namespace": "test",
                "tenant_id": "tenant_1"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["embeddings_count"] == 1
        assert len(data["record_ids"]) == 1
        assert data["namespace"] == "test"
        assert data["tenant_id"] == "tenant_1"
        
        # Check the response data structure (endpoint working correctly)
        assert "record_ids" in data
        assert isinstance(data["record_ids"], list)
        assert "model_info" in data
        assert data["model_info"]["name"] == "all-MiniLM-L6-v2"
        
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_embed_text_multiple(self, mock_get_agent, client, auth_headers):
        """Test embedding multiple texts."""
        
        mock_agent = Mock()
        mock_agent.store_embeddings = AsyncMock(return_value=["record_1", "record_2", "record_3"])
        mock_agent.default_model = "all-MiniLM-L6-v2"
        mock_agent.get_model_info.return_value = Mock(
            name="all-MiniLM-L6-v2",
            provider="sentence-transformers",
            dimension=384,
            description="Test model"
        )
        mock_get_agent.return_value = mock_agent
        
        texts = ["Text one", "Text two", "Text three"]
        response = client.post(
            "/api/v1/vector/embed",
            json={
                "texts": texts,
                "dataset_name": "test_dataset",
                "model_name": "all-MiniLM-L6-v2",
                "provider": "sentence-transformers"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["embeddings_count"] == 3
        assert len(data["record_ids"]) == 3
        
    def test_embed_text_empty(self, client, auth_headers):
        """Test embedding empty text returns error."""
        
        response = client.post(
            "/api/v1/vector/embed",
            json={
                "texts": "",
                "dataset_name": "test_dataset"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
        
    def test_embed_text_no_auth(self, client):
        """Test embedding without authentication fails."""
        response = client.post(
            "/api/v1/vector/embed",
            json={
                "texts": "Hello world",
                "dataset_name": "test_dataset"
            })
        
        assert response.status_code == 401


class TestSimilaritySearchEndpoints:
    """Test similarity search endpoints."""
    
    def test_similarity_search(self, client, auth_headers):
        """Test basic similarity search."""
        
        # Mock search results
        mock_results = [
            {
                "record_id": "record_1",
                "similarity_score": 0.95,
                "rank": 1,
                "source_text": "Similar text",
                "metadata": {"category": "test"},
                "namespace": "default",
                "tenant_id": "default"
            },
            {
                "record_id": "record_2", 
                "similarity_score": 0.87,
                "rank": 2,
                "source_text": "Another similar text",
                "metadata": {"category": "test"},
                "namespace": "default",
                "tenant_id": "default"
            }
        ]
        
        # Create a mock embedding agent
        mock_agent = Mock()
        mock_agent.similarity_search = AsyncMock(return_value=mock_results)
        mock_agent.vector_indexes = {"test_dataset": Mock()}
        
        # Override the dependency in the FastAPI app
        from tensorus.api.routers.vector import get_embedding_agent
        from tensorus.api import app
        
        def mock_get_embedding_agent():
            return mock_agent
            
        app.dependency_overrides[get_embedding_agent] = mock_get_embedding_agent
        
        try:
            response = client.post(
                "/api/v1/vector/search",
                json={
                    "query": "test query",
                    "dataset_name": "test_dataset",
                    "k": 5,
                    "namespace": "test_namespace"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["query"] == "test query"
            assert len(data["results"]) == 2
            assert data["total_results"] == 2
            assert "search_time_ms" in data
        finally:
            # Clean up dependency override
            app.dependency_overrides.pop(get_embedding_agent, None)
        
        # Check result structure
        result = data["results"][0]
        assert "record_id" in result
        assert "similarity_score" in result
        assert "rank" in result
        assert "source_text" in result
        
        # Verify agent was called correctly
        mock_agent.similarity_search.assert_called_once()
        call_args = mock_agent.similarity_search.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["dataset_name"] == "test_dataset"
        assert call_args[1]["k"] == 5
        assert call_args[1]["namespace"] == "test_namespace"
        
    
    def test_similarity_search_with_filters(self, client, auth_headers):
        """Test similarity search with filters."""
        
        mock_agent = Mock()
        mock_agent.similarity_search = AsyncMock(return_value=[])
        mock_agent.vector_indexes = {"test_dataset": Mock()}
        
        # Override the dependency
        from tensorus.api.routers.vector import get_embedding_agent
        from tensorus.api import app
        
        def mock_get_embedding_agent():
            return mock_agent
            
        app.dependency_overrides[get_embedding_agent] = mock_get_embedding_agent
        
        try:
            response = client.post(
                "/api/v1/vector/search",
                json={
                    "query": "test query",
                    "dataset_name": "test_dataset",
                    "k": 3,
                    "namespace": "prod",
                    "tenant_id": "tenant_123",
                    "similarity_threshold": 0.8,
                    "include_vectors": True
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            
            # Verify filters were passed
            call_args = mock_agent.similarity_search.call_args
            assert call_args[1]["namespace"] == "prod"
            assert call_args[1]["tenant_id"] == "tenant_123"
            assert call_args[1]["similarity_threshold"] == 0.8
        finally:
            # Clean up dependency override
            app.dependency_overrides.pop(get_embedding_agent, None)


class TestHybridSearchEndpoints:
    """Test hybrid search endpoints combining semantic and computational relevance."""
    
    
    def test_hybrid_search(self, client, auth_headers):
        """Test hybrid search functionality."""
        
        # Mock hybrid search results
        mock_results = [
            Mock(
                record_id="record_1",
                semantic_score=0.9,
                computational_score=0.8,
                hybrid_score=0.86,
                rank=1,
                source_text="Test text",
                tensor_shape=(4, 4),
                computational_lineage=["parent_1"],
                metadata={"type": "matrix"}
            )
        ]
        
        mock_engine = Mock()
        mock_engine.hybrid_search = AsyncMock(return_value=mock_results)
        
        # Mock the embedding agent to avoid vector index issues
        mock_agent = Mock()
        mock_agent.vector_indexes = {"test_dataset": Mock()}
        
        # Override the dependencies
        from tensorus.api.routers.vector import get_hybrid_search, get_embedding_agent
        from tensorus.api import app
        
        def mock_get_hybrid_search():
            return mock_engine
            
        def mock_get_embedding_agent():
            return mock_agent
            
        app.dependency_overrides[get_hybrid_search] = mock_get_hybrid_search
        app.dependency_overrides[get_embedding_agent] = mock_get_embedding_agent
        
        try:
            response = client.post(
                "/api/v1/vector/hybrid-search",
                json={
                    "text_query": "machine learning algorithms",
                    "dataset_name": "test_dataset",
                    "tensor_operations": [
                        {
                            "operation_name": "svd",
                            "parameters": {},
                            "description": "SVD decomposition"
                        }
                    ],
                    "similarity_weight": 0.7,
                    "computation_weight": 0.3,
                    "k": 5
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["results"]) == 1
            assert data["semantic_weight"] == 0.7
            assert data["computation_weight"] == 0.3
            assert "search_time_ms" in data
            
            # Check result structure
            result = data["results"][0]
            assert "record_id" in result
            assert "semantic_score" in result
            assert "computational_score" in result
            assert "hybrid_score" in result
            assert "tensor_shape" in result
            assert "computational_lineage" in result
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.pop(get_hybrid_search, None)
            app.dependency_overrides.pop(get_embedding_agent, None)
        
    
    def test_hybrid_search_invalid_weights(self, client, auth_headers):
        """Test hybrid search with invalid weight sum."""
        
        response = client.post(
            "/api/v1/vector/hybrid-search",
            json={
                "text_query": "test query",
                "dataset_name": "test_dataset",
                "similarity_weight": 0.8,
                "computation_weight": 0.3  # Sum = 1.1, should fail
            },
            headers=auth_headers
        )
        
        assert response.status_code == 400
        assert "sum to 1.0" in response.json()["detail"]


class TestTensorWorkflowEndpoints:
    """Test tensor workflow execution endpoints."""
    
    
    @patch('tensorus.api.routers.vector.get_hybrid_search')
    def test_tensor_workflow(self, mock_get_hybrid, client, auth_headers):
        """Test tensor workflow execution."""
        
        # Mock workflow results
        mock_workflow_result = {
            "workflow_id": "workflow_123",
            "operations_executed": [
                {
                    "operation_name": "svd",
                    "parameters": {},
                    "input_shape": (4, 4),
                    "output_shape": (4, 4),
                    "operation_id": "op_1"
                }
            ],
            "final_result": {
                "tensor_id": "final_tensor_123",
                "tensor_shape": (4, 4),
                "dataset": "test_dataset_workflow_results"
            },
            "intermediate_results": [
                {
                    "step": 1,
                    "tensor_id": "intermediate_1",
                    "tensor_shape": (4, 4),
                    "operation_name": "svd"
                }
            ],
            "computational_lineage": ["input_tensor", "intermediate_1", "final_tensor_123"]
        }
        
        mock_engine = Mock()
        mock_engine.execute_tensor_workflow = AsyncMock(return_value=mock_workflow_result)
        mock_get_hybrid.return_value = mock_engine
        
        response = client.post(
            "/api/v1/vector/tensor-workflow",
            json={
                "workflow_query": "matrix decomposition workflow",
                "dataset_name": "test_dataset",
                "operations": [
                    {
                        "operation_name": "svd",
                        "parameters": {},
                        "description": "Singular value decomposition"
                    }
                ],
                "save_intermediates": True
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["workflow_id"] == "workflow_123"
        assert len(data["operations_executed"]) == 1
        assert "final_result" in data
        assert "intermediate_results" in data
        assert "computational_lineage" in data
        
        # Verify workflow engine was called correctly
        mock_engine.execute_tensor_workflow.assert_called_once()
        call_args = mock_engine.execute_tensor_workflow.call_args
        assert call_args[1]["workflow_query"] == "matrix decomposition workflow"
        assert call_args[1]["dataset_name"] == "test_dataset"
        assert call_args[1]["save_intermediates"] is True


class TestVectorIndexEndpoints:
    """Test vector index management endpoints."""
    
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_build_vector_index(self, mock_get_agent, client, auth_headers):
        """Test building vector index."""
        
        # Mock index build results
        mock_index_stats = {
            "index_type": "partitioned",
            "metric": "cosine", 
            "build_time_seconds": 1.23,
            "total_vectors": 1000,
            "partitions": 8,
            "index_size_mb": 15.6,
            "created_at": "2023-01-01T12:00:00"
        }
        
        mock_agent = Mock()
        mock_agent.build_vector_index = AsyncMock(return_value=mock_index_stats)
        mock_get_agent.return_value = mock_agent
        
        response = client.post(
            "/api/v1/vector/index/build",
            json={
                "dataset_name": "test_dataset",
                "index_type": "partitioned",
                "metric": "cosine",
                "num_partitions": 8
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "index_stats" in data
        assert data["index_stats"]["total_vectors"] == 1000
        assert data["index_stats"]["partitions"] == 8
        
        # Verify agent was called correctly
        mock_agent.build_vector_index.assert_called_once()
        call_args = mock_agent.build_vector_index.call_args
        assert call_args[1]["dataset_name"] == "test_dataset"
        assert call_args[1]["index_type"] == "partitioned"
        assert call_args[1]["metric"] == "cosine"
        assert call_args[1]["num_partitions"] == 8


class TestModelAndStatsEndpoints:
    """Test model listing and statistics endpoints."""
    
    
    def test_list_models(self, client, auth_headers):
        """Test listing available models."""
        
        # Mock model list - using dicts instead of Mock objects for serialization
        from tensorus.embedding_agent import EmbeddingModelInfo
        mock_models = {
            "sentence-transformers": [
                EmbeddingModelInfo(
                    name="all-MiniLM-L6-v2",
                    provider="sentence-transformers",
                    dimension=384,
                    max_tokens=256,
                    description="Fast model",
                    supports_batch=True,
                    cost_per_1k_tokens=0.0
                )
            ],
            "openai": [
                EmbeddingModelInfo(
                    name="text-embedding-3-small",
                    provider="openai",
                    dimension=1536,
                    max_tokens=8192,
                    description="OpenAI model",
                    supports_batch=True,
                    cost_per_1k_tokens=0.00002
                )
            ]
        }
        
        mock_agent = Mock()
        mock_agent.list_available_models.return_value = mock_models
        
        # Override the dependency
        from tensorus.api.routers.vector import get_embedding_agent
        from tensorus.api import app
        
        def mock_get_embedding_agent():
            return mock_agent
            
        app.dependency_overrides[get_embedding_agent] = mock_get_embedding_agent
        
        try:
            response = client.get(
                "/api/v1/vector/models",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["total_models"] == 2
            assert "sentence-transformers" in data["providers"]
            assert "openai" in data["providers"]
            
            # Check model structure
            st_model = data["providers"]["sentence-transformers"][0]
            assert st_model["name"] == "all-MiniLM-L6-v2"
            assert st_model["dimension"] == 384
        finally:
            # Clean up dependency override
            app.dependency_overrides.pop(get_embedding_agent, None)
        
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_get_embedding_stats(self, mock_get_agent, client, auth_headers):
        """Test getting embedding statistics."""
        
        # Mock stats
        mock_stats = {
            "dataset_name": "test_dataset",
            "total_embeddings": 5000,
            "created_at": "2023-01-01T10:00:00",
            "last_updated": "2023-01-01T12:00:00",
            "vector_index_size_mb": 20.5,
            "partitions": 8
        }
        
        mock_agent = Mock()
        mock_agent.get_embedding_stats.return_value = mock_stats
        mock_get_agent.return_value = mock_agent
        
        response = client.get(
            "/api/v1/vector/stats/test_dataset",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["dataset_name"] == "test_dataset"
        assert data["stats"]["total_embeddings"] == 5000
        assert data["stats"]["vector_index_size_mb"] == 20.5
        
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    @patch('tensorus.api.routers.vector.get_hybrid_search')
    def test_get_performance_metrics(self, mock_get_hybrid, mock_get_agent, client, auth_headers):
        """Test getting performance metrics."""
        
        # Mock metrics
        mock_embedding_metrics = {
            "total_embeddings_generated": 10000,
            "average_embedding_time": 0.05,
            "cache_hit_rate": 0.75
        }
        
        mock_search_metrics = {
            "total_searches": 500,
            "average_search_time": 0.02,
            "semantic_search_ratio": 0.8
        }
        
        mock_lineage_metrics = {
            "total_tensors_tracked": 1000,
            "total_operations_recorded": 2500,
            "average_lineage_depth": 3.2
        }
        
        mock_agent = Mock()
        mock_agent.get_metrics.return_value = mock_embedding_metrics
        mock_get_agent.return_value = mock_agent
        
        mock_engine = Mock()
        mock_engine.get_metrics.return_value = mock_search_metrics
        mock_engine.get_lineage_stats.return_value = mock_lineage_metrics
        mock_get_hybrid.return_value = mock_engine
        
        response = client.get(
            "/api/v1/vector/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "embedding_metrics" in data
        assert "search_metrics" in data
        assert "lineage_metrics" in data
        
        assert data["embedding_metrics"]["total_embeddings_generated"] == 10000
        assert data["search_metrics"]["total_searches"] == 500
        assert data["lineage_metrics"]["total_tensors_tracked"] == 1000


class TestVectorDeletion:
    """Test vector deletion endpoints."""
    
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_delete_vectors(self, mock_get_agent, client, auth_headers):
        """Test deleting vectors from dataset."""
        
        # Mock embedding agent with vector index and tensor storage
        mock_vector_index = Mock()
        mock_vector_index.delete_vectors = AsyncMock()
        
        mock_tensor_storage = Mock()
        mock_tensor_storage.delete_tensor = Mock()
        
        mock_agent = Mock()
        mock_agent.vector_indexes = {"test_dataset": mock_vector_index}
        mock_agent.tensor_storage = mock_tensor_storage
        mock_get_agent.return_value = mock_agent
        
        vector_ids = ["vec_1", "vec_2", "vec_3"]
        response = client.delete(
            f"/api/v1/vector/vectors/test_dataset",
            params={"vector_ids": vector_ids},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["deleted_count"] == 3
        assert data["requested_count"] == 3
        
        # Verify deletions were called
        mock_vector_index.delete_vectors.assert_called_once_with(set(vector_ids))
        assert mock_tensor_storage.delete_tensor.call_count == 3
        
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_delete_vectors_no_index(self, mock_get_agent, client, auth_headers):
        """Test deleting vectors when no index exists."""
        
        mock_agent = Mock()
        mock_agent.vector_indexes = {}  # No index for dataset
        mock_get_agent.return_value = mock_agent
        
        response = client.delete(
            "/api/v1/vector/vectors/nonexistent_dataset",
            params={"vector_ids": ["vec_1"]},
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestErrorHandling:
    """Test error handling across vector API endpoints."""
    
    def test_unauthenticated_requests(self, client):
        """Test that unauthenticated requests are rejected."""
        endpoints = [
            ("/api/v1/vector/embed", "POST"),
            ("/api/v1/vector/search", "POST"), 
            ("/api/v1/vector/hybrid-search", "POST"),
            ("/api/v1/vector/models", "GET"),
            ("/api/v1/vector/stats/test", "GET"),
            ("/api/v1/vector/metrics", "GET")
        ]
        
        for endpoint, method in endpoints:
            if method == "POST":
                response = client.post(endpoint, json={})
            else:
                response = client.get(endpoint)
                
            assert response.status_code == 401
            
    
    @patch('tensorus.api.routers.vector.get_embedding_agent')
    def test_internal_server_errors(self, mock_get_agent, client, auth_headers):
        """Test handling of internal server errors."""
        
        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.store_embeddings = AsyncMock(side_effect=Exception("Internal error"))
        mock_get_agent.return_value = mock_agent
        
        response = client.post(
            "/api/v1/vector/embed",
            json={
                "texts": "test text",
                "dataset_name": "test_dataset"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])