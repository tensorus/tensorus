"""
Test suite for Tensorus Vector Database capabilities.

This module tests the comprehensive vector database implementation including:
- Vector indexing with geometric partitioning
- Multi-tenant namespace isolation
- Freshness layers for real-time updates
- Embedding generation and storage
- Hybrid search combining semantic and computational relevance
- Tensor workflow execution with lineage tracking
"""

import asyncio
import numpy as np
import pytest
import torch
from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4

from tensorus.tensor_storage import TensorStorage
from tensorus.vector_database import (
    VectorMetadata, GeometricPartitioner, FreshnessLayer,
    FAISSVectorIndex, PartitionedVectorIndex, SearchResult
)
from tensorus.embedding_agent import EmbeddingAgent, SentenceTransformersProvider
from tensorus.tensor_ops import TensorOps
from tensorus.hybrid_search import (
    HybridSearchEngine, HybridQuery, TensorOperation,
    ComputationalScorer, LineageTracker
)


@pytest.fixture
def tensor_storage():
    """Create tensor storage instance for testing."""
    return TensorStorage(storage_path=None)  # In-memory storage


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return {
        f"vec_{i}": (
            np.random.randn(384).astype(np.float32),
            VectorMetadata(
                vector_id=f"vec_{i}",
                namespace="test",
                tenant_id="tenant_1",
                content=f"Sample text {i}",
                metadata={"category": "test", "index": i}
            )
        )
        for i in range(100)
    }


class TestGeometricPartitioner:
    """Test geometric partitioning for vector indexes."""
    
    def test_initialization(self):
        """Test partitioner initialization."""
        partitioner = GeometricPartitioner(num_partitions=8, dimension=384)
        assert partitioner.num_partitions == 8
        assert partitioner.dimension == 384
        assert partitioner.centroids is None
        
    def test_fitting(self, sample_vectors):
        """Test fitting partitioner to vector distribution."""
        partitioner = GeometricPartitioner(num_partitions=4, dimension=384)
        
        vectors = np.array([vec for vec, _ in sample_vectors.values()])
        partitioner.fit(vectors)
        
        assert partitioner.centroids is not None
        assert partitioner.centroids.shape == (4, 384)
        
    def test_partition_assignment(self, sample_vectors):
        """Test vector assignment to partitions."""
        partitioner = GeometricPartitioner(num_partitions=4, dimension=384)
        
        vectors = np.array([vec for vec, _ in sample_vectors.values()])
        partitioner.fit(vectors)
        
        # Test assignment
        test_vector = vectors[0]
        test_id = "test_vec"
        
        partition = partitioner.assign_partition(test_vector, test_id)
        assert 0 <= partition < 4
        assert partitioner.get_partition(test_id) == partition
        
    def test_partition_balancing(self, sample_vectors):
        """Test that partitions are reasonably balanced."""
        partitioner = GeometricPartitioner(num_partitions=4, dimension=384)
        
        vectors = np.array([vec for vec, _ in sample_vectors.values()])
        partitioner.fit(vectors)
        
        # Assign all vectors
        partition_counts = {i: 0 for i in range(4)}
        
        for i, (vector, _) in enumerate(sample_vectors.values()):
            partition = partitioner.assign_partition(vector, f"vec_{i}")
            partition_counts[partition] += 1
            
        # Check that no partition is completely empty and none has too many
        for count in partition_counts.values():
            assert count > 0  # No empty partitions
            assert count < len(sample_vectors) * 0.8  # No partition with >80% of vectors


class TestFreshnessLayer:
    """Test freshness layer for real-time updates."""
    
    def test_initialization(self):
        """Test freshness layer initialization."""
        freshness = FreshnessLayer(max_size=1000, compaction_threshold=0.8)
        assert freshness.max_size == 1000
        assert freshness.compaction_threshold == 0.8
        assert len(freshness.fresh_vectors) == 0
        assert len(freshness.deleted_ids) == 0
        
    def test_add_vector(self):
        """Test adding vectors to freshness layer."""
        freshness = FreshnessLayer()
        
        vector = np.random.randn(384).astype(np.float32)
        metadata = VectorMetadata(vector_id="test_1", namespace="default")
        
        freshness.add_vector("test_1", vector, metadata)
        
        assert "test_1" in freshness.fresh_vectors
        assert freshness.fresh_vectors["test_1"][1].vector_id == "test_1"
        
    def test_delete_vector(self):
        """Test deleting vectors from freshness layer."""
        freshness = FreshnessLayer()
        
        # Add then delete
        vector = np.random.randn(384).astype(np.float32)
        metadata = VectorMetadata(vector_id="test_1", namespace="default")
        
        freshness.add_vector("test_1", vector, metadata)
        freshness.delete_vector("test_1")
        
        assert "test_1" not in freshness.fresh_vectors
        assert "test_1" in freshness.deleted_ids
        
    def test_compaction_threshold(self):
        """Test compaction threshold detection."""
        freshness = FreshnessLayer(max_size=10, compaction_threshold=0.8)
        
        # Add vectors below threshold
        for i in range(7):
            vector = np.random.randn(384).astype(np.float32)
            metadata = VectorMetadata(vector_id=f"test_{i}", namespace="default")
            freshness.add_vector(f"test_{i}", vector, metadata)
            
        assert not freshness.needs_compaction()
        
        # Add one more to trigger threshold
        vector = np.random.randn(384).astype(np.float32)
        metadata = VectorMetadata(vector_id="test_8", namespace="default")
        freshness.add_vector("test_8", vector, metadata)
        
        assert freshness.needs_compaction()
        
    def test_clear(self):
        """Test clearing freshness layer."""
        freshness = FreshnessLayer()
        
        # Add some data
        vector = np.random.randn(384).astype(np.float32)
        metadata = VectorMetadata(vector_id="test_1", namespace="default")
        freshness.add_vector("test_1", vector, metadata)
        freshness.delete_vector("test_2")
        
        assert len(freshness.fresh_vectors) > 0
        assert len(freshness.deleted_ids) > 0
        
        freshness.clear()
        
        assert len(freshness.fresh_vectors) == 0
        assert len(freshness.deleted_ids) == 0


@pytest.mark.skipif(not pytest.importorskip("faiss", reason="FAISS not available"), 
                   reason="FAISS not available")
class TestFAISSVectorIndex:
    """Test FAISS-based vector index."""
    
    def test_initialization(self):
        """Test FAISS index initialization."""
        index = FAISSVectorIndex(dimension=384, metric="cosine")
        assert index.dimension == 384
        assert index.metric == "cosine"
        assert index.index.ntotal == 0
        
    @pytest.mark.asyncio
    async def test_add_vectors(self, sample_vectors):
        """Test adding vectors to FAISS index."""
        index = FAISSVectorIndex(dimension=384, metric="cosine")
        
        # Add subset of vectors
        subset = dict(list(sample_vectors.items())[:10])
        await index.add_vectors(subset)
        
        assert index.index.ntotal == 10
        assert len(index.vector_metadata) == 10
        assert len(index.id_to_index) == 10
        
    @pytest.mark.asyncio
    async def test_search(self, sample_vectors):
        """Test vector similarity search."""
        index = FAISSVectorIndex(dimension=384, metric="cosine")
        
        # Add vectors
        await index.add_vectors(sample_vectors)
        
        # Search with first vector
        query_vector = list(sample_vectors.values())[0][0]
        results = await index.search(query_vector, k=5)
        
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].rank == 1
        
        # Scores should be in descending order for cosine similarity
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        
    @pytest.mark.asyncio
    async def test_search_with_filters(self, sample_vectors):
        """Test filtered vector search."""
        index = FAISSVectorIndex(dimension=384, metric="cosine")
        
        # Modify some vectors to have different tenant_id
        modified_vectors = sample_vectors.copy()
        for i, (vec_id, (vector, metadata)) in enumerate(list(modified_vectors.items())[:5]):
            metadata.tenant_id = "tenant_2"
            
        await index.add_vectors(modified_vectors)
        
        # Search with tenant filter
        query_vector = list(modified_vectors.values())[0][0]
        results = await index.search(
            query_vector, 
            k=10, 
            filters={"tenant_id": "tenant_2"}
        )
        
        assert len(results) <= 5  # Only 5 vectors have tenant_2
        assert all(r.metadata.tenant_id == "tenant_2" for r in results)


@pytest.mark.asyncio
class TestPartitionedVectorIndex:
    """Test partitioned vector index with freshness layer."""
    
    async def test_initialization(self):
        """Test partitioned index initialization."""
        index = PartitionedVectorIndex(dimension=384, num_partitions=4, metric="cosine")
        
        assert index.dimension == 384
        assert len(index.partitions) == 4
        assert index.freshness_layer is not None
        
    async def test_add_and_search(self, sample_vectors):
        """Test adding vectors and searching across partitions."""
        index = PartitionedVectorIndex(dimension=384, num_partitions=4, metric="cosine")
        
        # Add vectors (should go to freshness layer first)
        await index.add_vectors(sample_vectors)
        
        # Search should find results from freshness layer
        query_vector = list(sample_vectors.values())[0][0]
        results = await index.search(query_vector, k=5)
        
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
        
    async def test_compaction(self, sample_vectors):
        """Test automatic compaction from freshness layer to partitions."""
        # Use small freshness layer for testing
        index = PartitionedVectorIndex(dimension=384, num_partitions=2, metric="cosine")
        index.freshness_layer.max_size = 10
        index.freshness_layer.compaction_threshold = 0.8
        
        # Add enough vectors to trigger compaction
        subset = dict(list(sample_vectors.items())[:15])
        await index.add_vectors(subset)
        
        # Should trigger compaction automatically
        # Check that some vectors moved to main partitions
        total_partition_vectors = sum(p.get_stats().total_vectors for p in index.partitions.values())
        assert total_partition_vectors > 0
        
    async def test_delete_vectors(self, sample_vectors):
        """Test vector deletion."""
        index = PartitionedVectorIndex(dimension=384, num_partitions=2, metric="cosine")
        
        # Add vectors
        await index.add_vectors(sample_vectors)
        
        # Delete some vectors
        delete_ids = set(list(sample_vectors.keys())[:5])
        await index.delete_vectors(delete_ids)
        
        # Check that vectors are marked as deleted
        deleted_ids = index.freshness_layer.get_deleted_ids()
        assert delete_ids.issubset(deleted_ids)


class TestEmbeddingAgent:
    """Test embedding agent functionality."""
    
    @pytest.fixture
    def embedding_agent(self, tensor_storage):
        """Create embedding agent for testing."""
        return EmbeddingAgent(
            tensor_storage=tensor_storage,
            default_provider="sentence-transformers",
            default_model="all-MiniLM-L6-v2"
        )
        
    @pytest.mark.skipif(not pytest.importorskip("sentence_transformers", reason="sentence-transformers not available"), 
                       reason="sentence-transformers not available")
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, embedding_agent):
        """Test embedding generation."""
        texts = ["Hello world", "Machine learning", "Vector database"]
        
        embeddings, model_info = await embedding_agent.generate_embeddings(texts)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
        assert model_info.name == "all-MiniLM-L6-v2"
        assert model_info.provider == "sentence-transformers"
        
    @pytest.mark.skipif(not pytest.importorskip("sentence_transformers", reason="sentence-transformers not available"), 
                       reason="sentence-transformers not available")
    @pytest.mark.asyncio
    async def test_store_embeddings(self, embedding_agent):
        """Test storing embeddings in tensor storage and vector index."""
        texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
        dataset_name = "test_embeddings"
        
        record_ids = await embedding_agent.store_embeddings(
            texts=texts,
            dataset_name=dataset_name,
            namespace="test",
            tenant_id="tenant_1"
        )
        
        assert len(record_ids) == 3
        assert dataset_name in embedding_agent.tensor_storage.datasets
        assert dataset_name in embedding_agent.vector_indexes
        
        # Check tensor storage
        for record_id in record_ids:
            tensor_record = embedding_agent.tensor_storage.get_tensor(dataset_name, record_id)
            assert tensor_record["tensor"].shape[0] == 384
            assert "source_text" in tensor_record["metadata"]
            
    @pytest.mark.skipif(not pytest.importorskip("sentence_transformers", reason="sentence-transformers not available"), 
                       reason="sentence-transformers not available")
    @pytest.mark.asyncio
    async def test_similarity_search(self, embedding_agent):
        """Test similarity search."""
        texts = ["Machine learning algorithms", "Deep neural networks", "Computer vision tasks"]
        dataset_name = "test_search"
        
        # Store embeddings
        await embedding_agent.store_embeddings(
            texts=texts,
            dataset_name=dataset_name
        )
        
        # Search
        results = await embedding_agent.similarity_search(
            query="artificial intelligence",
            dataset_name=dataset_name,
            k=2
        )
        
        assert len(results) <= 2
        assert all("similarity_score" in r for r in results)
        assert all("source_text" in r for r in results)
        assert all(r["similarity_score"] >= 0 for r in results)


class TestComputationalScorer:
    """Test computational scoring for hybrid search."""
    
    @pytest.fixture
    def tensor_ops(self):
        """Create TensorOps instance."""
        return TensorOps()
        
    @pytest.fixture
    def computational_scorer(self, tensor_ops):
        """Create computational scorer."""
        return ComputationalScorer(tensor_ops)
        
    def test_shape_scoring(self, computational_scorer):
        """Test tensor shape-based scoring."""
        tensor = torch.randn(3, 4)
        
        # Test exact shape match
        query = HybridQuery(filters={"preferred_shape": [3, 4]})
        score = computational_scorer.score_tensor_properties(tensor, query)
        assert score > 0.2  # Should get points for shape match
        
        # Test different shape
        query = HybridQuery(filters={"preferred_shape": [2, 2]})
        score = computational_scorer.score_tensor_properties(tensor, query)
        assert score >= 0  # Should still get some points
        
    def test_sparsity_scoring(self, computational_scorer):
        """Test sparsity-based scoring."""
        # Create sparse tensor
        sparse_tensor = torch.zeros(10, 10)
        sparse_tensor[0, 0] = 1.0
        sparse_tensor[5, 5] = 1.0
        
        query = HybridQuery(filters={"sparsity_preference": 0.8})  # 80% sparse
        score = computational_scorer.score_tensor_properties(sparse_tensor, query)
        assert score > 0
        
    def test_operation_compatibility(self, computational_scorer):
        """Test operation compatibility scoring."""
        matrix = torch.randn(5, 5)
        tensor_3d = torch.randn(3, 4, 5)
        
        # SVD operation should prefer matrices
        svd_op = TensorOperation(operation_name="svd", function=None)
        matrix_score = computational_scorer.score_operation_compatibility(matrix, [svd_op])
        tensor_score = computational_scorer.score_operation_compatibility(tensor_3d, [svd_op])
        
        assert matrix_score > tensor_score
        
        # Tucker decomposition should prefer 3D+ tensors
        tucker_op = TensorOperation(operation_name="tucker_decomposition", function=None)
        matrix_score = computational_scorer.score_operation_compatibility(matrix, [tucker_op])
        tensor_score = computational_scorer.score_operation_compatibility(tensor_3d, [tucker_op])
        
        assert tensor_score > matrix_score


class TestLineageTracker:
    """Test computational lineage tracking."""
    
    @pytest.fixture
    def lineage_tracker(self):
        """Create lineage tracker."""
        return LineageTracker()
        
    def test_record_operation(self, lineage_tracker):
        """Test recording tensor operations."""
        input_ids = ["input_1", "input_2"]
        output_id = "output_1"
        operation = TensorOperation(operation_name="add", function=None, parameters={"alpha": 1.0})
        
        op_id = lineage_tracker.record_operation(input_ids, output_id, operation)
        
        assert op_id is not None
        assert output_id in lineage_tracker.lineage_graph
        assert lineage_tracker.lineage_graph[output_id] == input_ids
        assert output_id in lineage_tracker.operation_history
        
    def test_get_lineage(self, lineage_tracker):
        """Test getting computational lineage."""
        # Create a chain of operations: A -> B -> C -> D
        operations = [
            (["A"], "B", TensorOperation("op1", None)),
            (["B"], "C", TensorOperation("op2", None)),
            (["C"], "D", TensorOperation("op3", None))
        ]
        
        for inputs, output, op in operations:
            lineage_tracker.record_operation(inputs, output, op)
            
        # Get lineage for D
        lineage = lineage_tracker.get_lineage("D", depth=5)
        expected_lineage = ["C", "B", "A"]
        
        assert set(lineage) == set(expected_lineage)
        
    def test_operation_history(self, lineage_tracker):
        """Test getting operation history."""
        input_ids = ["input_1"]
        output_id = "output_1"
        operation = TensorOperation(
            operation_name="multiply",
            function=None,
            parameters={"factor": 2.0},
            description="Multiply by factor"
        )
        
        lineage_tracker.record_operation(input_ids, output_id, operation)
        
        history = lineage_tracker.get_operation_history(output_id)
        
        assert len(history) == 1
        assert history[0]["operation_name"] == "multiply"
        assert history[0]["parameters"] == {"factor": 2.0}
        assert history[0]["input_tensor_ids"] == input_ids


@pytest.mark.skipif(not all(pytest.importorskip(pkg, reason=f"{pkg} not available") 
                           for pkg in ["sentence_transformers", "faiss"]), 
                   reason="Required packages not available")
class TestHybridSearchEngine:
    """Test hybrid search engine combining semantic and computational search."""
    
    @pytest.fixture
    def tensor_ops(self):
        """Create TensorOps instance."""
        return TensorOps()
        
    @pytest.fixture
    def hybrid_search_engine(self, tensor_storage, tensor_ops):
        """Create hybrid search engine."""
        embedding_agent = EmbeddingAgent(tensor_storage)
        return HybridSearchEngine(tensor_storage, embedding_agent, tensor_ops)
        
    @pytest.mark.asyncio
    async def test_hybrid_search(self, hybrid_search_engine):
        """Test hybrid search combining semantic and computational relevance."""
        dataset_name = "test_hybrid"
        
        # Store some tensors with different properties
        texts = [
            "Linear algebra operations on matrices",
            "Deep learning with neural networks", 
            "Computer vision image processing"
        ]
        
        # Store embeddings first
        await hybrid_search_engine.embedding_agent.store_embeddings(
            texts=texts,
            dataset_name=dataset_name
        )
        
        # Create hybrid query
        query = HybridQuery(
            text_query="machine learning algorithms",
            tensor_operations=[
                TensorOperation(operation_name="svd", function=None, description="SVD decomposition")
            ],
            similarity_weight=0.7,
            computation_weight=0.3,
            k=3
        )
        
        # Execute hybrid search
        results = await hybrid_search_engine.hybrid_search(query, dataset_name)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'hybrid_score') for r in results)
        assert all(hasattr(r, 'semantic_score') for r in results)
        assert all(hasattr(r, 'computational_score') for r in results)
        
        # Results should be sorted by hybrid score
        scores = [r.hybrid_score for r in results]
        assert scores == sorted(scores, reverse=True)
        
    @pytest.mark.asyncio
    async def test_tensor_workflow(self, hybrid_search_engine):
        """Test tensor workflow execution with lineage tracking."""
        dataset_name = "test_workflow"
        
        # Store a matrix for testing
        matrix = torch.randn(4, 4)
        hybrid_search_engine.tensor_storage.add_tensor(
            dataset_name=dataset_name,
            record_id="matrix_1",
            tensor=matrix,
            metadata={
                "source_text": "Random matrix for testing",
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Create embedding for semantic search
        await hybrid_search_engine.embedding_agent.store_embeddings(
            texts=["Random matrix for testing"],
            dataset_name=dataset_name
        )
        
        # Define workflow operations
        operations = [
            TensorOperation(
                operation_name="transpose",
                function=None,
                description="Transpose the matrix"
            )
        ]
        
        # Execute workflow
        result = await hybrid_search_engine.execute_tensor_workflow(
            workflow_query="matrix operations",
            dataset_name=dataset_name,
            operations=operations,
            save_intermediates=True
        )
        
        assert "workflow_id" in result
        assert "operations_executed" in result
        assert "final_result" in result
        assert "computational_lineage" in result
        
        assert len(result["operations_executed"]) == 1
        assert result["operations_executed"][0]["operation_name"] == "transpose"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])