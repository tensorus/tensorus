"""
Tests for Tensor Embeddings module.

Tests the core Tensor Embeddings functionality which provides
tensor-native queries based on mathematical properties.
"""

import pytest
import torch
import numpy as np
from uuid import UUID

from tensorus.tensor_embeddings import (
    TensorEmbeddingIndex,
    TensorDescriptor,
    TensorSimilarityResult,
    SimilarityMetric,
    TensorProperty,
    get_global_index
)


class TestTensorDescriptor:
    """Tests for TensorDescriptor creation and properties."""
    
    def test_basic_descriptor_creation(self):
        """Test creating a descriptor from a simple tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        descriptor = TensorDescriptor.from_tensor(tensor, name="test_matrix")
        
        assert descriptor.shape == (2, 2)
        assert descriptor.name == "test_matrix"
        assert descriptor.frobenius_norm > 0
        assert descriptor.is_square
    
    def test_symmetric_matrix_detection(self):
        """Test detection of symmetric matrices."""
        # Create a symmetric matrix
        symmetric = torch.tensor([[1.0, 2.0, 3.0],
                                   [2.0, 4.0, 5.0],
                                   [3.0, 5.0, 6.0]])
        descriptor = TensorDescriptor.from_tensor(symmetric)
        
        assert descriptor.is_symmetric
        assert descriptor.is_square
    
    def test_positive_definite_detection(self):
        """Test detection of positive definite matrices."""
        # Create a positive definite matrix (identity)
        pd_matrix = torch.eye(3)
        descriptor = TensorDescriptor.from_tensor(pd_matrix)
        
        assert descriptor.is_positive_definite
        assert descriptor.is_symmetric
        assert descriptor.is_square
    
    def test_diagonal_matrix_detection(self):
        """Test detection of diagonal matrices."""
        diagonal = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        descriptor = TensorDescriptor.from_tensor(diagonal)
        
        assert descriptor.is_diagonal
        assert descriptor.is_symmetric
        assert descriptor.is_square
    
    def test_sparse_tensor_detection(self):
        """Test detection of sparse tensors."""
        # Create a sparse tensor (>50% zeros)
        sparse = torch.zeros(10, 10)
        sparse[0, 0] = 1.0
        sparse[5, 5] = 2.0
        descriptor = TensorDescriptor.from_tensor(sparse)
        
        assert descriptor.is_sparse
        assert descriptor.sparsity > 0.5
    
    def test_norm_computation(self):
        """Test norm computation."""
        tensor = torch.ones(3, 3)
        descriptor = TensorDescriptor.from_tensor(tensor)
        
        # Frobenius norm of 3x3 ones matrix = sqrt(9) = 3
        assert abs(descriptor.frobenius_norm - 3.0) < 0.01
        # L1 norm = 9
        assert abs(descriptor.l1_norm - 9.0) < 0.01
    
    def test_eigenvalue_computation(self):
        """Test eigenvalue computation for square matrices."""
        # Identity matrix has all eigenvalues = 1
        identity = torch.eye(3)
        descriptor = TensorDescriptor.from_tensor(identity)
        
        assert descriptor.max_eigenvalue is not None
        assert abs(descriptor.max_eigenvalue - 1.0) < 0.01
        assert abs(descriptor.min_eigenvalue - 1.0) < 0.01
    
    def test_rank_computation(self):
        """Test rank computation."""
        # Full rank matrix
        full_rank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        descriptor = TensorDescriptor.from_tensor(full_rank)
        assert descriptor.rank == 2
        
        # Rank 1 matrix - use a clearer example
        # Rows are scalar multiples of each other
        rank_one = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        descriptor = TensorDescriptor.from_tensor(rank_one)
        assert descriptor.rank == 1


class TestTensorEmbeddingIndex:
    """Tests for TensorEmbeddingIndex operations."""
    
    @pytest.fixture
    def index(self):
        """Create a fresh index for each test."""
        return TensorEmbeddingIndex()
    
    def test_add_and_get_tensor(self, index):
        """Test adding and retrieving a tensor."""
        tensor = torch.randn(3, 3)
        descriptor = index.add_tensor(tensor, name="test", dataset="default")
        
        assert descriptor.tensor_id is not None
        retrieved = index.get_tensor(descriptor.tensor_id)
        assert retrieved is not None
        assert torch.allclose(tensor, retrieved)
    
    def test_find_tensors_by_shape(self, index):
        """Test finding tensors by shape."""
        # Add tensors with different shapes
        index.add_tensor(torch.randn(2, 2), dataset="test")
        index.add_tensor(torch.randn(2, 2), dataset="test")
        index.add_tensor(torch.randn(3, 3), dataset="test")
        
        results = index.find_tensors_by_shape((2, 2), dataset="test")
        assert len(results) == 2
        
        results = index.find_tensors_by_shape((3, 3), dataset="test")
        assert len(results) == 1
    
    def test_find_similar_tensors_frobenius(self, index):
        """Test finding similar tensors using Frobenius distance."""
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        similar = torch.tensor([[1.1, 2.1], [3.1, 4.1]])  # Very similar
        different = torch.tensor([[10.0, 20.0], [30.0, 40.0]])  # Very different
        
        index.add_tensor(base, name="base")
        index.add_tensor(similar, name="similar")
        index.add_tensor(different, name="different")
        
        results = index.find_similar_tensors(
            base, metric=SimilarityMetric.FROBENIUS, top_k=3
        )
        
        assert len(results) == 3
        # The most similar should be the base itself or the similar tensor
        assert results[0].descriptor.name in ["base", "similar"]
    
    def test_find_similar_tensors_cosine(self, index):
        """Test finding similar tensors using cosine similarity."""
        base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        parallel = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # Same direction, different magnitude
        orthogonal = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])  # Orthogonal
        
        index.add_tensor(base, name="base")
        index.add_tensor(parallel, name="parallel")
        index.add_tensor(orthogonal, name="orthogonal")
        
        results = index.find_similar_tensors(
            base, metric=SimilarityMetric.COSINE, top_k=3
        )
        
        # Parallel should be more similar than orthogonal
        names = [r.descriptor.name for r in results]
        assert names.index("parallel") < names.index("orthogonal")
    
    def test_find_tensors_by_norm_range(self, index):
        """Test finding tensors by norm range."""
        small = torch.tensor([[0.1, 0.1], [0.1, 0.1]])  # Small norm
        medium = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # Medium norm
        large = torch.tensor([[10.0, 10.0], [10.0, 10.0]])  # Large norm
        
        index.add_tensor(small, name="small", dataset="test")
        index.add_tensor(medium, name="medium", dataset="test")
        index.add_tensor(large, name="large", dataset="test")
        
        results = index.find_tensors_by_norm_range(1.0, 5.0, dataset="test")
        assert len(results) == 1
        assert results[0].name == "medium"
    
    def test_find_tensors_by_property(self, index):
        """Test finding tensors by mathematical property."""
        symmetric = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        non_symmetric = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        index.add_tensor(symmetric, name="symmetric", dataset="test")
        index.add_tensor(non_symmetric, name="non_symmetric", dataset="test")
        
        results = index.find_tensors_by_property(TensorProperty.SYMMETRIC, dataset="test")
        assert len(results) == 1
        assert results[0].name == "symmetric"
    
    def test_find_matmul_compatible(self, index):
        """Test finding tensors compatible for matrix multiplication."""
        # Query is 2x3, so we need tensors with shape (3, *)
        index.add_tensor(torch.randn(3, 4), name="compatible", dataset="test")
        index.add_tensor(torch.randn(2, 3), name="incompatible", dataset="test")
        
        query = torch.randn(2, 3)
        results = index.find_matmul_compatible(query, dataset="test")
        
        assert len(results) == 1
        assert results[0].name == "compatible"
    
    def test_find_decomposition_candidates(self, index):
        """Test finding tensors suitable for decompositions."""
        square = torch.randn(4, 4)
        pd = torch.eye(3)  # Positive definite
        rectangular = torch.randn(3, 5)
        
        index.add_tensor(square, name="square", dataset="test")
        index.add_tensor(pd, name="pd", dataset="test")
        index.add_tensor(rectangular, name="rectangular", dataset="test")
        
        # Cholesky requires positive definite
        chol_candidates = index.find_decomposition_candidates("cholesky", dataset="test")
        assert len(chol_candidates) == 1
        assert chol_candidates[0].name == "pd"
        
        # Eigen requires square
        eigen_candidates = index.find_decomposition_candidates("eigen", dataset="test")
        assert len(eigen_candidates) == 2  # square and pd
    
    def test_remove_tensor(self, index):
        """Test removing a tensor from the index."""
        tensor = torch.randn(3, 3)
        descriptor = index.add_tensor(tensor, dataset="test")
        
        assert index.count_tensors() == 1
        
        result = index.remove_tensor(descriptor.tensor_id)
        assert result is True
        assert index.count_tensors() == 0
        assert index.get_tensor(descriptor.tensor_id) is None
    
    def test_statistics(self, index):
        """Test getting index statistics."""
        index.add_tensor(torch.randn(2, 2), dataset="test")
        index.add_tensor(torch.randn(2, 2), dataset="test")
        index.add_tensor(torch.randn(3, 3), dataset="test")
        
        stats = index.get_statistics(dataset="test")
        
        assert stats["count"] == 3
        assert stats["unique_shapes"] == 2
        assert (2, 2) in stats["shapes"]
        assert (3, 3) in stats["shapes"]


class TestGlobalIndex:
    """Tests for the global index singleton."""
    
    def test_get_global_index(self):
        """Test getting the global index."""
        idx1 = get_global_index()
        idx2 = get_global_index()
        assert idx1 is idx2


class TestTensorEmbeddingsIntegration:
    """Integration tests for Tensor Embeddings with SDK."""
    
    def test_sdk_tensor_embedding_integration(self):
        """Test tensor embeddings through the SDK."""
        from tensorus import Tensorus
        
        ts = Tensorus(enable_nql=False, enable_embeddings=False)
        
        # Create some tensors
        t1 = ts.create_tensor([[1.0, 2.0], [3.0, 4.0]], name="matrix_a", dataset="test")
        t2 = ts.create_tensor([[1.1, 2.1], [3.1, 4.1]], name="matrix_b", dataset="test")
        t3 = ts.create_tensor([[10.0, 20.0], [30.0, 40.0]], name="matrix_c", dataset="test")
        
        # Find similar tensors
        similar = ts.find_similar_tensors(t1, metric="frobenius", top_k=3)
        assert len(similar) == 3
        
        # Find by shape
        by_shape = ts.find_tensors_by_shape((2, 2), dataset="test")
        assert len(by_shape) == 3
        
        # Get descriptor
        desc = ts.get_tensor_descriptor(t1)
        assert desc is not None
        assert desc.shape == (2, 2)
        
        # Get stats
        stats = ts.get_tensor_embedding_stats(dataset="test")
        assert stats["count"] == 3
