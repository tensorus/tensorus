# test_vector_ops.py
"""
Tests for VectorOps class and vector similarity functions.
"""

import pytest
import torch
import numpy as np
from tensorus.vector_ops import VectorOps


class TestVectorOps:
    """Test cases for VectorOps class."""

    def test_cosine_similarity_single(self):
        """Test cosine similarity between two vectors."""
        vec1 = torch.tensor([1.0, 2.0, 3.0])
        vec2 = torch.tensor([4.0, 5.0, 6.0])
        
        similarity = VectorOps.cosine_similarity(vec1, vec2)
        expected = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
        
        assert torch.allclose(similarity, expected, atol=1e-6)

    def test_cosine_similarity_batch(self):
        """Test batch cosine similarity."""
        vectors1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        vectors2 = torch.tensor([[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]])
        
        similarities = VectorOps.cosine_similarity_batch(vectors1, vectors2)
        
        assert similarities.shape == (2,)
        assert torch.all(similarities >= -1.0) and torch.all(similarities <= 1.0)

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        vec1 = torch.tensor([0.0, 0.0, 0.0])
        vec2 = torch.tensor([3.0, 4.0, 0.0])
        
        distance = VectorOps.euclidean_distance(vec1, vec2)
        expected = 5.0  # 3-4-5 triangle
        
        assert torch.allclose(distance, torch.tensor(expected), atol=1e-6)

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        vec1 = torch.tensor([1.0, 2.0, 3.0])
        vec2 = torch.tensor([4.0, 6.0, 8.0])
        
        distance = VectorOps.manhattan_distance(vec1, vec2)
        expected = 3.0 + 4.0 + 5.0  # |1-4| + |2-6| + |3-8|
        
        assert torch.allclose(distance, torch.tensor(expected))

    def test_hamming_distance(self):
        """Test Hamming distance for binary vectors."""
        vec1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
        vec2 = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
        
        distance = VectorOps.hamming_distance(vec1, vec2)
        expected = 2  # positions 1 and 2 differ
        
        assert distance == expected

    def test_top_k_similarity(self):
        """Test top-k similarity search."""
        query = torch.tensor([1.0, 0.0, 0.0])
        vectors = torch.tensor([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.5, 0.5, 0.0],  # 45 degrees
            [-1.0, 0.0, 0.0]  # opposite
        ])
        
        indices, similarities = VectorOps.top_k_similarity(query, vectors, k=2)
        
        assert len(indices) == 2
        assert len(similarities) == 2
        assert indices[0] == 0  # most similar should be identical vector
        assert torch.allclose(similarities[0], torch.tensor(1.0), atol=1e-6)

    def test_normalize_vectors(self):
        """Test vector normalization."""
        vectors = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
        normalized = VectorOps.normalize_vectors(vectors)
        
        # Check that norms are 1
        norms = torch.norm(normalized, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-6)

    def test_vector_angle(self):
        """Test angle calculation between vectors."""
        vec1 = torch.tensor([1.0, 0.0])
        vec2 = torch.tensor([0.0, 1.0])
        
        angle = VectorOps.vector_angle(vec1, vec2)
        expected = torch.pi / 2  # 90 degrees
        
        assert torch.allclose(angle, torch.tensor(expected), atol=1e-6)

    def test_pairwise_distance_matrix(self):
        """Test pairwise distance matrix computation."""
        vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        distances = VectorOps.pairwise_distance_matrix(vectors, metric="euclidean")
        
        assert distances.shape == (3, 3)
        assert torch.allclose(distances.diagonal(), torch.zeros(3))  # diagonal should be 0
        assert torch.allclose(distances, distances.t())  # should be symmetric

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        vec1 = torch.tensor([1.0, 2.0])
        vec2 = torch.tensor([1.0, 2.0, 3.0])  # different dimension
        
        with pytest.raises(ValueError):
            VectorOps.cosine_similarity(vec1, vec2)

    def test_empty_tensors(self):
        """Test handling of empty tensors."""
        empty_tensor = torch.tensor([])
        
        with pytest.raises((ValueError, RuntimeError)):
            VectorOps.cosine_similarity(empty_tensor, empty_tensor)


if __name__ == "__main__":
    pytest.main([__file__])
