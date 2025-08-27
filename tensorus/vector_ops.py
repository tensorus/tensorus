# vector_ops.py
"""
Provides a library of vector operations for Tensorus vector database capabilities.

This module defines the `VectorOps` class containing methods for vector similarity,
distance calculations, and nearest neighbor operations. It extends the existing
tensor operations with specialized vector database functionality.

Integration Notes:
- Inherits from TensorOps for consistency with existing architecture
- Maintains backward compatibility with all existing tensor operations
- Optimized for 1D vector operations while supporting multi-dimensional tensors
"""

if __package__ in (None, ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tensorus"

import torch
import logging
from typing import Tuple, Optional, List, Union, Dict, Any
import numpy as np
from tensorus.tensor_ops import TensorOps

logger = logging.getLogger(__name__)

class VectorOps(TensorOps):
    """
    A static library class providing vector operations for similarity search and distance calculations.
    Extends TensorOps to maintain consistency with existing tensor operations architecture.
    All methods are static and operate on provided torch.Tensor objects.
    """

    @staticmethod
    def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two vectors.
        
        Cosine similarity measures the cosine of the angle between two vectors,
        ranging from -1 (opposite) to 1 (identical direction).
        
        Args:
            v1 (torch.Tensor): First vector tensor
            v2 (torch.Tensor): Second vector tensor
            
        Returns:
            torch.Tensor: Cosine similarity score as a scalar tensor
            
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If vectors have different shapes or are not 1D
            RuntimeError: If either vector has zero norm (undefined cosine similarity)
        """
        VectorOps._check_tensor(v1, v2)
        
        # Ensure vectors are 1D
        if v1.ndim != 1 or v2.ndim != 1:
            raise ValueError(f"Cosine similarity requires 1D vectors, got shapes {v1.shape} and {v2.shape}")
        
        # Ensure vectors have the same dimension
        if v1.shape[0] != v2.shape[0]:
            raise ValueError(f"Vector dimensions must match: {v1.shape[0]} vs {v2.shape[0]}")
        
        try:
            # Compute dot product
            dot_product = VectorOps.dot(v1, v2)
            
            # Compute norms
            norm_v1 = torch.linalg.norm(v1)
            norm_v2 = torch.linalg.norm(v2)
            
            # Check for zero norms
            if norm_v1 == 0 or norm_v2 == 0:
                logger.warning("Cosine similarity undefined for zero vectors")
                raise RuntimeError("Cosine similarity is undefined for zero-norm vectors")
            
            # Compute cosine similarity
            similarity = dot_product / (norm_v1 * norm_v2)
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            raise

    @staticmethod
    def euclidean_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Euclidean (L2) distance between two vectors.
        
        Args:
            v1 (torch.Tensor): First vector tensor
            v2 (torch.Tensor): Second vector tensor
            
        Returns:
            torch.Tensor: Euclidean distance as a scalar tensor
            
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If vectors have different shapes or are not 1D
        """
        VectorOps._check_tensor(v1, v2)
        
        # Ensure vectors are 1D
        if v1.ndim != 1 or v2.ndim != 1:
            raise ValueError(f"Euclidean distance requires 1D vectors, got shapes {v1.shape} and {v2.shape}")
        
        # Ensure vectors have the same dimension
        if v1.shape[0] != v2.shape[0]:
            raise ValueError(f"Vector dimensions must match: {v1.shape[0]} vs {v2.shape[0]}")
        
        try:
            # Compute L2 distance
            diff = v1 - v2
            distance = torch.linalg.norm(diff)
            return distance
            
        except Exception as e:
            logger.error(f"Error computing Euclidean distance: {e}")
            raise

    @staticmethod
    def manhattan_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Manhattan (L1) distance between two vectors.
        
        Args:
            v1 (torch.Tensor): First vector tensor
            v2 (torch.Tensor): Second vector tensor
            
        Returns:
            torch.Tensor: Manhattan distance as a scalar tensor
            
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If vectors have different shapes or are not 1D
        """
        VectorOps._check_tensor(v1, v2)
        
        # Ensure vectors are 1D
        if v1.ndim != 1 or v2.ndim != 1:
            raise ValueError(f"Manhattan distance requires 1D vectors, got shapes {v1.shape} and {v2.shape}")
        
        # Ensure vectors have the same dimension
        if v1.shape[0] != v2.shape[0]:
            raise ValueError(f"Vector dimensions must match: {v1.shape[0]} vs {v2.shape[0]}")
        
        try:
            # Compute L1 distance
            diff = torch.abs(v1 - v2)
            distance = torch.sum(diff)
            return distance
            
        except Exception as e:
            logger.error(f"Error computing Manhattan distance: {e}")
            raise

    @staticmethod
    def batch_cosine_similarity(query_vector: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between a query vector and a batch of vectors efficiently.
        
        Args:
            query_vector (torch.Tensor): Query vector of shape (d,)
            vectors (torch.Tensor): Batch of vectors of shape (n, d)
            
        Returns:
            torch.Tensor: Similarity scores of shape (n,)
            
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If dimensions don't match properly
        """
        VectorOps._check_tensor(query_vector, vectors)
        
        # Validate shapes
        if query_vector.ndim != 1:
            raise ValueError(f"Query vector must be 1D, got shape {query_vector.shape}")
        
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D (batch, features), got shape {vectors.shape}")
        
        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_vector.shape[0]} vs vectors {vectors.shape[1]}")
        
        try:
            # Normalize query vector
            query_norm = torch.linalg.norm(query_vector)
            if query_norm == 0:
                raise RuntimeError("Query vector has zero norm")
            query_normalized = query_vector / query_norm
            
            # Normalize batch vectors
            vector_norms = torch.linalg.norm(vectors, dim=1)
            zero_norm_mask = vector_norms == 0
            if zero_norm_mask.any():
                logger.warning(f"Found {zero_norm_mask.sum()} vectors with zero norm")
                # Set zero norm vectors to small epsilon to avoid division by zero
                vector_norms = torch.where(zero_norm_mask, torch.tensor(1e-8), vector_norms)
            
            vectors_normalized = vectors / vector_norms.unsqueeze(1)
            
            # Compute batch cosine similarity
            similarities = torch.matmul(vectors_normalized, query_normalized)
            
            # Set similarity to 0 for zero-norm vectors
            similarities = torch.where(zero_norm_mask, torch.tensor(0.0), similarities)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing batch cosine similarity: {e}")
            raise

    @staticmethod
    def batch_euclidean_distance(query_vector: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """
        Computes Euclidean distance between a query vector and a batch of vectors efficiently.
        
        Args:
            query_vector (torch.Tensor): Query vector of shape (d,)
            vectors (torch.Tensor): Batch of vectors of shape (n, d)
            
        Returns:
            torch.Tensor: Distance scores of shape (n,)
            
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If dimensions don't match properly
        """
        VectorOps._check_tensor(query_vector, vectors)
        
        # Validate shapes
        if query_vector.ndim != 1:
            raise ValueError(f"Query vector must be 1D, got shape {query_vector.shape}")
        
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D (batch, features), got shape {vectors.shape}")
        
        if query_vector.shape[0] != vectors.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_vector.shape[0]} vs vectors {vectors.shape[1]}")
        
        try:
            # Compute batch Euclidean distance
            diff = vectors - query_vector.unsqueeze(0)
            distances = torch.linalg.norm(diff, dim=1)
            return distances
            
        except Exception as e:
            logger.error(f"Error computing batch Euclidean distance: {e}")
            raise

    @staticmethod
    def top_k_similar(query_vector: torch.Tensor, vectors: torch.Tensor, k: int = 5, 
                     metric: str = "cosine") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finds the top-k most similar vectors to a query vector.
        
        Args:
            query_vector (torch.Tensor): Query vector of shape (d,)
            vectors (torch.Tensor): Batch of vectors of shape (n, d)
            k (int): Number of top results to return
            metric (str): Similarity metric ("cosine" or "euclidean")
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (scores, indices) where:
                - scores: Top-k similarity/distance scores
                - indices: Indices of top-k vectors in the original batch
                
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If k is invalid or metric is unsupported
        """
        VectorOps._check_tensor(query_vector, vectors)
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        if k > vectors.shape[0]:
            logger.warning(f"k ({k}) is larger than number of vectors ({vectors.shape[0]}), returning all vectors")
            k = vectors.shape[0]
        
        if metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'")
        
        try:
            if metric == "cosine":
                scores = VectorOps.batch_cosine_similarity(query_vector, vectors)
                # For cosine similarity, higher is better
                top_scores, top_indices = torch.topk(scores, k, largest=True)
            else:  # euclidean
                scores = VectorOps.batch_euclidean_distance(query_vector, vectors)
                # For distance, lower is better
                top_scores, top_indices = torch.topk(scores, k, largest=False)
            
            return top_scores, top_indices
            
        except Exception as e:
            logger.error(f"Error finding top-k similar vectors: {e}")
            raise

    @staticmethod
    def normalize_vector(vector: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        """
        Normalizes a vector using the specified norm.
        
        Args:
            vector (torch.Tensor): Input vector to normalize
            p (float): Norm type (1.0 for L1, 2.0 for L2, etc.)
            
        Returns:
            torch.Tensor: Normalized vector
            
        Raises:
            TypeError: If input is not a torch.Tensor
            ValueError: If vector is not 1D or has zero norm
        """
        VectorOps._check_tensor(vector)
        
        if vector.ndim != 1:
            raise ValueError(f"Input must be 1D vector, got shape {vector.shape}")
        
        try:
            norm = torch.linalg.norm(vector, ord=p)
            if norm == 0:
                logger.warning("Cannot normalize zero vector")
                raise ValueError("Cannot normalize vector with zero norm")
            
            normalized = vector / norm
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing vector: {e}")
            raise

    @staticmethod
    def vector_angle(v1: torch.Tensor, v2: torch.Tensor, degrees: bool = False) -> torch.Tensor:
        """
        Computes the angle between two vectors.
        
        Args:
            v1 (torch.Tensor): First vector tensor
            v2 (torch.Tensor): Second vector tensor
            degrees (bool): If True, return angle in degrees; otherwise radians
            
        Returns:
            torch.Tensor: Angle between vectors
            
        Raises:
            TypeError: If inputs are not torch.Tensor objects
            ValueError: If vectors have different shapes or are not 1D
        """
        try:
            # Use cosine similarity to compute angle
            cos_sim = VectorOps.cosine_similarity(v1, v2)
            
            # Clamp to avoid numerical errors with arccos
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            
            # Compute angle in radians
            angle_rad = torch.arccos(cos_sim)
            
            if degrees:
                angle_deg = torch.rad2deg(angle_rad)
                return angle_deg
            else:
                return angle_rad
                
        except Exception as e:
            logger.error(f"Error computing vector angle: {e}")
            raise
