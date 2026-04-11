"""
Tensor Embeddings - Native Tensor Storage with Tensor-Operation-Based Queries

This module implements Tensor Embeddings, which go beyond traditional vector embeddings
by storing native tensor data and enabling diverse queries using tensor operations.

Key Concepts:
- **Native Tensor Storage**: Store tensors in their original form, not just flattened vectors
- **Mathematical Properties**: Index tensors by norm, rank, eigenvalues, shape, sparsity
- **Tensor Similarity**: Find similar tensors using Frobenius norm, spectral similarity, etc.
- **Operation-Based Queries**: Find tensors suitable for specific mathematical operations
"""

import logging
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import UUID, uuid4
from enum import Enum

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported tensor similarity metrics."""
    FROBENIUS = "frobenius"       # Frobenius norm of difference
    COSINE = "cosine"             # Cosine similarity (flattened)
    SPECTRAL = "spectral"         # Spectral (eigenvalue-based) similarity
    RELATIVE_ERROR = "relative"   # Relative Frobenius error


class TensorProperty(Enum):
    """Mathematical properties for tensor filtering."""
    SYMMETRIC = "symmetric"
    POSITIVE_DEFINITE = "positive_definite"
    ORTHOGONAL = "orthogonal"
    SPARSE = "sparse"
    DENSE = "dense"
    SQUARE = "square"
    DIAGONAL = "diagonal"


@dataclass
class TensorDescriptor:
    """
    Describes a tensor's mathematical properties for embedding queries.
    
    This is the "embedding" of a tensor - a compact representation of its
    mathematical characteristics that enables efficient querying.
    """
    tensor_id: UUID
    shape: Tuple[int, ...]
    dtype: str
    
    # Scalar properties (pre-computed for efficient querying)
    frobenius_norm: float = 0.0
    l1_norm: float = 0.0
    l_inf_norm: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0
    sparsity: float = 0.0  # Fraction of zeros
    
    # Matrix-specific properties (only for 2D tensors)
    rank: Optional[int] = None
    trace: Optional[float] = None
    determinant: Optional[float] = None
    condition_number: Optional[float] = None
    
    # Eigenvalue summary (for square matrices)
    max_eigenvalue: Optional[float] = None
    min_eigenvalue: Optional[float] = None
    eigenvalue_spread: Optional[float] = None
    
    # Boolean properties
    is_symmetric: bool = False
    is_positive_definite: bool = False
    is_orthogonal: bool = False
    is_sparse: bool = False  # More than 50% zeros
    is_square: bool = False
    is_diagonal: bool = False
    
    # Metadata
    name: Optional[str] = None
    dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, tensor_id: Optional[UUID] = None,
                    name: Optional[str] = None, dataset: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> 'TensorDescriptor':
        """
        Create a TensorDescriptor from a PyTorch tensor by computing its properties.
        """
        if tensor_id is None:
            tensor_id = uuid4()
            
        # Convert to float for computations
        t = tensor.float()
        
        # Basic properties
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        
        # Compute norms
        frobenius_norm = torch.norm(t, p='fro').item() if t.numel() > 0 else 0.0
        l1_norm = torch.norm(t.flatten(), p=1).item() if t.numel() > 0 else 0.0
        l_inf_norm = torch.norm(t.flatten(), p=float('inf')).item() if t.numel() > 0 else 0.0
        
        # Statistics
        mean_value = t.mean().item() if t.numel() > 0 else 0.0
        std_value = t.std().item() if t.numel() > 1 else 0.0
        
        # Sparsity
        num_zeros = (t == 0).sum().item()
        sparsity = num_zeros / t.numel() if t.numel() > 0 else 0.0
        is_sparse = sparsity > 0.5
        
        # Matrix-specific properties
        rank = None
        trace = None
        determinant = None
        condition_number = None
        max_eigenvalue = None
        min_eigenvalue = None
        eigenvalue_spread = None
        is_symmetric = False
        is_positive_definite = False
        is_orthogonal = False
        is_square = False
        is_diagonal = False
        
        if len(shape) == 2:
            rows, cols = shape
            is_square = rows == cols
            
            # Compute rank for any 2D matrix (not just square)
            try:
                _, s, _ = torch.linalg.svd(t, full_matrices=False)
                rank = (s > 1e-10).sum().item()
                
                # Condition number
                if s[-1] > 1e-10:
                    condition_number = (s[0] / s[-1]).item()
            except Exception:
                pass
            
            if is_square and rows > 0:
                # Trace
                trace = torch.trace(t).item()
                
                # Check symmetry
                is_symmetric = torch.allclose(t, t.T, rtol=1e-5, atol=1e-8)
                
                # Check diagonal
                diag_mask = torch.eye(rows, dtype=torch.bool, device=t.device)
                is_diagonal = torch.allclose(t[~diag_mask], torch.zeros_like(t[~diag_mask]), rtol=1e-5, atol=1e-8)
                
                try:
                    # Determinant
                    determinant = torch.linalg.det(t).item()
                except Exception:
                    pass
                
                try:
                    # Eigenvalues (for square matrices)
                    if is_symmetric:
                        eigenvalues = torch.linalg.eigvalsh(t)
                    else:
                        eigenvalues = torch.linalg.eigvals(t).real
                    
                    max_eigenvalue = eigenvalues.max().item()
                    min_eigenvalue = eigenvalues.min().item()
                    eigenvalue_spread = max_eigenvalue - min_eigenvalue
                    
                    # Check positive definiteness
                    if is_symmetric and eigenvalues.min() > 0:
                        is_positive_definite = True
                except Exception:
                    pass
                
                try:
                    # Check orthogonality
                    product = torch.mm(t, t.T)
                    identity = torch.eye(rows, device=t.device)
                    is_orthogonal = torch.allclose(product, identity, rtol=1e-5, atol=1e-8)
                except Exception:
                    pass
        
        return cls(
            tensor_id=tensor_id,
            shape=shape,
            dtype=dtype,
            frobenius_norm=frobenius_norm,
            l1_norm=l1_norm,
            l_inf_norm=l_inf_norm,
            mean_value=mean_value,
            std_value=std_value,
            sparsity=sparsity,
            rank=rank,
            trace=trace,
            determinant=determinant,
            condition_number=condition_number,
            max_eigenvalue=max_eigenvalue,
            min_eigenvalue=min_eigenvalue,
            eigenvalue_spread=eigenvalue_spread,
            is_symmetric=is_symmetric,
            is_positive_definite=is_positive_definite,
            is_orthogonal=is_orthogonal,
            is_sparse=is_sparse,
            is_square=is_square,
            is_diagonal=is_diagonal,
            name=name,
            dataset=dataset,
            metadata=metadata or {}
        )


@dataclass
class TensorSimilarityResult:
    """Result of a tensor similarity search."""
    tensor_id: UUID
    descriptor: TensorDescriptor
    similarity_score: float
    rank: int
    tensor: Optional[torch.Tensor] = None  # Optional: the actual tensor data


class TensorEmbeddingIndex:
    """
    Index for efficient tensor embedding queries.
    
    This class maintains an index of TensorDescriptors and provides
    methods for finding similar tensors based on various metrics.
    """
    
    def __init__(self):
        self.descriptors: Dict[UUID, TensorDescriptor] = {}
        self.tensors: Dict[UUID, torch.Tensor] = {}
        self._shape_index: Dict[Tuple[int, ...], List[UUID]] = {}
        self._dataset_index: Dict[str, List[UUID]] = {}
        
    def add_tensor(self, tensor: torch.Tensor, tensor_id: Optional[UUID] = None,
                   name: Optional[str] = None, dataset: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> TensorDescriptor:
        """
        Add a tensor to the index.
        
        Args:
            tensor: PyTorch tensor to add
            tensor_id: Optional UUID (auto-generated if not provided)
            name: Optional name for the tensor
            dataset: Dataset name
            metadata: Additional metadata
            
        Returns:
            TensorDescriptor for the added tensor
        """
        descriptor = TensorDescriptor.from_tensor(
            tensor, tensor_id=tensor_id, name=name, 
            dataset=dataset, metadata=metadata
        )
        
        self.descriptors[descriptor.tensor_id] = descriptor
        self.tensors[descriptor.tensor_id] = tensor.clone()
        
        # Update shape index
        shape = descriptor.shape
        if shape not in self._shape_index:
            self._shape_index[shape] = []
        self._shape_index[shape].append(descriptor.tensor_id)
        
        # Update dataset index
        if dataset:
            if dataset not in self._dataset_index:
                self._dataset_index[dataset] = []
            self._dataset_index[dataset].append(descriptor.tensor_id)
        
        return descriptor
    
    def get_tensor(self, tensor_id: UUID) -> Optional[torch.Tensor]:
        """Get a tensor by its ID."""
        return self.tensors.get(tensor_id)
    
    def get_descriptor(self, tensor_id: UUID) -> Optional[TensorDescriptor]:
        """Get a tensor's descriptor by its ID."""
        return self.descriptors.get(tensor_id)
    
    def remove_tensor(self, tensor_id: UUID) -> bool:
        """Remove a tensor from the index."""
        if tensor_id not in self.descriptors:
            return False
        
        descriptor = self.descriptors[tensor_id]
        
        # Remove from shape index
        if descriptor.shape in self._shape_index:
            self._shape_index[descriptor.shape].remove(tensor_id)
            if not self._shape_index[descriptor.shape]:
                del self._shape_index[descriptor.shape]
        
        # Remove from dataset index
        if descriptor.dataset and descriptor.dataset in self._dataset_index:
            self._dataset_index[descriptor.dataset].remove(tensor_id)
            if not self._dataset_index[descriptor.dataset]:
                del self._dataset_index[descriptor.dataset]
        
        del self.descriptors[tensor_id]
        del self.tensors[tensor_id]
        
        return True
    
    def find_similar_tensors(
        self,
        query_tensor: torch.Tensor,
        metric: SimilarityMetric = SimilarityMetric.FROBENIUS,
        top_k: int = 10,
        dataset: Optional[str] = None,
        shape_filter: Optional[Tuple[int, ...]] = None,
        include_tensors: bool = False
    ) -> List[TensorSimilarityResult]:
        """
        Find tensors similar to the query tensor.
        
        Args:
            query_tensor: The tensor to compare against
            metric: Similarity metric to use
            top_k: Number of results to return
            dataset: Filter by dataset (optional)
            shape_filter: Only consider tensors with this shape (optional)
            include_tensors: Include actual tensor data in results
            
        Returns:
            List of TensorSimilarityResult sorted by similarity (descending)
        """
        query_flat = query_tensor.float().flatten()
        results = []
        
        # Get candidate tensor IDs
        if shape_filter:
            candidate_ids = set(self._shape_index.get(shape_filter, []))
        elif dataset and dataset in self._dataset_index:
            candidate_ids = set(self._dataset_index[dataset])
        else:
            candidate_ids = set(self.descriptors.keys())
        
        # Filter by dataset if specified
        if dataset and not shape_filter:
            dataset_ids = set(self._dataset_index.get(dataset, []))
            candidate_ids = candidate_ids.intersection(dataset_ids)
        
        for tensor_id in candidate_ids:
            stored_tensor = self.tensors.get(tensor_id)
            if stored_tensor is None:
                continue
            
            stored_flat = stored_tensor.float().flatten()
            
            # Compute similarity based on metric
            similarity = self._compute_similarity(query_flat, stored_flat, metric)
            
            results.append(TensorSimilarityResult(
                tensor_id=tensor_id,
                descriptor=self.descriptors[tensor_id],
                similarity_score=similarity,
                rank=0,  # Will be set after sorting
                tensor=stored_tensor if include_tensors else None
            ))
        
        # Sort by similarity (higher is better)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Assign ranks and limit to top_k
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]
    
    def _compute_similarity(
        self, 
        t1_flat: torch.Tensor, 
        t2_flat: torch.Tensor,
        metric: SimilarityMetric
    ) -> float:
        """Compute similarity between two flattened tensors."""
        
        # Handle size mismatch by padding
        if t1_flat.numel() != t2_flat.numel():
            max_size = max(t1_flat.numel(), t2_flat.numel())
            t1_padded = torch.zeros(max_size)
            t2_padded = torch.zeros(max_size)
            t1_padded[:t1_flat.numel()] = t1_flat
            t2_padded[:t2_flat.numel()] = t2_flat
            t1_flat, t2_flat = t1_padded, t2_padded
        
        if metric == SimilarityMetric.FROBENIUS:
            # Convert distance to similarity (1 / (1 + distance))
            distance = torch.norm(t1_flat - t2_flat, p=2).item()
            return 1.0 / (1.0 + distance)
        
        elif metric == SimilarityMetric.COSINE:
            norm1 = torch.norm(t1_flat, p=2)
            norm2 = torch.norm(t2_flat, p=2)
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.0
            dot_product = torch.dot(t1_flat, t2_flat)
            return (dot_product / (norm1 * norm2)).item()
        
        elif metric == SimilarityMetric.RELATIVE_ERROR:
            norm1 = torch.norm(t1_flat, p=2)
            if norm1 < 1e-10:
                return 0.0
            distance = torch.norm(t1_flat - t2_flat, p=2)
            relative_error = (distance / norm1).item()
            return 1.0 / (1.0 + relative_error)
        
        elif metric == SimilarityMetric.SPECTRAL:
            # For spectral similarity, compare singular values
            # This is a simplified version - reshape back to 2D if possible
            try:
                side = int(math.sqrt(t1_flat.numel()))
                if side * side == t1_flat.numel():
                    mat1 = t1_flat.reshape(side, side)
                    mat2 = t2_flat.reshape(side, side)
                    _, s1, _ = torch.linalg.svd(mat1, full_matrices=False)
                    _, s2, _ = torch.linalg.svd(mat2, full_matrices=False)
                    sv_distance = torch.norm(s1 - s2, p=2).item()
                    return 1.0 / (1.0 + sv_distance)
            except Exception:
                pass
            # Fall back to Frobenius if spectral fails
            distance = torch.norm(t1_flat - t2_flat, p=2).item()
            return 1.0 / (1.0 + distance)
        
        return 0.0
    
    def find_tensors_by_shape(
        self,
        shape: Tuple[int, ...],
        dataset: Optional[str] = None
    ) -> List[TensorDescriptor]:
        """Find all tensors with a specific shape."""
        tensor_ids = self._shape_index.get(shape, [])
        
        if dataset:
            dataset_ids = set(self._dataset_index.get(dataset, []))
            tensor_ids = [tid for tid in tensor_ids if tid in dataset_ids]
        
        return [self.descriptors[tid] for tid in tensor_ids]
    
    def find_tensors_by_norm_range(
        self,
        min_norm: float = 0.0,
        max_norm: float = float('inf'),
        norm_type: str = "frobenius",
        dataset: Optional[str] = None
    ) -> List[TensorDescriptor]:
        """Find tensors with norm in the specified range."""
        results = []
        
        for tensor_id, descriptor in self.descriptors.items():
            if dataset and descriptor.dataset != dataset:
                continue
            
            if norm_type == "frobenius":
                norm = descriptor.frobenius_norm
            elif norm_type == "l1":
                norm = descriptor.l1_norm
            elif norm_type == "linf":
                norm = descriptor.l_inf_norm
            else:
                norm = descriptor.frobenius_norm
            
            if min_norm <= norm <= max_norm:
                results.append(descriptor)
        
        return results
    
    def find_tensors_by_property(
        self,
        property_type: TensorProperty,
        dataset: Optional[str] = None
    ) -> List[TensorDescriptor]:
        """Find tensors with a specific mathematical property."""
        results = []
        
        for tensor_id, descriptor in self.descriptors.items():
            if dataset and descriptor.dataset != dataset:
                continue
            
            if property_type == TensorProperty.SYMMETRIC and descriptor.is_symmetric:
                results.append(descriptor)
            elif property_type == TensorProperty.POSITIVE_DEFINITE and descriptor.is_positive_definite:
                results.append(descriptor)
            elif property_type == TensorProperty.ORTHOGONAL and descriptor.is_orthogonal:
                results.append(descriptor)
            elif property_type == TensorProperty.SPARSE and descriptor.is_sparse:
                results.append(descriptor)
            elif property_type == TensorProperty.DENSE and not descriptor.is_sparse:
                results.append(descriptor)
            elif property_type == TensorProperty.SQUARE and descriptor.is_square:
                results.append(descriptor)
            elif property_type == TensorProperty.DIAGONAL and descriptor.is_diagonal:
                results.append(descriptor)
        
        return results
    
    def find_tensors_by_rank(
        self,
        rank: int,
        dataset: Optional[str] = None
    ) -> List[TensorDescriptor]:
        """Find matrices with a specific numerical rank."""
        results = []
        
        for tensor_id, descriptor in self.descriptors.items():
            if dataset and descriptor.dataset != dataset:
                continue
            
            if descriptor.rank == rank:
                results.append(descriptor)
        
        return results
    
    def find_matmul_compatible(
        self,
        query_tensor: torch.Tensor,
        dataset: Optional[str] = None
    ) -> List[TensorDescriptor]:
        """
        Find tensors that can be multiplied with the query tensor.
        
        For matrix multiplication A @ B, we need A.shape[-1] == B.shape[-2].
        """
        if len(query_tensor.shape) < 2:
            return []
        
        required_dim = query_tensor.shape[-1]
        results = []
        
        for tensor_id, descriptor in self.descriptors.items():
            if dataset and descriptor.dataset != dataset:
                continue
            
            if len(descriptor.shape) >= 2 and descriptor.shape[-2] == required_dim:
                results.append(descriptor)
        
        return results
    
    def find_decomposition_candidates(
        self,
        decomposition_type: str = "svd",
        min_size: int = 2,
        dataset: Optional[str] = None
    ) -> List[TensorDescriptor]:
        """
        Find tensors suitable for specific decompositions.
        
        Args:
            decomposition_type: Type of decomposition (svd, qr, cholesky, eigen)
            min_size: Minimum matrix size
            dataset: Filter by dataset
        """
        results = []
        
        for tensor_id, descriptor in self.descriptors.items():
            if dataset and descriptor.dataset != dataset:
                continue
            
            # Must be at least 2D
            if len(descriptor.shape) < 2:
                continue
            
            rows, cols = descriptor.shape[-2], descriptor.shape[-1]
            
            if decomposition_type == "svd":
                # SVD works on any 2D matrix
                if rows >= min_size and cols >= min_size:
                    results.append(descriptor)
            
            elif decomposition_type == "qr":
                # QR requires rows >= cols
                if rows >= cols >= min_size:
                    results.append(descriptor)
            
            elif decomposition_type == "cholesky":
                # Cholesky requires symmetric positive definite
                if descriptor.is_square and descriptor.is_positive_definite:
                    results.append(descriptor)
            
            elif decomposition_type == "eigen":
                # Eigendecomposition requires square matrices
                if descriptor.is_square and rows >= min_size:
                    results.append(descriptor)
        
        return results
    
    def list_datasets(self) -> List[str]:
        """List all datasets in the index."""
        return list(self._dataset_index.keys())
    
    def count_tensors(self, dataset: Optional[str] = None) -> int:
        """Count tensors in the index or a specific dataset."""
        if dataset:
            return len(self._dataset_index.get(dataset, []))
        return len(self.descriptors)
    
    def get_statistics(self, dataset: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the indexed tensors."""
        if dataset:
            tensor_ids = self._dataset_index.get(dataset, [])
            descriptors = [self.descriptors[tid] for tid in tensor_ids]
        else:
            descriptors = list(self.descriptors.values())
        
        if not descriptors:
            return {"count": 0}
        
        norms = [d.frobenius_norm for d in descriptors]
        
        return {
            "count": len(descriptors),
            "unique_shapes": len(set(d.shape for d in descriptors)),
            "shapes": list(set(d.shape for d in descriptors)),
            "avg_norm": sum(norms) / len(norms),
            "min_norm": min(norms),
            "max_norm": max(norms),
            "symmetric_count": sum(1 for d in descriptors if d.is_symmetric),
            "sparse_count": sum(1 for d in descriptors if d.is_sparse),
            "square_count": sum(1 for d in descriptors if d.is_square),
        }


# Global index instance for convenience
_global_index: Optional[TensorEmbeddingIndex] = None


def get_global_index() -> TensorEmbeddingIndex:
    """Get or create the global tensor embedding index."""
    global _global_index
    if _global_index is None:
        _global_index = TensorEmbeddingIndex()
    return _global_index
