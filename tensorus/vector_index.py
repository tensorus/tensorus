# vector_index.py
"""
Provides approximate nearest neighbor (ANN) indexing capabilities for Tensorus.

This module implements efficient vector indexing using FAISS for fast similarity
search operations. It integrates with the existing TensorStorage and maintains
compatibility with all tensor operations.

Features:
- Multiple index types (Flat, IVF, HNSW)
- Persistent index storage and loading
- Batch operations for efficiency
- Integration with existing metadata systems
"""

if __package__ in (None, ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tensorus"

import torch
import logging
import pickle
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
import uuid

from tensorus.tensor_storage import TensorStorage
from tensorus.vector_ops import VectorOps
from tensorus.metadata.schemas import VectorIndexInfo

logger = logging.getLogger(__name__)

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

class VectorIndex:
    """
    Approximate Nearest Neighbor index for fast vector similarity search.
    
    This class provides a high-level interface for building, saving, loading,
    and querying vector indexes using FAISS backend.
    """

    def __init__(self, 
                 dimension: int,
                 index_type: str = "flat",
                 metric: str = "cosine",
                 index_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a VectorIndex.
        
        Args:
            dimension: Dimensionality of vectors to index
            index_type: Type of index ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "euclidean", "manhattan")
            index_params: Additional parameters for index configuration
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
        
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.index_params = index_params or {}
        
        # Initialize FAISS index
        self.index = self._create_index()
        self.is_trained = False
        self.vector_ids: List[str] = []  # Track vector IDs for mapping
        self.metadata_map: Dict[int, Dict[str, Any]] = {}  # Map index position to metadata
        
        logger.info(f"Initialized {self.index_type} index with dimension {dimension}")

    def _create_index(self) -> Any:
        """Create and return a FAISS index based on configuration."""
        if self.index_type == "flat":
            if self.metric == "cosine":
                # Use inner product for cosine similarity (requires normalized vectors)
                index = faiss.IndexFlatIP(self.dimension)
            elif self.metric == "euclidean":
                index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported metric for flat index: {self.metric}")
                
        elif self.index_type == "ivf":
            # IVF (Inverted File) index for larger datasets
            nlist = self.index_params.get("nlist", min(100, max(1, self.dimension // 4)))
            
            if self.metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            elif self.metric == "euclidean":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                raise ValueError(f"Unsupported metric for IVF index: {self.metric}")
                
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) index
            m = self.index_params.get("m", 16)  # Number of connections
            
            if self.metric == "cosine":
                index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_INNER_PRODUCT)
            elif self.metric == "euclidean":
                index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_L2)
            else:
                raise ValueError(f"Unsupported metric for HNSW index: {self.metric}")
                
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index

    def add_vectors(self, 
                   vectors: torch.Tensor, 
                   vector_ids: List[str],
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            vectors: Tensor of vectors with shape (n, dimension)
            vector_ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector
        """
        if vectors.shape[0] != len(vector_ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Convert to numpy and ensure float32
        vectors_np = vectors.cpu().numpy().astype(np.float32)
        
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors_np)
        
        # Train index if necessary
        if not self.is_trained and self.index_type in ["ivf"]:
            logger.info("Training index...")
            self.index.train(vectors_np)
            self.is_trained = True
        
        # Add vectors to index
        start_idx = len(self.vector_ids)
        self.index.add(vectors_np)
        
        # Update ID mapping
        self.vector_ids.extend(vector_ids)
        
        # Update metadata mapping
        if metadata:
            for i, meta in enumerate(metadata):
                self.metadata_map[start_idx + i] = meta
        
        logger.debug(f"Added {len(vector_ids)} vectors to index")

    def search(self, 
               query_vectors: torch.Tensor,
               k: int = 5,
               return_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_vectors: Query vectors with shape (n_queries, dimension)
            k: Number of nearest neighbors to return
            return_metadata: Whether to include metadata in results
            
        Returns:
            List of search results for each query vector
        """
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Convert to numpy and ensure float32
        query_np = query_vectors.cpu().numpy().astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_np)
        
        # Perform search
        distances, indices = self.index.search(query_np, k)
        
        # Process results
        results = []
        for i in range(len(query_np)):
            query_results = []
            for j in range(k):
                idx = indices[i][j]
                if idx == -1:  # No more results
                    break
                
                distance = float(distances[i][j])
                
                # Convert distance to similarity for cosine metric
                if self.metric == "cosine":
                    similarity = distance  # FAISS returns inner product for IP metric
                else:
                    similarity = 1.0 / (1.0 + distance)  # Simple distance to similarity conversion
                
                result = {
                    "vector_id": self.vector_ids[idx],
                    "distance": distance,
                    "similarity": similarity,
                    "rank": j + 1
                }
                
                # Add metadata if requested
                if return_metadata and idx in self.metadata_map:
                    result["metadata"] = self.metadata_map[idx]
                
                query_results.append(result)
            
            results.append(query_results)
        
        return results

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save the index to disk.
        
        Args:
            file_path: Path to save the index
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(file_path.with_suffix('.faiss')))
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "index_params": self.index_params,
            "is_trained": self.is_trained,
            "vector_ids": self.vector_ids,
            "metadata_map": self.metadata_map,
            "created_at": datetime.utcnow().isoformat()
        }
        
        with open(file_path.with_suffix('.meta'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved index to {file_path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'VectorIndex':
        """
        Load an index from disk.
        
        Args:
            file_path: Path to load the index from
            
        Returns:
            Loaded VectorIndex instance
        """
        file_path = Path(file_path)
        
        # Load metadata
        with open(file_path.with_suffix('.meta'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
            index_params=metadata["index_params"]
        )
        
        # Load FAISS index
        instance.index = faiss.read_index(str(file_path.with_suffix('.faiss')))
        instance.is_trained = metadata["is_trained"]
        instance.vector_ids = metadata["vector_ids"]
        instance.metadata_map = metadata["metadata_map"]
        
        logger.info(f"Loaded index from {file_path}")
        return instance

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "total_vectors": len(self.vector_ids),
            "is_trained": self.is_trained,
            "index_params": self.index_params
        }

class VectorIndexManager:
    """
    Manager class for handling multiple vector indexes and integration with TensorStorage.
    """

    def __init__(self, 
                 tensor_storage: TensorStorage,
                 index_storage_path: Optional[str] = None):
        """
        Initialize VectorIndexManager.
        
        Args:
            tensor_storage: TensorStorage instance
            index_storage_path: Path to store index files
        """
        self.tensor_storage = tensor_storage
        self.index_storage_path = Path(index_storage_path) if index_storage_path else Path("vector_indexes")
        self.index_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.indexes: Dict[str, VectorIndex] = {}
        
        logger.info(f"VectorIndexManager initialized with storage path: {self.index_storage_path}")

    def build_index(self,
                   dataset_name: str,
                   index_name: Optional[str] = None,
                   index_type: str = "flat",
                   metric: str = "cosine",
                   index_params: Optional[Dict[str, Any]] = None,
                   embedding_filter: Optional[Dict[str, Any]] = None) -> VectorIndex:
        """
        Build a vector index from embeddings in a dataset.
        
        Args:
            dataset_name: Name of dataset containing embeddings
            index_name: Name for the index (defaults to dataset_name)
            index_type: Type of index to build
            metric: Distance metric to use
            index_params: Additional index parameters
            embedding_filter: Filter to select specific embeddings
            
        Returns:
            Built VectorIndex instance
        """
        index_name = index_name or f"{dataset_name}_{index_type}"
        
        # Get embeddings from dataset
        try:
            records = self.tensor_storage.get_dataset_with_metadata(dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")
        
        # Filter for embeddings
        embedding_records = []
        for record in records:
            metadata = record.get("metadata", {})
            if metadata.get("is_embedding", False):
                # Apply additional filters if specified
                if embedding_filter:
                    matches = all(
                        metadata.get(key) == value 
                        for key, value in embedding_filter.items()
                    )
                    if not matches:
                        continue
                embedding_records.append(record)
        
        if not embedding_records:
            raise RuntimeError(f"No embeddings found in dataset '{dataset_name}'")
        
        # Extract vectors and metadata
        vectors = torch.stack([record["tensor"] for record in embedding_records])
        vector_ids = [record["metadata"].get("record_id", str(uuid.uuid4())) for record in embedding_records]
        metadata_list = [record["metadata"] for record in embedding_records]
        
        # Create and build index
        dimension = vectors.shape[1]
        index = VectorIndex(
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            index_params=index_params
        )
        
        # Add vectors to index
        index.add_vectors(vectors, vector_ids, metadata_list)
        
        # Store index
        self.indexes[index_name] = index
        
        # Save to disk
        index_path = self.index_storage_path / index_name
        index.save(index_path)
        
        logger.info(f"Built index '{index_name}' with {len(embedding_records)} vectors")
        return index

    def load_index(self, index_name: str) -> VectorIndex:
        """
        Load an index from disk.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            Loaded VectorIndex instance
        """
        if index_name in self.indexes:
            return self.indexes[index_name]
        
        index_path = self.index_storage_path / index_name
        if not index_path.with_suffix('.meta').exists():
            raise FileNotFoundError(f"Index '{index_name}' not found")
        
        index = VectorIndex.load(index_path)
        self.indexes[index_name] = index
        
        return index

    def search_index(self,
                    index_name: str,
                    query: Union[str, torch.Tensor],
                    k: int = 5,
                    embedding_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search an index with a query.
        
        Args:
            index_name: Name of index to search
            query: Query text or vector
            k: Number of results to return
            embedding_model: Model to use for text encoding
            
        Returns:
            List of search results
        """
        # Load index if not in memory
        if index_name not in self.indexes:
            self.load_index(index_name)
        
        index = self.indexes[index_name]
        
        # Handle text query
        if isinstance(query, str):
            from tensorus.embedding_agent import EmbeddingAgent
            embedding_agent = EmbeddingAgent(self.tensor_storage)
            query_vector = embedding_agent.encode_text(query, embedding_model)
            query_vector = query_vector.squeeze(0)  # Remove batch dimension
        else:
            query_vector = query
        
        # Ensure query is 2D for search
        if query_vector.ndim == 1:
            query_vector = query_vector.unsqueeze(0)
        
        # Perform search
        results = index.search(query_vector, k=k, return_metadata=True)
        
        # Return results for first (and only) query
        return results[0] if results else []

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all available indexes."""
        indexes_info = []
        
        # Check disk for saved indexes
        for index_file in self.index_storage_path.glob("*.meta"):
            index_name = index_file.stem
            
            try:
                if index_name in self.indexes:
                    # Get stats from loaded index
                    stats = self.indexes[index_name].get_stats()
                else:
                    # Load metadata only
                    with open(index_file, 'rb') as f:
                        metadata = pickle.load(f)
                    stats = {
                        "dimension": metadata["dimension"],
                        "index_type": metadata["index_type"],
                        "metric": metadata["metric"],
                        "total_vectors": len(metadata["vector_ids"]),
                        "is_trained": metadata["is_trained"]
                    }
                
                stats["name"] = index_name
                stats["loaded"] = index_name in self.indexes
                indexes_info.append(stats)
                
            except Exception as e:
                logger.warning(f"Failed to read index metadata for '{index_name}': {e}")
        
        return indexes_info

    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index from memory and disk.
        
        Args:
            index_name: Name of index to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Remove from memory
            if index_name in self.indexes:
                del self.indexes[index_name]
            
            # Remove from disk
            index_path = self.index_storage_path / index_name
            for suffix in ['.faiss', '.meta']:
                file_path = index_path.with_suffix(suffix)
                if file_path.exists():
                    file_path.unlink()
            
            logger.info(f"Deleted index '{index_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index '{index_name}': {e}")
            return False
