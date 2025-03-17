import faiss
import numpy as np
import pickle
import os
import logging
from typing import Dict, Tuple, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorIndexer:
    """
    Indexing layer for tensor data using FAISS.
    Provides functionality for adding tensors to an index and performing similarity searches.
    """
    
    def __init__(self, 
                 dimension: int, 
                 index_type: str = "flat", 
                 metric: str = "l2",
                 use_gpu: bool = False,
                 index_filename: Optional[str] = None):
        """
        Initialize the tensor indexer.
        
        Args:
            dimension: The dimension of the flattened tensor vectors
            index_type: Type of index ('flat', 'ivf', 'hnsw', etc.)
            metric: Distance metric ('l2' for Euclidean, 'ip' for inner product)
            use_gpu: Whether to use GPU acceleration if available
            index_filename: Path to load an existing index from
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.tensor_ids = []  # Maps index positions to tensor IDs
        
        # Create or load the index
        if index_filename and os.path.exists(index_filename):
            self._load_index(index_filename)
        else:
            self._create_index(dimension, index_type, metric, use_gpu)
    
    def _create_index(self, dimension: int, index_type: str, metric: str, use_gpu: bool):
        """Create a new FAISS index based on specified parameters."""
        if metric == "l2":
            if index_type == "flat":
                self.index = faiss.IndexFlatL2(dimension)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
                self.index.train(np.random.random((1000, dimension)).astype('float32'))
            elif index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
            else:
                logger.warning(f"Unsupported index type {index_type}, falling back to flat")
                self.index = faiss.IndexFlatL2(dimension)
        elif metric == "ip":  # Inner product (cosine similarity with normalized vectors)
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                logger.warning(f"Unsupported combination: {index_type} with {metric}, falling back to flat")
                self.index = faiss.IndexFlatIP(dimension)
        else:
            logger.warning(f"Unsupported metric {metric}, falling back to L2")
            self.index = faiss.IndexFlatL2(dimension)
            
        # Move to GPU if requested and available
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Successfully moved index to GPU")
            except Exception as e:
                logger.warning(f"Failed to use GPU: {e}")
    
    def _load_index(self, filename: str):
        """Load an existing index from disk."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.tensor_ids = data['tensor_ids']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
                self.metric = data['metric']
            logger.info(f"Loaded index from {filename}")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._create_index(self.dimension, self.index_type, self.metric, False)
    
    def save_index(self, filename: str) -> bool:
        """
        Save the current index to disk.
        
        Args:
            filename: Path to save the index
            
        Returns:
            success: True if save was successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save index and metadata
            with open(filename, 'wb') as f:
                pickle.dump({
                    'index': faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index,
                    'tensor_ids': self.tensor_ids,
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'metric': self.metric
                }, f)
            logger.info(f"Saved index to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def add_tensor(self, tensor: np.ndarray, tensor_id: str) -> bool:
        """
        Add a tensor to the index.
        
        Args:
            tensor: The tensor to add (will be flattened)
            tensor_id: Unique identifier for the tensor
            
        Returns:
            success: True if addition was successful
        """
        try:
            # Flatten and convert to float32
            flat_tensor = tensor.flatten().astype('float32').reshape(1, -1)
            
            # Verify dimensions
            if flat_tensor.shape[1] != self.dimension:
                raise ValueError(
                    f"Tensor dimension mismatch: expected {self.dimension}, got {flat_tensor.shape[1]}"
                )
                
            # Add to index
            self.index.add(flat_tensor)
            self.tensor_ids.append(tensor_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add tensor to index: {e}")
            return False
    
    def search_tensor(self, query_tensor: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        """
        Search for similar tensors in the index.
        
        Args:
            query_tensor: The query tensor (will be flattened)
            k: Number of nearest neighbors to return
            
        Returns:
            tensor_ids: List of tensor IDs for the nearest neighbors
            distances: List of distances to the nearest neighbors
        """
        # Flatten and convert to float32
        flat_query = query_tensor.flatten().astype('float32').reshape(1, -1)
        
        # Verify dimensions
        if flat_query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {flat_query.shape[1]}"
            )
        
        # Search
        distances, indices = self.index.search(flat_query, min(k, len(self.tensor_ids)))
        
        # Map indices to tensor IDs
        result_ids = [self.tensor_ids[int(i)] for i in indices[0] if i >= 0 and i < len(self.tensor_ids)]
        result_distances = distances[0].tolist()
        
        return result_ids, result_distances

    def batch_add_tensors(self, tensors: np.ndarray, tensor_ids: List[str]) -> bool:
        """
        Add multiple tensors to the index in batch mode.
        
        Args:
            tensors: Array of tensors to add (will be flattened)
            tensor_ids: List of unique identifiers for the tensors
            
        Returns:
            success: True if all additions were successful
        """
        try:
            if len(tensors) != len(tensor_ids):
                raise ValueError("Number of tensors must match number of tensor IDs")
                
            # Flatten and convert to float32
            flat_tensors = np.vstack([t.flatten().astype('float32') for t in tensors])
            
            # Verify dimensions
            if flat_tensors.shape[1] != self.dimension:
                raise ValueError(
                    f"Tensor dimension mismatch: expected {self.dimension}, got {flat_tensors.shape[1]}"
                )
                
            # Add to index
            self.index.add(flat_tensors)
            self.tensor_ids.extend(tensor_ids)
            return True
        except Exception as e:
            logger.error(f"Failed to batch add tensors to index: {e}")
            return False 