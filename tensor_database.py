from typing import Dict, Tuple, List, Optional, Union, Any
import numpy as np
import os
import logging
import json
from pathlib import Path
import time

from tensor_data import TensorStorage
from tensor_indexer import TensorIndexer
from tensor_processor import TensorProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorDatabase:
    """
    Core database class that integrates storage, indexing, and processing.
    This is the main interface for interacting with the Tensorus system.
    """
    
    def __init__(self, 
                 storage_path: str = "data/tensor_db.h5",
                 index_path: Optional[str] = "data/tensor_index.pkl",
                 config_path: Optional[str] = "config/db_config.json",
                 use_gpu: bool = False):
        """
        Initialize the tensor database.
        
        Args:
            storage_path: Path to the HDF5 storage file
            index_path: Path to save/load the index
            config_path: Path to the database configuration file
            use_gpu: Whether to use GPU acceleration if available
        """
        self.config = self._load_config(config_path)
        self.use_gpu = use_gpu
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Initialize the storage layer
        self.storage = TensorStorage(filename=storage_path)
        
        # Initialize the processor
        self.processor = TensorProcessor(use_gpu=use_gpu)
        
        # Indexer will be initialized later when we know the dimension
        self.indexer = None
        self.index_path = index_path
        
        logger.info(f"TensorDatabase initialized with storage at {storage_path}")
        
        # Initialize metrics tracking
        self.metrics = {
            "queries": 0,
            "inserts": 0,
            "updates": 0,
            "deletes": 0,
            "search_time_total": 0,
            "insert_time_total": 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            "index_type": "flat",
            "metric": "l2",
            "default_dimension": 1024,
            "auto_index": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        
        logger.info("Using default configuration")
        return default_config
    
    def _init_indexer(self, dimension: int):
        """Initialize the indexer with the given dimension."""
        if self.indexer is None:
            self.indexer = TensorIndexer(
                dimension=dimension,
                index_type=self.config.get("index_type", "flat"),
                metric=self.config.get("metric", "l2"),
                use_gpu=self.use_gpu,
                index_filename=self.index_path
            )
            logger.info(f"Initialized indexer with dimension {dimension}")
    
    def save(self, 
             tensor: np.ndarray, 
             metadata: Optional[Dict[str, Any]] = None, 
             index: bool = None) -> str:
        """
        Save a tensor to the database.
        
        Args:
            tensor: Tensor data
            metadata: Optional metadata
            index: Whether to index the tensor (defaults to config setting)
            
        Returns:
            tensor_id: Unique ID for the stored tensor
        """
        start_time = time.time()
        
        # Default to config setting if not specified
        if index is None:
            index = self.config.get("auto_index", True)
        
        # Save to storage
        tensor_id = self.storage.save_tensor(tensor, metadata)
        
        # Index if requested
        if index:
            if self.indexer is None:
                dimension = tensor.size  # Use flattened tensor size
                self._init_indexer(dimension)
            
            self.indexer.add_tensor(tensor, tensor_id)
            logger.info(f"Indexed tensor {tensor_id}")
        
        # Update metrics
        self.metrics["inserts"] += 1
        self.metrics["insert_time_total"] += time.time() - start_time
        
        logger.info(f"Saved tensor {tensor_id}")
        return tensor_id
    
    def get(self, tensor_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Retrieve a tensor from the database.
        
        Args:
            tensor_id: The unique identifier of the tensor
            
        Returns:
            tensor: The tensor data
            metadata: The tensor metadata
        """
        self.metrics["queries"] += 1
        return self.storage.load_tensor(tensor_id)
    
    def search_similar(self, query_tensor: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tensors similar to the query.
        
        Args:
            query_tensor: The query tensor
            k: Number of results to return
            
        Returns:
            results: List of dictionaries with tensor_id, distance, and tensor data
        """
        start_time = time.time()
        
        if self.indexer is None:
            dimension = query_tensor.size
            self._init_indexer(dimension)
        
        tensor_ids, distances = self.indexer.search_tensor(query_tensor, k)
        
        results = []
        for tensor_id, distance in zip(tensor_ids, distances):
            tensor, metadata = self.storage.load_tensor(tensor_id)
            results.append({
                "tensor_id": tensor_id,
                "distance": distance,
                "tensor": tensor,
                "metadata": metadata
            })
        
        # Update metrics
        self.metrics["queries"] += 1
        self.metrics["search_time_total"] += time.time() - start_time
        
        return results
    
    def update(self, 
              tensor_id: str, 
              tensor: np.ndarray = None, 
              metadata: Dict[str, Any] = None) -> bool:
        """
        Update a tensor or its metadata.
        
        Args:
            tensor_id: The unique identifier of the tensor
            tensor: New tensor data (or None to keep existing)
            metadata: New metadata (or None to keep existing)
            
        Returns:
            success: True if update was successful
        """
        if tensor is None:
            # Only updating metadata
            current_tensor, _ = self.storage.load_tensor(tensor_id)
            success = self.storage.update_tensor(tensor_id, current_tensor, metadata)
        else:
            # Update both tensor and metadata
            success = self.storage.update_tensor(tensor_id, tensor, metadata)
            
            # Re-index if needed and if update was successful
            if success and self.indexer is not None and self.config.get("auto_index", True):
                self.indexer.add_tensor(tensor, tensor_id)
        
        if success:
            self.metrics["updates"] += 1
            
        return success
    
    def delete(self, tensor_id: str) -> bool:
        """
        Delete a tensor from the database.
        
        Args:
            tensor_id: The unique identifier of the tensor
            
        Returns:
            success: True if deletion was successful
        """
        success = self.storage.delete_tensor(tensor_id)
        
        if success:
            self.metrics["deletes"] += 1
            
        return success
    
    def list_tensors(self) -> List[Dict[str, Any]]:
        """
        List all tensors in the database.
        
        Returns:
            tensors: List of dictionaries with tensor IDs and metadata
        """
        self.metrics["queries"] += 1
        return self.storage.list_tensors()
    
    def process(self, 
               operation: str, 
               tensors: List[Union[str, np.ndarray]], 
               **kwargs) -> np.ndarray:
        """
        Process tensors using the specified operation.
        
        Args:
            operation: Name of the operation to perform
            tensors: List of tensor IDs or numpy arrays
            **kwargs: Additional arguments for the operation
            
        Returns:
            result: Result of the operation
        """
        # Load tensors if IDs were provided
        loaded_tensors = []
        for t in tensors:
            if isinstance(t, str):
                tensor_data, _ = self.storage.load_tensor(t)
                loaded_tensors.append(tensor_data)
            else:
                loaded_tensors.append(t)
        
        # Get the operation function
        op_func = getattr(self.processor, operation, None)
        if op_func is None:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Perform the operation
        if len(loaded_tensors) == 1:
            return op_func(loaded_tensors[0], **kwargs)
        else:
            return op_func(*loaded_tensors, **kwargs)
    
    def save_index(self) -> bool:
        """
        Save the current index to disk.
        
        Returns:
            success: True if save was successful
        """
        if self.indexer is None or self.index_path is None:
            logger.warning("No indexer or index path specified")
            return False
        
        return self.indexer.save_index(self.index_path)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get database performance metrics.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        metrics = dict(self.metrics)
        
        # Calculate averages
        if metrics["queries"] > 0:
            metrics["avg_search_time"] = metrics["search_time_total"] / max(1, metrics["queries"])
        
        if metrics["inserts"] > 0:
            metrics["avg_insert_time"] = metrics["insert_time_total"] / metrics["inserts"]
            
        return metrics
    
    def batch_save(self, 
                  tensors: List[np.ndarray], 
                  metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """
        Save multiple tensors in batch mode.
        
        Args:
            tensors: List of tensors to save
            metadatas: List of metadata dictionaries (optional)
            
        Returns:
            tensor_ids: List of unique IDs for the stored tensors
        """
        start_time = time.time()
        
        # Handle default metadata
        if metadatas is None:
            metadatas = [None] * len(tensors)
        elif len(metadatas) != len(tensors):
            raise ValueError("Number of tensors must match number of metadata dictionaries")
        
        # Save each tensor to storage
        tensor_ids = []
        for tensor, metadata in zip(tensors, metadatas):
            tensor_id = self.storage.save_tensor(tensor, metadata)
            tensor_ids.append(tensor_id)
        
        # Index all tensors if indexer exists
        if self.indexer is not None and self.config.get("auto_index", True):
            # Ensure all tensors have the same dimension when flattened
            dimensions = [t.size for t in tensors]
            if len(set(dimensions)) > 1:
                logger.warning(
                    "Tensors have different sizes when flattened. "
                    "This may cause issues with indexing. "
                    "Consider using individual save() calls."
                )
            else:
                self.indexer.batch_add_tensors(tensors, tensor_ids)
        
        # Update metrics
        self.metrics["inserts"] += len(tensors)
        self.metrics["insert_time_total"] += time.time() - start_time
        
        return tensor_ids 