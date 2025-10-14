"""
Tensorus Unified SDK

This module provides a high-level, unified interface to all Tensorus components,
making it easy to work with tensor storage, agents, vector search, and operations.
"""

import logging
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import UUID
from pathlib import Path

from .tensor_storage import TensorStorage, TensorNotFoundError, DatasetNotFoundError
from .tensor_ops import TensorOps
from .vector_database import PartitionedVectorIndex, VectorMetadata, SearchResult
from .embedding_agent import EmbeddingAgent
from .nql_agent import NQLAgent
from .ingestion_agent import DataIngestionAgent
from .rl_agent import RLAgent
from .automl_agent import AutoMLAgent
from .agent_orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class TensorWrapper:
    """Wrapper class for tensors with metadata and operations."""
    
    def __init__(self, data: torch.Tensor, tensor_id: Optional[UUID] = None,
                 name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                 description: Optional[str] = None, storage_ref: Optional['Tensorus'] = None):
        self._data = data
        self.id = tensor_id
        self.name = name
        self.metadata = metadata or {}
        self.description = description
        self._storage_ref = storage_ref
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return tuple(self._data.shape)
    
    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the tensor."""
        return self._data.dtype
    
    @property
    def device(self) -> torch.device:
        """Return the device where tensor is stored."""
        return self._data.device
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._data.cpu().numpy()
    
    def to_tensor(self) -> torch.Tensor:
        """Get the underlying PyTorch tensor."""
        return self._data
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update tensor metadata."""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates like "processing.augmented"
                keys = key.split('.')
                current = self.metadata
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                self.metadata[key] = value
        
        # Sync with storage if available
        if self._storage_ref and self.id:
            logger.debug(f"Metadata update for tensor {self.id} would sync to storage")
    
    def transpose(self, dim0: int = 0, dim1: int = 1) -> 'TensorWrapper':
        """Transpose the tensor."""
        transposed_data = self._data.transpose(dim0, dim1)
        return TensorWrapper(transposed_data, name=f"{self.name}_T" if self.name else None,
                           metadata=self.metadata.copy(), storage_ref=self._storage_ref)
    
    def __repr__(self) -> str:
        return f"TensorWrapper(shape={self.shape}, dtype={self.dtype}, name={self.name})"
    
    def __add__(self, other: Union['TensorWrapper', torch.Tensor, float]) -> 'TensorWrapper':
        """Add tensors."""
        other_data = other._data if isinstance(other, TensorWrapper) else other
        result = TensorOps.add(self._data, other_data)
        return TensorWrapper(result, storage_ref=self._storage_ref)
    
    def __sub__(self, other: Union['TensorWrapper', torch.Tensor, float]) -> 'TensorWrapper':
        """Subtract tensors."""
        other_data = other._data if isinstance(other, TensorWrapper) else other
        result = TensorOps.subtract(self._data, other_data)
        return TensorWrapper(result, storage_ref=self._storage_ref)
    
    def __mul__(self, other: Union['TensorWrapper', torch.Tensor, float]) -> 'TensorWrapper':
        """Multiply tensors."""
        other_data = other._data if isinstance(other, TensorWrapper) else other
        result = TensorOps.multiply(self._data, other_data)
        return TensorWrapper(result, storage_ref=self._storage_ref)


class SearchResults:
    """Wrapper for search results."""
    
    def __init__(self, ids: List[str], scores: List[float], 
                 metadata: Optional[List[Dict[str, Any]]] = None):
        self.ids = ids
        self.scores = scores
        self.metadata = metadata or [{} for _ in ids]
    
    def __repr__(self) -> str:
        return f"SearchResults(count={len(self.ids)})"
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __iter__(self):
        return zip(self.ids, self.scores, self.metadata)


class Tensorus:
    """
    Unified interface to the Tensorus agentic tensor database.
    
    This class provides high-level methods for:
    - Creating and managing tensors
    - Vector similarity search
    - Natural language queries
    - Autonomous agent operations
    - Tensor operations and transformations
    
    Example:
        >>> ts = Tensorus()
        >>> tensor = ts.create_tensor([[1, 2], [3, 4]], name="matrix_a")
        >>> result = ts.matmul(tensor, tensor)
    """
    
    def __init__(self,
                 storage_path: Optional[str] = None,
                 storage_backend: str = "in_memory",
                 enable_nql: bool = True,
                 enable_embeddings: bool = True,
                 enable_vector_search: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 **kwargs):
        """
        Initialize Tensorus with specified configuration.
        
        Args:
            storage_path: Path for persistent storage (file or S3)
            storage_backend: Backend type ("in_memory", "postgres", "s3")
            enable_nql: Enable Natural Query Language agent
            enable_embeddings: Enable embedding generation
            enable_vector_search: Enable vector similarity search
            embedding_model: Model to use for embeddings
            **kwargs: Additional configuration options
        """
        logger.info(f"Initializing Tensorus SDK with backend: {storage_backend}")
        
        # Initialize core storage
        self.storage = TensorStorage(
            storage_path=storage_path,
            enable_compression=kwargs.get("enable_compression", True),
            enable_indexing=kwargs.get("enable_indexing", True)
        )
        
        # Initialize agents
        self._nql_agent = None
        self._embedding_agent = None
        self._ingestion_agent = None
        self._rl_agent = None
        self._automl_agent = None
        
        if enable_nql:
            try:
                self._nql_agent = NQLAgent(
                    self.storage,
                    use_llm=kwargs.get("use_llm", False),
                    llm_model=kwargs.get("llm_model", "gemini-2.0-flash-exp")
                )
                logger.info("NQL Agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NQL Agent: {e}")
        
        if enable_embeddings:
            try:
                self._embedding_agent = EmbeddingAgent(
                    self.storage,
                    model_name=embedding_model,
                    enable_caching=kwargs.get("enable_caching", True)
                )
                logger.info(f"Embedding Agent initialized with model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Embedding Agent: {e}")
        
        # Vector indexes storage
        self._vector_indexes: Dict[str, PartitionedVectorIndex] = {}
        
        # Agent orchestrator
        self._orchestrator = None
        if kwargs.get("enable_orchestrator", True):
            self._orchestrator = AgentOrchestrator(self.storage)
            # Auto-register agents
            if self._nql_agent:
                self._orchestrator.register_nql_agent(self._nql_agent)
            if self._embedding_agent:
                self._orchestrator.register_embedding_agent(self._embedding_agent)
            logger.info("Agent Orchestrator initialized")
        
        logger.info("Tensorus SDK initialized successfully")
    
    # ==================== Tensor Creation & Management ====================
    
    def create_tensor(self,
                     data: Union[np.ndarray, torch.Tensor, List],
                     name: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     description: Optional[str] = None,
                     dataset: str = "default") -> TensorWrapper:
        """
        Create a new tensor with optional metadata.
        
        Args:
            data: Tensor data (numpy array, torch tensor, or list)
            name: Human-readable name for the tensor
            metadata: Additional metadata to attach
            description: Text description of the tensor
            dataset: Dataset name to store the tensor in
            
        Returns:
            TensorWrapper object
        """
        # Convert to torch tensor if needed
        if isinstance(data, np.ndarray):
            tensor_data = torch.from_numpy(data)
        elif isinstance(data, list):
            tensor_data = torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            tensor_data = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Ensure dataset exists
        try:
            self.storage.get_dataset(dataset)
        except (ValueError, DatasetNotFoundError):
            self.storage.create_dataset(dataset)
        
        # Store in tensor storage
        full_metadata = metadata or {}
        if name:
            full_metadata["name"] = name
        if description:
            full_metadata["description"] = description
        
        tensor_id = self.storage.insert(
            dataset_name=dataset,
            tensor=tensor_data,
            metadata=full_metadata
        )
        
        logger.debug(f"Created tensor {tensor_id} in dataset '{dataset}'")
        
        return TensorWrapper(
            tensor_data,
            tensor_id=tensor_id,
            name=name,
            metadata=full_metadata,
            description=description,
            storage_ref=self
        )
    
    def get_tensor(self, tensor_id: UUID, dataset: str = "default") -> TensorWrapper:
        """Retrieve a tensor by ID."""
        result = self.storage.get(dataset, tensor_id)
        if result is None:
            raise TensorNotFoundError(f"Tensor {tensor_id} not found in dataset '{dataset}'")
        
        tensor_data, metadata = result
        return TensorWrapper(
            tensor_data,
            tensor_id=tensor_id,
            name=metadata.get("name"),
            metadata=metadata,
            description=metadata.get("description"),
            storage_ref=self
        )
    
    def list_tensors(self, dataset: str = "default") -> List[Dict[str, Any]]:
        """List all tensors in a dataset."""
        try:
            dataset_info = self.storage.get_dataset(dataset)
            return [
                {
                    "id": tid,
                    "metadata": dataset_info["tensors"][tid]["metadata"]
                }
                for tid in dataset_info["tensors"]
            ]
        except (ValueError, DatasetNotFoundError):
            return []
    
    def delete_tensor(self, tensor_id: UUID, dataset: str = "default") -> bool:
        """Delete a tensor from storage."""
        return self.storage.delete(dataset, tensor_id)
    
    # ==================== Vector Database Operations ====================
    
    def create_index(self, index_name: str, dimensions: int,
                    metric: str = "cosine", use_partitioning: bool = True) -> None:
        """
        Create a vector index for similarity search.
        
        Args:
            index_name: Name of the index
            dimensions: Dimensionality of vectors
            metric: Distance metric ("cosine", "euclidean", "dot")
            use_partitioning: Use geometric partitioning for scaling
        """
        if index_name in self._vector_indexes:
            logger.warning(f"Index '{index_name}' already exists")
            return
        
        # Create partitioned vector index
        index = PartitionedVectorIndex(
            dimensions=dimensions,
            metric=metric,
            num_partitions=8,  # Reasonable default
            enable_freshness_layer=True
        )
        
        self._vector_indexes[index_name] = index
        logger.info(f"Created vector index '{index_name}' with {dimensions} dimensions")
    
    def index_exists(self, index_name: str) -> bool:
        """Check if a vector index exists."""
        return index_name in self._vector_indexes
    
    def add_vectors(self,
                   index_name: str,
                   vector_ids: List[str],
                   vectors: Union[np.ndarray, List[List[float]]],
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add vectors to an index.
        
        Args:
            index_name: Name of the index
            vector_ids: List of vector IDs
            vectors: Vector embeddings (2D array)
            metadata: Optional metadata for each vector
        """
        if index_name not in self._vector_indexes:
            raise ValueError(f"Index '{index_name}' does not exist. Create it first.")
        
        # Convert to numpy if needed
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        elif isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().numpy().astype(np.float32)
        
        # Prepare metadata
        if metadata is None:
            metadata = [{"id": vid} for vid in vector_ids]
        else:
            for i, meta in enumerate(metadata):
                meta["id"] = vector_ids[i]
        
        # Convert to VectorMetadata objects
        vector_metadata = [
            VectorMetadata(
                vector_id=vector_ids[i],
                vector=vectors[i],
                metadata=metadata[i]
            )
            for i in range(len(vector_ids))
        ]
        
        # Add to index
        index = self._vector_indexes[index_name]
        index.add(vector_metadata)
        
        logger.info(f"Added {len(vector_ids)} vectors to index '{index_name}'")
    
    def search_vectors(self,
                      index_name: str,
                      query: Union[np.ndarray, List[float], torch.Tensor],
                      k: int = 10,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> SearchResults:
        """
        Search for similar vectors in an index.
        
        Args:
            index_name: Name of the index to search
            query: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            SearchResults object with ids, scores, and metadata
        """
        if index_name not in self._vector_indexes:
            raise ValueError(f"Index '{index_name}' does not exist")
        
        # Convert query to numpy
        if isinstance(query, list):
            query = np.array(query, dtype=np.float32)
        elif isinstance(query, torch.Tensor):
            query = query.cpu().numpy().astype(np.float32)
        
        # Search the index
        index = self._vector_indexes[index_name]
        results = index.search(query, k=k, filter_fn=None)  # TODO: Implement filter_fn
        
        # Extract results
        ids = [r.vector_id for r in results]
        scores = [r.score for r in results]
        metadata = [r.metadata for r in results]
        
        return SearchResults(ids, scores, metadata)
    
    def delete_index(self, index_name: str) -> None:
        """Delete a vector index."""
        if index_name in self._vector_indexes:
            del self._vector_indexes[index_name]
            logger.info(f"Deleted vector index '{index_name}'")
    
    # ==================== Embedding Operations ====================
    
    def generate_embeddings(self,
                          texts: Union[str, List[str]],
                          model: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            model: Optional model name override
            
        Returns:
            Numpy array of embeddings
        """
        if self._embedding_agent is None:
            raise RuntimeError("Embedding agent not initialized. Set enable_embeddings=True")
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._embedding_agent.generate_embeddings(texts)
        return embeddings
    
    def embed_and_index(self,
                       texts: Union[str, List[str]],
                       index_name: str,
                       ids: Optional[List[str]] = None,
                       metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Generate embeddings and add them to an index in one step.
        
        Args:
            texts: Text(s) to embed
            index_name: Target index name
            ids: Optional custom IDs (auto-generated if None)
            metadata: Optional metadata for each text
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"text_{i}" for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create index if it doesn't exist
        if not self.index_exists(index_name):
            self.create_index(index_name, dimensions=embeddings.shape[1])
        
        # Add to index
        self.add_vectors(index_name, ids, embeddings, metadata)
        
        logger.info(f"Embedded and indexed {len(texts)} texts in '{index_name}'")
    
    # ==================== Natural Query Language ====================
    
    def query(self, query_text: str, dataset: Optional[str] = None) -> List[Tuple[UUID, torch.Tensor, Dict]]:
        """
        Execute a natural language query.
        
        Args:
            query_text: Natural language query
            dataset: Optional dataset to query (None for all)
            
        Returns:
            List of (tensor_id, tensor, metadata) tuples
        """
        if self._nql_agent is None:
            raise RuntimeError("NQL agent not initialized. Set enable_nql=True")
        
        results = self._nql_agent.process_query(query_text)
        return results
    
    # ==================== Tensor Operations ====================
    
    def matmul(self, a: Union[TensorWrapper, torch.Tensor],
               b: Union[TensorWrapper, torch.Tensor]) -> Union[TensorWrapper, torch.Tensor]:
        """Matrix multiplication."""
        a_data = a._data if isinstance(a, TensorWrapper) else a
        b_data = b._data if isinstance(b, TensorWrapper) else b
        
        result = TensorOps.matmul(a_data, b_data)
        
        if isinstance(a, TensorWrapper):
            return TensorWrapper(result, storage_ref=self)
        return result
    
    def stack(self, tensors: List[TensorWrapper], axis: int = 0) -> TensorWrapper:
        """Stack tensors along a new dimension."""
        tensor_data = [t._data if isinstance(t, TensorWrapper) else t for t in tensors]
        stacked = torch.stack(tensor_data, dim=axis)
        return TensorWrapper(stacked, storage_ref=self)
    
    def svd(self, tensor: Union[TensorWrapper, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Singular Value Decomposition."""
        data = tensor._data if isinstance(tensor, TensorWrapper) else tensor
        return TensorOps.svd(data)
    
    def reshape(self, tensor: Union[TensorWrapper, torch.Tensor],
               shape: Tuple[int, ...]) -> Union[TensorWrapper, torch.Tensor]:
        """Reshape tensor."""
        data = tensor._data if isinstance(tensor, TensorWrapper) else tensor
        result = TensorOps.reshape(data, shape)
        
        if isinstance(tensor, TensorWrapper):
            return TensorWrapper(result, storage_ref=self)
        return result
    
    def transpose(self, tensor: Union[TensorWrapper, torch.Tensor],
                 dim0: int = 0, dim1: int = 1) -> Union[TensorWrapper, torch.Tensor]:
        """Transpose tensor."""
        data = tensor._data if isinstance(tensor, TensorWrapper) else tensor
        result = TensorOps.transpose(data, dim0, dim1)
        
        if isinstance(tensor, TensorWrapper):
            return TensorWrapper(result, storage_ref=self)
        return result
    
    # ==================== Metadata Search ====================
    
    def search_metadata(self, filters: Dict[str, Any], dataset: str = "default") -> List[TensorWrapper]:
        """
        Search tensors by metadata criteria.
        
        Args:
            filters: Dictionary of metadata filters (e.g., {"tags": "example"})
            dataset: Dataset to search
            
        Returns:
            List of matching TensorWrapper objects
        """
        results = []
        
        try:
            dataset_info = self.storage.get_dataset(dataset)
            
            for tensor_id, tensor_info in dataset_info["tensors"].items():
                metadata = tensor_info["metadata"]
                
                # Check if metadata matches all filters
                match = True
                for key, value in filters.items():
                    if key in metadata:
                        meta_value = metadata[key]
                        # Handle tags specially (check if value in list)
                        if isinstance(meta_value, list):
                            if value not in meta_value:
                                match = False
                                break
                        elif meta_value != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    tensor_data = self.storage.get(dataset, tensor_id)[0]
                    results.append(TensorWrapper(
                        tensor_data,
                        tensor_id=tensor_id,
                        name=metadata.get("name"),
                        metadata=metadata,
                        description=metadata.get("description"),
                        storage_ref=self
                    ))
        
        except (ValueError, DatasetNotFoundError):
            logger.warning(f"Dataset '{dataset}' not found")
        
        return results
    
    # ==================== Agent Operations ====================
    
    def start_ingestion(self,
                       dataset_name: str,
                       source_directory: str,
                       polling_interval: int = 10) -> DataIngestionAgent:
        """
        Start autonomous data ingestion from a directory.
        
        Args:
            dataset_name: Target dataset
            source_directory: Directory to monitor
            polling_interval: Seconds between checks
            
        Returns:
            DataIngestionAgent instance
        """
        agent = DataIngestionAgent(
            self.storage,
            dataset_name,
            source_directory,
            polling_interval_sec=polling_interval
        )
        self._ingestion_agent = agent
        return agent
    
    def create_rl_agent(self,
                       state_dim: int,
                       action_dim: int,
                       **kwargs) -> RLAgent:
        """
        Create a reinforcement learning agent.
        
        Args:
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            **kwargs: Additional RL agent parameters
            
        Returns:
            RLAgent instance
        """
        agent = RLAgent(
            self.storage,
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
        self._rl_agent = agent
        return agent
    
    def create_automl_agent(self,
                           search_space: Dict[str, Callable],
                           input_dim: int,
                           output_dim: int,
                           task_type: str = "regression") -> AutoMLAgent:
        """
        Create an AutoML agent for hyperparameter optimization.
        
        Args:
            search_space: Hyperparameter search space
            input_dim: Model input dimension
            output_dim: Model output dimension
            task_type: "regression" or "classification"
            
        Returns:
            AutoMLAgent instance
        """
        agent = AutoMLAgent(
            self.storage,
            search_space=search_space,
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type
        )
        self._automl_agent = agent
        return agent
    
    # ==================== Dataset Management ====================
    
    def create_dataset(self, name: str, schema: Optional[Dict[str, Any]] = None) -> None:
        """Create a new dataset."""
        self.storage.create_dataset(name, schema=schema)
        logger.info(f"Created dataset '{name}'")
    
    def list_datasets(self) -> List[str]:
        """List all datasets."""
        return self.storage.list_datasets()
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        return self.storage.get_dataset(name)
    
    def delete_dataset(self, name: str) -> None:
        """Delete a dataset and all its tensors."""
        self.storage.delete_dataset(name)
        logger.info(f"Deleted dataset '{name}'")
    
    # ==================== Orchestrator Access ====================
    
    @property
    def orchestrator(self) -> Optional[AgentOrchestrator]:
        """Get the agent orchestrator."""
        return self._orchestrator
    
    def create_workflow(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Create a new workflow using the orchestrator.
        
        Args:
            name: Workflow name
            metadata: Optional metadata
            
        Returns:
            Workflow object
        """
        if not self._orchestrator:
            raise RuntimeError("Orchestrator not initialized. Set enable_orchestrator=True")
        return self._orchestrator.create_workflow(name, metadata)
    
    def execute_workflow(self, workflow) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow object to execute
            
        Returns:
            Dictionary of task results
        """
        if not self._orchestrator:
            raise RuntimeError("Orchestrator not initialized")
        return self._orchestrator.execute_workflow(workflow)
    
    # ==================== Utility Methods ====================
    
    def __repr__(self) -> str:
        dataset_count = len(self.list_datasets())
        index_count = len(self._vector_indexes)
        orchestrator_status = "enabled" if self._orchestrator else "disabled"
        return (f"Tensorus(datasets={dataset_count}, "
                f"vector_indexes={index_count}, "
                f"nql_enabled={self._nql_agent is not None}, "
                f"orchestrator={orchestrator_status})")
