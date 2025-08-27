# embedding_agent.py
"""
Provides embedding generation and vector similarity search capabilities for Tensorus.

This module defines the `EmbeddingAgent` class that handles automatic vectorization
of text, images, and other data types using various embedding models. It integrates
seamlessly with the existing Tensorus architecture and agent framework.

Integration Notes:
- Follows existing agent patterns (NQLAgent, RLAgent, etc.)
- Uses TensorStorage for vector persistence
- Supports multiple embedding providers (sentence-transformers, OpenAI, etc.)
- Maintains backward compatibility with existing tensor operations
"""

if __package__ in (None, ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tensorus"

import torch
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np
from datetime import datetime
import uuid

from tensorus.tensor_storage import TensorStorage
from tensorus.vector_ops import VectorOps
from tensorus.metadata.schemas import EmbeddingModelInfo, VectorMetadata

logger = logging.getLogger(__name__)

# Optional imports for embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not available. Install with: pip install openai")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available. Install with: pip install faiss-cpu or faiss-gpu")

class EmbeddingAgent:
    """
    Agent for generating embeddings and performing vector similarity search.
    
    This agent provides a unified interface for various embedding models and
    integrates with Tensorus's tensor storage and metadata systems.
    """

    def __init__(self, 
                 tensor_storage: TensorStorage,
                 default_model: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_embeddings: bool = True):
        """
        Initializes the EmbeddingAgent.

        Args:
            tensor_storage: TensorStorage instance for persisting vectors
            default_model: Default embedding model to use
            device: Device for model inference ("cpu", "cuda", or None for auto)
            cache_embeddings: Whether to cache embeddings in memory
        """
        if not isinstance(tensor_storage, TensorStorage):
            raise TypeError("tensor_storage must be an instance of TensorStorage")

        self.tensor_storage = tensor_storage
        self.default_model = default_model
        self.cache_embeddings = cache_embeddings
        
        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        
        # Initialize default model
        self._load_model(default_model)
        
        logger.info(f"EmbeddingAgent initialized with device: {self.device}")

    def _load_model(self, model_name: str) -> Any:
        """
        Loads an embedding model and caches it.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model instance
            
        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If model_name is not supported
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        try:
            if model_name.startswith("text-embedding-") or model_name.startswith("gpt-"):
                # OpenAI models
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI package not available. Install with: pip install openai")
                
                # OpenAI models don't need to be loaded, just store the name
                model = {"type": "openai", "name": model_name}
                
            else:
                # Sentence-transformers models
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
                
                model = SentenceTransformer(model_name, device=self.device)
                
            self._model_cache[model_name] = model
            logger.info(f"Loaded embedding model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model_info(self, model_name: str) -> EmbeddingModelInfo:
        """
        Gets information about an embedding model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            EmbeddingModelInfo object with model details
        """
        model = self._load_model(model_name)
        
        if isinstance(model, dict) and model.get("type") == "openai":
            # OpenAI model info
            if model_name.startswith("text-embedding-ada-002"):
                embedding_dim = 1536
            elif model_name.startswith("text-embedding-3-small"):
                embedding_dim = 1536
            elif model_name.startswith("text-embedding-3-large"):
                embedding_dim = 3072
            else:
                embedding_dim = 1536  # Default
                
            return EmbeddingModelInfo(
                model_name=model_name,
                provider="openai",
                embedding_dimension=embedding_dim,
                max_sequence_length=8191,
                normalization="l2"
            )
        else:
            # Sentence-transformers model info
            try:
                embedding_dim = model.get_sentence_embedding_dimension()
                max_seq_len = getattr(model, 'max_seq_length', None)
                
                return EmbeddingModelInfo(
                    model_name=model_name,
                    provider="sentence-transformers",
                    embedding_dimension=embedding_dim,
                    max_sequence_length=max_seq_len,
                    normalization="l2"
                )
            except Exception as e:
                logger.warning(f"Could not get model info for {model_name}: {e}")
                return EmbeddingModelInfo(
                    model_name=model_name,
                    provider="sentence-transformers",
                    embedding_dimension=384  # Common default
                )

    def encode_text(self, 
                   texts: Union[str, List[str]], 
                   model_name: Optional[str] = None,
                   normalize: bool = True) -> torch.Tensor:
        """
        Encodes text(s) into embedding vectors.
        
        Args:
            texts: Single text or list of texts to encode
            model_name: Model to use (defaults to default_model)
            normalize: Whether to normalize embeddings
            
        Returns:
            Tensor of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts is empty or invalid
            RuntimeError: If encoding fails
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        model_name = model_name or self.default_model
        
        # Check cache
        cache_key = f"{model_name}:{hash(tuple(texts))}"
        if self.cache_embeddings and cache_key in self._embedding_cache:
            embeddings = self._embedding_cache[cache_key]
            return embeddings[0:1] if single_text else embeddings
        
        try:
            model = self._load_model(model_name)
            
            if isinstance(model, dict) and model.get("type") == "openai":
                # OpenAI embeddings
                if not OPENAI_AVAILABLE:
                    raise RuntimeError("OpenAI package not available")
                
                client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
                
                embeddings_list = []
                for text in texts:
                    response = client.embeddings.create(
                        model=model_name,
                        input=text
                    )
                    embedding = response.data[0].embedding
                    embeddings_list.append(embedding)
                
                embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
                
            else:
                # Sentence-transformers embeddings
                embeddings = model.encode(texts, convert_to_tensor=True, device=self.device)
                
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Cache embeddings
            if self.cache_embeddings:
                self._embedding_cache[cache_key] = embeddings
            
            logger.debug(f"Encoded {len(texts)} texts with model {model_name}")
            
            return embeddings[0:1] if single_text else embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts with model {model_name}: {e}")
            raise RuntimeError(f"Encoding failed: {e}")

    def store_embeddings(self,
                        texts: Union[str, List[str]],
                        dataset_name: str,
                        model_name: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generates embeddings for texts and stores them in TensorStorage.
        
        Args:
            texts: Text(s) to embed and store
            dataset_name: Dataset to store embeddings in
            model_name: Embedding model to use
            metadata: Additional metadata for each embedding
            
        Returns:
            List of record IDs for stored embeddings
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If storage fails
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("texts cannot be empty")
        
        model_name = model_name or self.default_model
        
        try:
            # Generate embeddings
            embeddings = self.encode_text(texts, model_name, normalize=True)
            
            # Get model info
            model_info = self.get_model_info(model_name)
            
            # Store each embedding
            record_ids = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Prepare metadata
                embedding_metadata = {
                    "source_text": text,
                    "source_data_type": "text",
                    "embedding_model": model_info.model_dump(),
                    "is_embedding": True,
                    "created_by": "EmbeddingAgent",
                    "timestamp_utc": datetime.utcnow().timestamp()
                }
                
                # Add user metadata
                if metadata:
                    if isinstance(metadata, list) and i < len(metadata):
                        embedding_metadata.update(metadata[i])
                    elif isinstance(metadata, dict):
                        embedding_metadata.update(metadata)
                
                # Store embedding
                record_id = self.tensor_storage.insert(
                    dataset_name, 
                    embedding.cpu(), 
                    metadata=embedding_metadata
                )
                record_ids.append(record_id)
            
            logger.info(f"Stored {len(record_ids)} embeddings in dataset '{dataset_name}'")
            return record_ids
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise RuntimeError(f"Storage failed: {e}")

    def similarity_search(self,
                         query: Union[str, torch.Tensor],
                         dataset_name: str,
                         k: int = 5,
                         model_name: Optional[str] = None,
                         similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Performs similarity search against stored embeddings.
        
        Args:
            query: Query text or embedding vector
            dataset_name: Dataset to search in
            k: Number of results to return
            model_name: Model to use for query encoding (if query is text)
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results with similarity scores and metadata
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If search fails
        """
        try:
            # Encode query if it's text
            if isinstance(query, str):
                model_name = model_name or self.default_model
                query_embedding = self.encode_text(query, model_name, normalize=True)
                query_embedding = query_embedding.squeeze(0)  # Remove batch dimension
            elif isinstance(query, torch.Tensor):
                query_embedding = query
                if query_embedding.ndim != 1:
                    raise ValueError("Query tensor must be 1D")
            else:
                raise ValueError("Query must be string or torch.Tensor")
            
            # Get all embeddings from dataset
            try:
                dataset_records = self.tensor_storage.get_dataset_with_metadata(dataset_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")
            
            if not dataset_records:
                return []
            
            # Filter for embedding vectors
            embedding_records = []
            for record in dataset_records:
                metadata = record.get("metadata", {})
                if metadata.get("is_embedding", False):
                    embedding_records.append(record)
            
            if not embedding_records:
                logger.warning(f"No embeddings found in dataset '{dataset_name}'")
                return []
            
            # Extract embeddings and compute similarities
            embeddings = torch.stack([record["tensor"] for record in embedding_records])
            similarities = VectorOps.batch_cosine_similarity(query_embedding, embeddings)
            
            # Get top-k results
            top_scores, top_indices = VectorOps.top_k_similar(
                query_embedding, embeddings, k=min(k, len(embeddings)), metric="cosine"
            )
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
                score_value = float(score)
                
                # Apply similarity threshold
                if similarity_threshold is not None and score_value < similarity_threshold:
                    continue
                
                record = embedding_records[idx]
                result = {
                    "record_id": record["metadata"].get("record_id"),
                    "similarity_score": score_value,
                    "rank": i + 1,
                    "source_text": record["metadata"].get("source_text"),
                    "metadata": record["metadata"],
                    "tensor": record["tensor"]
                }
                results.append(result)
            
            logger.debug(f"Found {len(results)} similar embeddings in dataset '{dataset_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")

    def build_vector_index(self, 
                          dataset_name: str,
                          index_type: str = "flat",
                          metric: str = "cosine") -> Dict[str, Any]:
        """
        Builds a vector index for fast similarity search.
        
        Args:
            dataset_name: Dataset to index
            index_type: Type of index ("flat", "ivf", "hnsw")
            metric: Distance metric for the index
            
        Returns:
            Dictionary with index information
            
        Raises:
            ImportError: If FAISS is not available
            RuntimeError: If index building fails
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")
        
        try:
            # Get embeddings from dataset
            dataset_records = self.tensor_storage.get_dataset_with_metadata(dataset_name)
            embedding_records = [r for r in dataset_records if r["metadata"].get("is_embedding", False)]
            
            if not embedding_records:
                raise RuntimeError(f"No embeddings found in dataset '{dataset_name}'")
            
            # Extract embeddings
            embeddings = torch.stack([record["tensor"] for record in embedding_records])
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)
            
            # Build FAISS index
            dimension = embeddings_np.shape[1]
            
            if index_type == "flat":
                if metric == "cosine":
                    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings_np)
                else:
                    index = faiss.IndexFlatL2(dimension)
                    
            elif index_type == "ivf":
                nlist = min(100, len(embeddings_np) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
                # Train the index
                index.train(embeddings_np)
                
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Add vectors to index
            index.add(embeddings_np)
            
            index_info = {
                "index_type": index_type,
                "metric": metric,
                "dimension": dimension,
                "num_vectors": len(embeddings_np),
                "is_trained": True,
                "build_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Built {index_type} index for dataset '{dataset_name}' with {len(embeddings_np)} vectors")
            return index_info
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            raise RuntimeError(f"Index building failed: {e}")

    def get_embedding_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Gets statistics about embeddings in a dataset.
        
        Args:
            dataset_name: Dataset to analyze
            
        Returns:
            Dictionary with embedding statistics
        """
        try:
            dataset_records = self.tensor_storage.get_dataset_with_metadata(dataset_name)
            embedding_records = [r for r in dataset_records if r["metadata"].get("is_embedding", False)]
            
            if not embedding_records:
                return {"total_embeddings": 0}
            
            # Extract embeddings
            embeddings = torch.stack([record["tensor"] for record in embedding_records])
            
            # Compute statistics
            stats = {
                "total_embeddings": len(embedding_records),
                "embedding_dimension": embeddings.shape[1],
                "mean_norm": float(torch.mean(torch.linalg.norm(embeddings, dim=1))),
                "std_norm": float(torch.std(torch.linalg.norm(embeddings, dim=1))),
                "models_used": list(set(r["metadata"].get("embedding_model", {}).get("model_name", "unknown") 
                                      for r in embedding_records))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clears the model and embedding caches."""
        self._model_cache.clear()
        self._embedding_cache.clear()
        logger.info("Cleared embedding agent caches")
