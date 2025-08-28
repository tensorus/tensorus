"""
Tensorus Embedding Agent - Automatic text vectorization and embedding management.

This module provides comprehensive embedding generation capabilities with:
- Multiple embedding model support (OpenAI, Sentence Transformers, etc.)
- Automatic batching and optimization
- Caching and persistence
- Multi-modal support preparation
- Integration with Tensorus vector database
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
import time
import torch
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import uuid4

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from .tensor_storage import TensorStorage
from .vector_database import VectorMetadata, PartitionedVectorIndex
from .metadata.schemas import EmbeddingModelInfo

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model."""
    name: str
    provider: str
    dimension: int
    max_tokens: int = 512
    description: str = ""
    supports_batch: bool = True
    cost_per_1k_tokens: float = 0.0


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def encode(self, texts: List[str], model_name: str) -> np.ndarray:
        """Encode texts into embeddings."""
        pass
        
    @abstractmethod
    def get_model_info(self, model_name: str) -> EmbeddingModelInfo:
        """Get information about a model."""
        pass
        
    @abstractmethod
    def list_models(self) -> List[EmbeddingModelInfo]:
        """List available models."""
        pass


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            
        self.cache_dir = cache_dir
        self.models: Dict[str, SentenceTransformer] = {}
        
        # Default models with their specifications
        self.model_specs = {
            "all-MiniLM-L6-v2": EmbeddingModelInfo(
                name="all-MiniLM-L6-v2",
                provider="sentence-transformers",
                dimension=384,
                max_tokens=256,
                description="Fast and efficient model for general use",
                supports_batch=True
            ),
            "all-mpnet-base-v2": EmbeddingModelInfo(
                name="all-mpnet-base-v2", 
                provider="sentence-transformers",
                dimension=768,
                max_tokens=384,
                description="High quality embeddings for semantic search",
                supports_batch=True
            ),
            "multi-qa-MiniLM-L6-cos-v1": EmbeddingModelInfo(
                name="multi-qa-MiniLM-L6-cos-v1",
                provider="sentence-transformers",
                dimension=384,
                max_tokens=512,
                description="Optimized for question-answering tasks",
                supports_batch=True
            )
        }
        
    async def encode(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """Encode texts using sentence transformers."""
        if model_name not in self.models:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.models[model_name] = SentenceTransformer(
                model_name, 
                cache_folder=self.cache_dir
            )
            
        model = self.models[model_name]
        
        # Use asyncio to run in thread pool for non-blocking execution
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            lambda: model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        )
        
        return embeddings
        
    def get_model_info(self, model_name: str) -> EmbeddingModelInfo:
        """Get model information."""
        if model_name in self.model_specs:
            return self.model_specs[model_name]
        
        # Default info for unknown models
        return EmbeddingModelInfo(
            name=model_name,
            provider="sentence-transformers",
            dimension=768,  # Common default
            description=f"Custom sentence transformer model: {model_name}"
        )
        
    def list_models(self) -> List[EmbeddingModelInfo]:
        """List available models."""
        return list(self.model_specs.values())


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available. Install with: pip install openai")
            
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        
        self.model_specs = {
            "text-embedding-ada-002": EmbeddingModelInfo(
                name="text-embedding-ada-002",
                provider="openai",
                dimension=1536,
                max_tokens=8192,
                description="OpenAI's general-purpose embedding model",
                cost_per_1k_tokens=0.0001
            ),
            "text-embedding-3-small": EmbeddingModelInfo(
                name="text-embedding-3-small",
                provider="openai", 
                dimension=1536,
                max_tokens=8192,
                description="OpenAI's latest small embedding model",
                cost_per_1k_tokens=0.00002
            ),
            "text-embedding-3-large": EmbeddingModelInfo(
                name="text-embedding-3-large",
                provider="openai",
                dimension=3072,
                max_tokens=8192,
                description="OpenAI's latest large embedding model",
                cost_per_1k_tokens=0.00013
            )
        }
        
    async def encode(self, texts: List[str], model_name: str = "text-embedding-3-small") -> np.ndarray:
        """Encode texts using OpenAI embeddings."""
        loop = asyncio.get_event_loop()
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.embeddings.create(
                input=texts,
                model=model_name
            )
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
        
    def get_model_info(self, model_name: str) -> EmbeddingModelInfo:
        """Get model information."""
        return self.model_specs.get(model_name, EmbeddingModelInfo(
            name=model_name,
            provider="openai", 
            dimension=1536,
            description=f"OpenAI model: {model_name}"
        ))
        
    def list_models(self) -> List[EmbeddingModelInfo]:
        """List available models."""
        return list(self.model_specs.values())


class EmbeddingAgent:
    """
    Advanced embedding agent with multi-provider support and vector database integration.
    
    Features:
    - Multiple embedding providers (Sentence Transformers, OpenAI)
    - Automatic batching and optimization
    - Caching and deduplication
    - Integration with Tensorus tensor storage and vector database
    - Performance monitoring and metrics
    """
    
    def __init__(self, 
                 tensor_storage: TensorStorage,
                 default_provider: str = "sentence-transformers",
                 default_model: str = "all-MiniLM-L6-v2",
                 cache_dir: Optional[str] = None,
                 batch_size: int = 32):
        
        self.tensor_storage = tensor_storage
        self.default_provider = default_provider
        self.default_model = default_model
        self.batch_size = batch_size
        
        # Initialize providers
        self.providers: Dict[str, EmbeddingProvider] = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.providers["sentence-transformers"] = SentenceTransformersProvider(cache_dir)
            
        if OPENAI_AVAILABLE:
            try:
                self.providers["openai"] = OpenAIEmbeddingProvider()
            except Exception as e:
                logger.warning(f"OpenAI provider initialization failed: {e}")
                
        if not self.providers:
            logger.warning("No embedding providers available")
            
        # Vector indexes by dataset
        self.vector_indexes: Dict[str, PartitionedVectorIndex] = {}
        
        # Performance metrics
        self.metrics = {
            "total_embeddings_generated": 0,
            "total_tokens_processed": 0,
            "average_embedding_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        # LRU Embedding cache with size limits
        self.cache_max_size = 10000  # Maximum cache entries
        self.cache_max_memory_mb = 500  # Maximum cache memory in MB
        self.embedding_cache: OrderedDict[str, Tuple[np.ndarray, datetime, float]] = OrderedDict()
        self.cache_ttl = 3600  # 1 hour
        self.cache_memory_usage = 0.0  # Current memory usage in MB
        
    def _get_provider(self, provider_name: Optional[str] = None) -> EmbeddingProvider:
        """Get embedding provider by name."""
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Provider '{provider_name}' not available. Available: {available}")
            
        return self.providers[provider_name]
        
    def _get_cache_key(self, text: str, model_name: str, provider: str) -> str:
        """Generate cache key for embedding."""
        content = f"{provider}:{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid."""
        return (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl
        
    def _calculate_embedding_size_mb(self, embedding: np.ndarray) -> float:
        """Calculate memory size of embedding in MB."""
        size_bytes = embedding.nbytes
        return size_bytes / (1024 * 1024)
        
    def _evict_lru_entries(self, required_space_mb: float = 0) -> None:
        """Evict least recently used entries to free memory."""
        # Remove expired entries first
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, (embedding, timestamp, size_mb) in self.embedding_cache.items():
            if not self._is_cache_valid(timestamp):
                expired_keys.append(key)
                
        for key in expired_keys:
            embedding, timestamp, size_mb = self.embedding_cache.pop(key)
            self.cache_memory_usage -= size_mb
            
        # Evict LRU entries if still over limits
        while (len(self.embedding_cache) >= self.cache_max_size or 
               self.cache_memory_usage + required_space_mb > self.cache_max_memory_mb):
            if not self.embedding_cache:
                break
                
            # Remove oldest entry (FIFO in OrderedDict)
            key, (embedding, timestamp, size_mb) = self.embedding_cache.popitem(last=False)
            self.cache_memory_usage -= size_mb
            
    def _add_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Add embedding to cache with LRU management."""
        size_mb = self._calculate_embedding_size_mb(embedding)
        
        # Evict entries if needed
        self._evict_lru_entries(size_mb)
        
        # Add new entry
        self.embedding_cache[cache_key] = (embedding, datetime.utcnow(), size_mb)
        self.cache_memory_usage += size_mb
        
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Get embedding from cache and update LRU order."""
        if cache_key not in self.embedding_cache:
            return None
            
        embedding, timestamp, size_mb = self.embedding_cache[cache_key]
        
        if not self._is_cache_valid(timestamp):
            # Remove expired entry
            del self.embedding_cache[cache_key]
            self.cache_memory_usage -= size_mb
            return None
            
        # Move to end (most recently used)
        self.embedding_cache.move_to_end(cache_key)
        return embedding
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.embedding_cache),
            "cache_max_size": self.cache_max_size,
            "cache_memory_usage_mb": self.cache_memory_usage,
            "cache_max_memory_mb": self.cache_max_memory_mb,
            "cache_utilization_pct": (len(self.embedding_cache) / self.cache_max_size) * 100,
            "memory_utilization_pct": (self.cache_memory_usage / self.cache_max_memory_mb) * 100
        }
        
    async def generate_embeddings(self, 
                                texts: Union[str, List[str]], 
                                model_name: Optional[str] = None,
                                provider: Optional[str] = None) -> Tuple[np.ndarray, EmbeddingModelInfo]:
        """Generate embeddings for texts with caching and batching."""
        if isinstance(texts, str):
            texts = [texts]
            
        model_name = model_name or self.default_model
        provider = provider or self.default_provider
        embedding_provider = self._get_provider(provider)
        
        start_time = time.time()
        
        # Check cache
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, model_name, provider)
            embedding = self._get_from_cache(cache_key)
            if embedding is not None:
                cached_embeddings.append((i, embedding))
                continue
                    
            uncached_texts.append(text)
            uncached_indices.append(i)
            
        # Generate embeddings for uncached texts
        new_embeddings = None
        if uncached_texts:
            # Process in batches
            all_new_embeddings = []
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i:i + self.batch_size]
                batch_embeddings = await embedding_provider.encode(batch, model_name)
                all_new_embeddings.extend(batch_embeddings)
                
            new_embeddings = np.array(all_new_embeddings)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, model_name, provider)
                self._add_to_cache(cache_key, embedding)
                
        # Combine cached and new embeddings
        final_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for original_idx, embedding in cached_embeddings:
            final_embeddings[original_idx] = embedding
            
        # Place new embeddings
        if new_embeddings is not None:
            for i, original_idx in enumerate(uncached_indices):
                final_embeddings[original_idx] = new_embeddings[i]
                
        result_embeddings = np.array(final_embeddings)
        
        # Update metrics
        embedding_time = time.time() - start_time
        self.metrics["total_embeddings_generated"] += len(texts)
        self.metrics["total_tokens_processed"] += sum(len(text.split()) for text in texts)
        self.metrics["average_embedding_time"] = (
            (self.metrics["average_embedding_time"] * (self.metrics["total_embeddings_generated"] - len(texts)) + 
             embedding_time) / self.metrics["total_embeddings_generated"]
        )
        
        cache_hits = len(cached_embeddings)
        total_requests = len(texts)
        self.metrics["cache_hit_rate"] = cache_hits / total_requests if total_requests > 0 else 0
        
        model_info = embedding_provider.get_model_info(model_name)
        return result_embeddings, model_info
        
    async def store_embeddings(self,
                              texts: Union[str, List[str]],
                              dataset_name: str,
                              model_name: Optional[str] = None,
                              provider: Optional[str] = None,
                              namespace: str = "default",
                              tenant_id: str = "default",
                              metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate and store embeddings in tensor storage and vector index."""
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate embeddings
        embeddings, model_info = await self.generate_embeddings(texts, model_name, provider)
        
        # Ensure dataset exists
        if not self.tensor_storage.dataset_exists(dataset_name):
            self.tensor_storage.create_dataset(dataset_name)
        
        # Create tensor storage records
        record_ids = []
        vector_entries = {}
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Store in tensor storage - let it generate the record_id
            tensor = torch.from_numpy(embedding).float()
            
            storage_metadata = {
                "source_text": text,
                "model_name": model_info.name,
                "provider": model_info.provider,
                "embedding_dimension": model_info.dimension,
                "created_at": datetime.utcnow().isoformat(),
                "namespace": namespace,
                "tenant_id": tenant_id
            }
            
            if metadata:
                storage_metadata.update(metadata)
                
            record_id = self.tensor_storage.insert(
                name=dataset_name,
                tensor=tensor,
                metadata=storage_metadata
            )
            record_ids.append(record_id)
            
            # Prepare for vector index
            vector_metadata = VectorMetadata(
                vector_id=record_id,
                namespace=namespace,
                tenant_id=tenant_id,
                content=text,
                metadata=storage_metadata
            )
            
            vector_entries[record_id] = (embedding, vector_metadata)
            
        # Initialize vector index if needed
        if dataset_name not in self.vector_indexes:
            self.vector_indexes[dataset_name] = PartitionedVectorIndex(
                dimension=model_info.dimension,
                num_partitions=8,
                metric="cosine"
            )
            
        # Add to vector index
        await self.vector_indexes[dataset_name].add_vectors(vector_entries)
        
        logger.info(f"Stored {len(embeddings)} embeddings in dataset '{dataset_name}'")
        return record_ids
        
    async def similarity_search(self,
                               query: str,
                               dataset_name: str,
                               k: int = 5,
                               model_name: Optional[str] = None,
                               provider: Optional[str] = None,
                               namespace: Optional[str] = None,
                               tenant_id: Optional[str] = None,
                               similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Perform similarity search against stored embeddings."""
        if dataset_name not in self.vector_indexes:
            raise ValueError(f"No vector index found for dataset '{dataset_name}'")
            
        # Generate query embedding
        query_embeddings, _ = await self.generate_embeddings([query], model_name, provider)
        query_vector = query_embeddings[0]
        
        # Prepare filters
        filters = {}
        if namespace is not None:
            filters["namespace"] = namespace
        if tenant_id is not None:
            filters["tenant_id"] = tenant_id
            
        # Search vector index
        search_results = await self.vector_indexes[dataset_name].search(
            query_vector=query_vector,
            k=k,
            filters=filters if filters else None
        )
        
        # Filter by similarity threshold
        if similarity_threshold is not None:
            search_results = [r for r in search_results if r.score >= similarity_threshold]
            
        # Format results
        results = []
        for result in search_results:
            result_dict = {
                "record_id": result.vector_id,
                "similarity_score": result.score,
                "rank": result.rank,
                "source_text": result.metadata.content,
                "metadata": result.metadata.metadata,
                "namespace": result.metadata.namespace,
                "tenant_id": result.metadata.tenant_id
            }
            
            # Get tensor from storage
            try:
                tensor_record = self.tensor_storage.get_tensor_by_id(dataset_name, result.vector_id)
                result_dict["tensor"] = tensor_record["tensor"]
            except Exception as e:
                logger.warning(f"Could not retrieve tensor for {result.vector_id}: {e}")
                
            results.append(result_dict)
            
        return results
        
    def get_embedding_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics about embeddings in a dataset."""
        try:
            dataset_info = self.tensor_storage.get_dataset_info(dataset_name)
            
            stats = {
                "dataset_name": dataset_name,
                "total_embeddings": dataset_info.get("record_count", 0),
                "created_at": dataset_info.get("created_at"),
                "last_updated": dataset_info.get("last_updated")
            }
            
            # Add vector index stats if available
            if dataset_name in self.vector_indexes:
                index_stats = self.vector_indexes[dataset_name].get_combined_stats()
                stats.update({
                    "vector_index_size_mb": index_stats.index_size_mb,
                    "partitions": index_stats.partitions,
                    "index_last_updated": index_stats.last_updated.isoformat()
                })
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get embedding stats for {dataset_name}: {e}")
            return {"error": str(e)}
            
    def get_model_info(self, model_name: str, provider: Optional[str] = None) -> EmbeddingModelInfo:
        """Get information about an embedding model."""
        embedding_provider = self._get_provider(provider)
        return embedding_provider.get_model_info(model_name)
        
    def list_available_models(self) -> Dict[str, List[EmbeddingModelInfo]]:
        """List all available models across providers."""
        models_by_provider = {}
        
        for provider_name, provider in self.providers.items():
            models_by_provider[provider_name] = provider.list_models()
            
        return models_by_provider
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics including cache statistics."""
        metrics = self.metrics.copy()
        metrics.update(self.get_cache_stats())
        return metrics
        
    async def build_vector_index(self,
                                dataset_name: str,
                                index_type: str = "partitioned",
                                metric: str = "cosine",
                                num_partitions: int = 8) -> Dict[str, Any]:
        """Build or rebuild vector index for a dataset."""
        if dataset_name not in self.tensor_storage.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
            
        # Get all tensors from dataset
        dataset = self.tensor_storage.datasets[dataset_name]
        
        if not dataset:
            return {"error": "Dataset is empty"}
            
        # Determine dimension from first tensor
        first_record = next(iter(dataset.values()))
        dimension = first_record["tensor"].shape[0]
        
        # Create new vector index
        vector_index = PartitionedVectorIndex(
            dimension=dimension,
            num_partitions=num_partitions,
            metric=metric
        )
        
        # Prepare vectors for indexing
        vector_entries = {}
        
        for record_id, record in dataset.items():
            tensor = record["tensor"]
            metadata_dict = record.get("metadata", {})
            
            # Create vector metadata
            vector_metadata = VectorMetadata(
                vector_id=record_id,
                namespace=metadata_dict.get("namespace", "default"),
                tenant_id=metadata_dict.get("tenant_id", "default"),
                content=metadata_dict.get("source_text", ""),
                metadata=metadata_dict,
                created_at=datetime.fromisoformat(metadata_dict.get("created_at", datetime.utcnow().isoformat()))
            )
            
            vector_entries[record_id] = (tensor.numpy(), vector_metadata)
            
        # Build index
        start_time = time.time()
        await vector_index.add_vectors(vector_entries)
        build_time = time.time() - start_time
        
        # Replace existing index
        self.vector_indexes[dataset_name] = vector_index
        
        # Get final stats
        stats = vector_index.get_combined_stats()
        
        return {
            "index_type": index_type,
            "metric": metric,
            "build_time_seconds": build_time,
            "total_vectors": stats.total_vectors,
            "partitions": stats.partitions,
            "index_size_mb": stats.index_size_mb,
            "created_at": datetime.utcnow().isoformat()
        }