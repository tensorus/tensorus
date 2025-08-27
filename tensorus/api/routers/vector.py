# vector.py
"""
FastAPI router for vector database operations including embedding generation
and similarity search endpoints.

This module provides REST API endpoints for:
- Text embedding generation
- Vector similarity search
- Hybrid search (semantic + metadata)
- Vector index management
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import logging

from ...tensor_storage import TensorStorage
from ...embedding_agent import EmbeddingAgent
from ...vector_ops import VectorOps
from ...vector_index import VectorIndexManager
from ...metadata.schemas import SimilaritySearchResult, EmbeddingModelInfo
from ..dependencies import get_tensor_storage
from ..security import verify_api_key
from ...utils import tensor_to_list

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API requests and responses

class EmbedTextRequest(BaseModel):
    """Request model for text embedding generation."""
    texts: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    dataset_name: str = Field(..., description="Dataset to store embeddings in")
    model_name: Optional[str] = Field(None, description="Embedding model to use")
    normalize: bool = Field(True, description="Whether to normalize embeddings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for embeddings")

class EmbedTextResponse(BaseModel):
    """Response model for text embedding generation."""
    success: bool
    message: str
    record_ids: List[str] = Field(description="IDs of stored embedding records")
    model_info: EmbeddingModelInfo
    embeddings_count: int

class SimilaritySearchRequest(BaseModel):
    """Request model for vector similarity search."""
    query: str = Field(..., description="Query text for similarity search")
    dataset_name: str = Field(..., description="Dataset to search in")
    k: int = Field(5, ge=1, le=100, description="Number of results to return")
    model_name: Optional[str] = Field(None, description="Model to use for query encoding")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_embeddings: bool = Field(False, description="Whether to include embedding vectors in response")

class SimilaritySearchResponse(BaseModel):
    """Response model for vector similarity search."""
    success: bool
    message: str
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: Optional[float] = None

class VectorIndexRequest(BaseModel):
    """Request model for building vector indexes."""
    dataset_name: str = Field(..., description="Dataset to index")
    index_type: str = Field("flat", description="Type of index to build")
    metric: str = Field("cosine", description="Distance metric for the index")

class VectorIndexResponse(BaseModel):
    """Response model for vector index operations."""
    success: bool
    message: str
    index_info: Dict[str, Any]

class HybridSearchRequest(BaseModel):
    """Request model for hybrid search combining vector similarity and metadata filtering."""
    query: str = Field(..., description="Search query")
    dataset_name: str = Field(..., description="Dataset to search in")
    k: int = Field(5, ge=1, le=100, description="Number of results to return")
    vector_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for vector similarity")
    metadata_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for metadata matching")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply")
    model_name: Optional[str] = Field(None, description="Embedding model to use")

class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    success: bool
    message: str
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    vector_weight: float
    metadata_weight: float

class EmbeddingStatsResponse(BaseModel):
    """Response model for embedding statistics."""
    success: bool
    dataset_name: str
    stats: Dict[str, Any]

class BuildIndexRequest(BaseModel):
    """Request model for building vector indexes."""
    dataset_name: str = Field(..., description="Name of dataset containing embeddings")
    index_name: Optional[str] = Field(None, description="Name for the index (defaults to dataset_name)")
    index_type: str = Field("flat", description="Type of index (flat, ivf, hnsw)")
    metric: str = Field("cosine", description="Distance metric (cosine, euclidean)")
    index_params: Optional[Dict[str, Any]] = Field(None, description="Additional index parameters")
    embedding_filter: Optional[Dict[str, Any]] = Field(None, description="Filter for selecting embeddings")

class BuildIndexResponse(BaseModel):
    """Response model for index building."""
    success: bool
    index_name: str
    stats: Dict[str, Any]
    message: str

class IndexSearchRequest(BaseModel):
    """Request model for index-based search."""
    index_name: str = Field(..., description="Name of index to search")
    query: str = Field(..., description="Query text")
    k: int = Field(5, ge=1, le=100, description="Number of results to return")
    embedding_model: Optional[str] = Field(None, description="Model for text encoding")

class IndexSearchResponse(BaseModel):
    """Response model for index search."""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    search_time_ms: float

class ListIndexesResponse(BaseModel):
    """Response model for listing indexes."""
    success: bool
    indexes: List[Dict[str, Any]]

# Global embedding agent and index manager instances
_embedding_agent: Optional[EmbeddingAgent] = None
_index_manager: Optional[VectorIndexManager] = None

def get_embedding_agent(storage: TensorStorage = Depends(get_tensor_storage)) -> EmbeddingAgent:
    """Dependency to get or create EmbeddingAgent instance."""
    global _embedding_agent
    if _embedding_agent is None:
        _embedding_agent = EmbeddingAgent(storage)
    return _embedding_agent

def get_index_manager(storage: TensorStorage = Depends(get_tensor_storage)) -> VectorIndexManager:
    """Dependency to get or create VectorIndexManager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = VectorIndexManager(storage)
    return _index_manager

# API Endpoints

@router.post("/embed", response_model=EmbedTextResponse, tags=["Vector Operations"])
async def embed_text(
    request: EmbedTextRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Generate embeddings for text(s) and store them in the specified dataset.
    
    This endpoint accepts single text or a list of texts, generates embeddings using
    the specified model, and stores them in TensorStorage with appropriate metadata.
    """
    try:
        # Validate weights sum for hybrid search compatibility
        if isinstance(request.texts, list) and len(request.texts) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="texts cannot be empty"
            )
        
        # Generate and store embeddings
        record_ids = embedding_agent.store_embeddings(
            texts=request.texts,
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            metadata=request.metadata
        )
        
        # Get model info
        model_name = request.model_name or embedding_agent.default_model
        model_info = embedding_agent.get_model_info(model_name)
        
        # Count embeddings
        embeddings_count = len(record_ids)
        
        return EmbedTextResponse(
            success=True,
            message=f"Successfully generated and stored {embeddings_count} embeddings",
            record_ids=record_ids,
            model_info=model_info,
            embeddings_count=embeddings_count
        )
        
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )

@router.post("/search", response_model=SimilaritySearchResponse, tags=["Vector Operations"])
async def similarity_search(
    request: SimilaritySearchRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Perform vector similarity search against stored embeddings.
    
    This endpoint encodes the query text and searches for the most similar
    embeddings in the specified dataset using cosine similarity.
    """
    import time
    start_time = time.time()
    
    try:
        # Perform similarity search
        results = embedding_agent.similarity_search(
            query=request.query,
            dataset_name=request.dataset_name,
            k=request.k,
            model_name=request.model_name,
            similarity_threshold=request.similarity_threshold
        )
        
        # Process results
        processed_results = []
        for result in results:
            processed_result = {
                "record_id": result["record_id"],
                "similarity_score": result["similarity_score"],
                "rank": result["rank"],
                "source_text": result["source_text"],
                "metadata": result["metadata"]
            }
            
            # Include embeddings if requested
            if request.include_embeddings:
                tensor = result["tensor"]
                shape, dtype, data_list = tensor_to_list(tensor)
                processed_result["embedding"] = {
                    "shape": shape,
                    "dtype": dtype,
                    "data": data_list
                }
            
            processed_results.append(processed_result)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SimilaritySearchResponse(
            success=True,
            message=f"Found {len(results)} similar embeddings",
            query=request.query,
            results=processed_results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {str(e)}"
        )

@router.post("/index", response_model=VectorIndexResponse, tags=["Vector Operations"])
async def build_vector_index(
    request: VectorIndexRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Build a vector index for fast similarity search on a dataset.
    
    This endpoint creates a FAISS index for the embeddings in the specified
    dataset to enable faster similarity search operations.
    """
    try:
        # Build vector index
        index_info = embedding_agent.build_vector_index(
            dataset_name=request.dataset_name,
            index_type=request.index_type,
            metric=request.metric
        )
        
        return VectorIndexResponse(
            success=True,
            message=f"Successfully built {request.index_type} index",
            index_info=index_info
        )
        
    except Exception as e:
        logger.error(f"Vector index building failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index building failed: {str(e)}"
        )

@router.post("/hybrid-search", response_model=HybridSearchResponse, tags=["Vector Operations"])
async def hybrid_search(
    request: HybridSearchRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    storage: TensorStorage = Depends(get_tensor_storage),
    api_key: str = Depends(verify_api_key)
):
    """
    Perform hybrid search combining vector similarity and metadata filtering.
    
    This endpoint combines semantic vector search with traditional metadata
    filtering to provide more comprehensive search results.
    """
    try:
        # Validate weights sum to 1.0
        if abs(request.vector_weight + request.metadata_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="vector_weight and metadata_weight must sum to 1.0"
            )
        
        # Perform vector similarity search
        vector_results = embedding_agent.similarity_search(
            query=request.query,
            dataset_name=request.dataset_name,
            k=request.k * 2,  # Get more candidates for hybrid ranking
            model_name=request.model_name
        )
        
        # Apply metadata filtering if specified
        if request.metadata_filters:
            filtered_results = []
            for result in vector_results:
                metadata = result["metadata"]
                matches_filter = True
                
                for key, expected_value in request.metadata_filters.items():
                    if key not in metadata or metadata[key] != expected_value:
                        matches_filter = False
                        break
                
                if matches_filter:
                    filtered_results.append(result)
            
            vector_results = filtered_results
        
        # Combine and rank results (simplified hybrid scoring)
        hybrid_results = []
        for result in vector_results[:request.k]:
            # Simple hybrid score: weighted combination
            vector_score = result["similarity_score"] * request.vector_weight
            metadata_score = 1.0 * request.metadata_weight  # Simplified: 1.0 if passes filters
            
            hybrid_score = vector_score + metadata_score
            
            hybrid_result = {
                "record_id": result["record_id"],
                "hybrid_score": hybrid_score,
                "vector_similarity": result["similarity_score"],
                "rank": result["rank"],
                "source_text": result["source_text"],
                "metadata": result["metadata"]
            }
            hybrid_results.append(hybrid_result)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(hybrid_results):
            result["rank"] = i + 1
        
        return HybridSearchResponse(
            success=True,
            message=f"Found {len(hybrid_results)} hybrid search results",
            query=request.query,
            results=hybrid_results,
            total_results=len(hybrid_results),
            vector_weight=request.vector_weight,
            metadata_weight=request.metadata_weight
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )

@router.get("/stats/{dataset_name}", response_model=EmbeddingStatsResponse, tags=["Vector Operations"])
async def get_embedding_stats(
    dataset_name: str = Path(..., description="Name of the dataset"),
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Get statistics about embeddings in a dataset.
    
    This endpoint provides information about the embeddings stored in a dataset,
    including count, dimensions, and model information.
    """
    try:
        stats = embedding_agent.get_embedding_stats(dataset_name)
        
        return EmbeddingStatsResponse(
            success=True,
            dataset_name=dataset_name,
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"Getting embedding stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedding stats: {str(e)}"
        )

@router.get("/models", tags=["Vector Operations"])
async def list_available_models(api_key: str = Depends(verify_api_key)):
    """
    List available embedding models.
    
    Returns information about supported embedding models and their capabilities.
    """
    try:
        # Common sentence-transformers models
        sentence_transformer_models = [
            {
                "name": "all-MiniLM-L6-v2",
                "provider": "sentence-transformers",
                "dimension": 384,
                "description": "Fast and efficient model for general use"
            },
            {
                "name": "all-mpnet-base-v2",
                "provider": "sentence-transformers", 
                "dimension": 768,
                "description": "High quality embeddings for semantic search"
            },
            {
                "name": "multi-qa-MiniLM-L6-cos-v1",
                "provider": "sentence-transformers",
                "dimension": 384,
                "description": "Optimized for question-answering tasks"
            }
        ]
        
        # OpenAI models
        openai_models = [
            {
                "name": "text-embedding-ada-002",
                "provider": "openai",
                "dimension": 1536,
                "description": "OpenAI's general-purpose embedding model"
            },
            {
                "name": "text-embedding-3-small",
                "provider": "openai",
                "dimension": 1536,
                "description": "OpenAI's latest small embedding model"
            },
            {
                "name": "text-embedding-3-large",
                "provider": "openai",
                "dimension": 3072,
                "description": "OpenAI's latest large embedding model"
            }
        ]
        
        # Combine all models
        all_models = sentence_transformer_models + openai_models
        return all_models
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )

# Vector Index Management Endpoints

@router.post("/index/build", response_model=BuildIndexResponse, tags=["Vector Indexing"])
async def build_vector_index(
    request: BuildIndexRequest,
    index_manager: VectorIndexManager = Depends(get_index_manager),
    api_key: str = Depends(verify_api_key)
):
    """Build a vector index from embeddings in a dataset."""
    try:
        import time
        start_time = time.time()
        
        index = index_manager.build_index(
            dataset_name=request.dataset_name,
            index_name=request.index_name,
            index_type=request.index_type,
            metric=request.metric,
            index_params=request.index_params,
            embedding_filter=request.embedding_filter
        )
        
        build_time = (time.time() - start_time) * 1000
        stats = index.get_stats()
        stats["build_time_ms"] = build_time
        
        return BuildIndexResponse(
            success=True,
            index_name=request.index_name or f"{request.dataset_name}_{request.index_type}",
            stats=stats,
            message=f"Successfully built {request.index_type} index with {stats['total_vectors']} vectors"
        )
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build index: {str(e)}"
        )

@router.post("/index/search", response_model=IndexSearchResponse, tags=["Vector Indexing"])
async def search_vector_index(
    request: IndexSearchRequest,
    index_manager: VectorIndexManager = Depends(get_index_manager),
    api_key: str = Depends(verify_api_key)
):
    """Search a vector index with a text query."""
    try:
        import time
        start_time = time.time()
        
        results = index_manager.search_index(
            index_name=request.index_name,
            query=request.query,
            k=request.k,
            embedding_model=request.embedding_model
        )
        
        search_time = (time.time() - start_time) * 1000
        
        return IndexSearchResponse(
            success=True,
            query=request.query,
            results=results,
            search_time_ms=search_time
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index '{request.index_name}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to search index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search index: {str(e)}"
        )

@router.get("/index/list", response_model=ListIndexesResponse, tags=["Vector Indexing"])
async def list_vector_indexes(
    index_manager: VectorIndexManager = Depends(get_index_manager),
    api_key: str = Depends(verify_api_key)
):
    """List all available vector indexes."""
    try:
        indexes = index_manager.list_indexes()
        return ListIndexesResponse(
            success=True,
            indexes=indexes
        )
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indexes: {str(e)}"
        )

@router.delete("/index/{index_name}", tags=["Vector Indexing"])
async def delete_vector_index(
    index_name: str = Path(..., description="Name of index to delete"),
    index_manager: VectorIndexManager = Depends(get_index_manager),
    api_key: str = Depends(verify_api_key)
):
    """Delete a vector index."""
    try:
        success = index_manager.delete_index(index_name)
        if success:
            return {"success": True, "message": f"Index '{index_name}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete index '{index_name}'"
            )
    except Exception as e:
        logger.error(f"Failed to delete index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete index: {str(e)}"
        )
