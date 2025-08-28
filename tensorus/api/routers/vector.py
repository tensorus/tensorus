"""
Tensorus Vector Database API Router

Comprehensive API endpoints for advanced vector database capabilities including:
- Multi-provider embedding generation and storage
- Geometric partitioning with freshness layers
- Hybrid search combining semantic similarity with tensor operations
- Multi-tenant namespace isolation
- Real-time performance monitoring
- Scientific computational lineage tracking

This router implements Tensorus's unique "Computational Vector Database" concept.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import logging
import asyncio

from ...tensor_storage import TensorStorage
from ...embedding_agent import EmbeddingAgent
from ...tensor_ops import TensorOps
from ...hybrid_search import HybridSearchEngine, HybridQuery, TensorOperation
from ...vector_database import VectorMetadata
from ..dependencies import get_tensor_storage
from ..security import verify_api_key
from ...utils import tensor_to_list

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances - will be initialized via dependencies
_embedding_agent: Optional[EmbeddingAgent] = None
_tensor_ops: Optional[TensorOps] = None
_hybrid_search: Optional[HybridSearchEngine] = None


def get_embedding_agent(storage: TensorStorage = Depends(get_tensor_storage)) -> EmbeddingAgent:
    """Dependency to get or create EmbeddingAgent instance."""
    global _embedding_agent
    if _embedding_agent is None:
        _embedding_agent = EmbeddingAgent(
            tensor_storage=storage,
            default_provider="sentence-transformers",
            default_model="all-MiniLM-L6-v2"
        )
    return _embedding_agent


def get_tensor_ops() -> TensorOps:
    """Dependency to get TensorOps instance."""
    global _tensor_ops
    if _tensor_ops is None:
        _tensor_ops = TensorOps()
    return _tensor_ops


def get_hybrid_search(storage: TensorStorage = Depends(get_tensor_storage),
                     embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
                     tensor_ops: TensorOps = Depends(get_tensor_ops)) -> HybridSearchEngine:
    """Dependency to get or create HybridSearchEngine instance."""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearchEngine(
            tensor_storage=storage,
            embedding_agent=embedding_agent,
            tensor_ops=tensor_ops
        )
    return _hybrid_search


# Pydantic Models for API

class EmbedTextRequest(BaseModel):
    """Request model for text embedding generation."""
    texts: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    dataset_name: str = Field(..., description="Dataset to store embeddings in")
    model_name: Optional[str] = Field(None, description="Embedding model to use")
    provider: Optional[str] = Field(None, description="Embedding provider (sentence-transformers, openai)")
    namespace: str = Field("default", description="Namespace for multi-tenancy")
    tenant_id: str = Field("default", description="Tenant ID for isolation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EmbedTextResponse(BaseModel):
    """Response model for text embedding generation."""
    success: bool
    message: str
    record_ids: List[str]
    embeddings_count: int
    model_info: Dict[str, Any]
    namespace: str
    tenant_id: str


class SimilaritySearchRequest(BaseModel):
    """Request model for vector similarity search."""
    query: str = Field(..., description="Query text for similarity search")
    dataset_name: str = Field(..., description="Dataset to search in")
    k: int = Field(5, ge=1, le=100, description="Number of results to return")
    model_name: Optional[str] = Field(None, description="Model for query encoding")
    provider: Optional[str] = Field(None, description="Embedding provider")
    namespace: Optional[str] = Field(None, description="Namespace filter")
    tenant_id: Optional[str] = Field(None, description="Tenant ID filter")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_vectors: bool = Field(False, description="Include embedding vectors in response")


class SimilaritySearchResponse(BaseModel):
    """Response model for vector similarity search."""
    success: bool
    message: str
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: Optional[float] = None


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search combining semantic and computational relevance."""
    text_query: Optional[str] = Field(None, description="Semantic search query")
    dataset_name: str = Field(..., description="Dataset to search in")
    tensor_operations: List[Dict[str, Any]] = Field(default=[], description="Tensor operations for computational scoring")
    similarity_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for semantic similarity")
    computation_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for computational relevance")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")
    namespace: str = Field("default", description="Namespace for search")
    tenant_id: str = Field("default", description="Tenant ID for search")
    filters: Dict[str, Any] = Field(default={}, description="Additional filters")


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    success: bool
    message: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float
    semantic_weight: float
    computation_weight: float


class TensorWorkflowRequest(BaseModel):
    """Request model for tensor workflow execution."""
    workflow_query: str = Field(..., description="Semantic description of workflow")
    dataset_name: str = Field(..., description="Dataset to search for input tensors")
    operations: List[Dict[str, Any]] = Field(..., description="Tensor operations to execute")
    save_intermediates: bool = Field(True, description="Save intermediate results")
    namespace: str = Field("default", description="Namespace")
    tenant_id: str = Field("default", description="Tenant ID")


class TensorWorkflowResponse(BaseModel):
    """Response model for tensor workflow execution."""
    success: bool
    workflow_id: str
    operations_executed: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    intermediate_results: List[Dict[str, Any]]
    computational_lineage: List[str]


class VectorIndexRequest(BaseModel):
    """Request model for building vector indexes."""
    dataset_name: str = Field(..., description="Dataset to index")
    index_type: str = Field("partitioned", description="Type of index to build")
    metric: str = Field("cosine", description="Distance metric")
    num_partitions: int = Field(8, ge=1, le=32, description="Number of partitions")


class VectorIndexResponse(BaseModel):
    """Response model for vector index operations."""
    success: bool
    message: str
    index_stats: Dict[str, Any]


class ModelListResponse(BaseModel):
    """Response model for listing available models."""
    success: bool
    providers: Dict[str, List[Dict[str, Any]]]
    total_models: int


class MetricsResponse(BaseModel):
    """Response model for performance metrics."""
    success: bool
    embedding_metrics: Dict[str, Any]
    search_metrics: Dict[str, Any]
    lineage_metrics: Dict[str, Any]


# API Endpoints

@router.post("/embed", response_model=EmbedTextResponse, tags=["Vector Database"])
async def embed_text(
    request: EmbedTextRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Generate embeddings for text(s) and store them with full vector database capabilities.
    
    Features:
    - Multi-provider embedding generation
    - Automatic vector indexing with geometric partitioning
    - Multi-tenant namespace isolation
    - Computational lineage tracking
    """
    try:
        texts_list = request.texts if isinstance(request.texts, list) else [request.texts]
        
        if not texts_list or (len(texts_list) == 1 and not texts_list[0].strip()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="texts cannot be empty"
            )
        
        # Generate and store embeddings
        record_ids = await embedding_agent.store_embeddings(
            texts=texts_list,
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            provider=request.provider,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            metadata=request.metadata
        )
        
        # Get model info
        model_name = request.model_name or embedding_agent.default_model
        model_info = embedding_agent.get_model_info(model_name, request.provider)
        
        return EmbedTextResponse(
            success=True,
            message=f"Successfully generated and stored {len(record_ids)} embeddings",
            record_ids=record_ids,
            embeddings_count=len(record_ids),
            model_info={
                "name": model_info.name,
                "provider": model_info.provider,
                "dimension": model_info.dimension,
                "description": model_info.description
            },
            namespace=request.namespace,
            tenant_id=request.tenant_id
        )
        
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.post("/search", response_model=SimilaritySearchResponse, tags=["Vector Database"])
async def similarity_search(
    request: SimilaritySearchRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Perform advanced vector similarity search with multi-tenant isolation.
    
    Features:
    - Sub-100ms latency with geometric partitioning
    - Freshness layer for real-time updates
    - Multi-tenant namespace filtering
    - Optional similarity thresholding
    """
    import time
    start_time = time.time()
    
    try:
        # Perform similarity search
        results = await embedding_agent.similarity_search(
            query=request.query,
            dataset_name=request.dataset_name,
            k=request.k,
            model_name=request.model_name,
            provider=request.provider,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
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
                "metadata": result["metadata"],
                "namespace": result["namespace"],
                "tenant_id": result["tenant_id"]
            }
            
            # Include vectors if requested
            if request.include_vectors and "tensor" in result:
                tensor = result["tensor"]
                shape, dtype, data_list = tensor_to_list(tensor)
                processed_result["vector"] = {
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


@router.post("/hybrid-search", response_model=HybridSearchResponse, tags=["Computational Vector Database"])
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    api_key: str = Depends(verify_api_key)
):
    """
    Perform hybrid search combining semantic similarity with computational tensor operations.
    
    This is Tensorus's unique "Computational Vector Database" capability, enabling searches
    based on both semantic meaning and mathematical tensor properties.
    
    Features:
    - Semantic vector similarity search
    - Computational tensor operation scoring
    - Mathematical property filtering
    - Scientific workflow integration
    """
    import time
    start_time = time.time()
    
    try:
        # Validate weights
        if abs(request.similarity_weight + request.computation_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="similarity_weight and computation_weight must sum to 1.0"
            )
        
        # Convert tensor operations
        tensor_operations = []
        for op_dict in request.tensor_operations:
            operation = TensorOperation(
                operation_name=op_dict.get("operation_name", ""),
                function=None,  # Will be resolved by hybrid search engine
                parameters=op_dict.get("parameters", {}),
                description=op_dict.get("description", ""),
                computational_cost=op_dict.get("computational_cost", 1.0)
            )
            tensor_operations.append(operation)
        
        # Create hybrid query
        query = HybridQuery(
            text_query=request.text_query,
            tensor_operations=tensor_operations,
            similarity_weight=request.similarity_weight,
            computation_weight=request.computation_weight,
            filters=request.filters,
            k=request.k,
            namespace=request.namespace,
            tenant_id=request.tenant_id
        )
        
        # Execute hybrid search
        results = await hybrid_search_engine.hybrid_search(query, request.dataset_name)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "record_id": result.record_id,
                "semantic_score": result.semantic_score,
                "computational_score": result.computational_score,
                "hybrid_score": result.hybrid_score,
                "rank": result.rank,
                "source_text": result.source_text,
                "tensor_shape": list(result.tensor_shape) if result.tensor_shape else None,
                "computational_lineage": result.computational_lineage,
                "metadata": result.metadata
            }
            formatted_results.append(formatted_result)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return HybridSearchResponse(
            success=True,
            message=f"Found {len(results)} hybrid search results",
            results=formatted_results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            semantic_weight=request.similarity_weight,
            computation_weight=request.computation_weight
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )


@router.post("/tensor-workflow", response_model=TensorWorkflowResponse, tags=["Computational Vector Database"])
async def execute_tensor_workflow(
    request: TensorWorkflowRequest,
    hybrid_search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    api_key: str = Depends(verify_api_key)
):
    """
    Execute complex tensor workflows with semantic context and computational lineage tracking.
    
    This endpoint enables scientific computing workflows where tensors are selected based
    on semantic relevance and then processed through mathematical operations with full
    lineage tracking.
    
    Features:
    - Semantic tensor selection
    - Mathematical operation chaining
    - Computational lineage tracking
    - Intermediate result storage
    - Scientific provenance
    """
    try:
        # Convert tensor operations
        tensor_operations = []
        for op_dict in request.operations:
            operation = TensorOperation(
                operation_name=op_dict.get("operation_name", ""),
                function=None,
                parameters=op_dict.get("parameters", {}),
                description=op_dict.get("description", ""),
                computational_cost=op_dict.get("computational_cost", 1.0)
            )
            tensor_operations.append(operation)
        
        # Execute workflow
        workflow_result = await hybrid_search_engine.execute_tensor_workflow(
            workflow_query=request.workflow_query,
            dataset_name=request.dataset_name,
            operations=tensor_operations,
            save_intermediates=request.save_intermediates
        )
        
        if "error" in workflow_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=workflow_result["error"]
            )
        
        return TensorWorkflowResponse(
            success=True,
            workflow_id=workflow_result["workflow_id"],
            operations_executed=workflow_result["operations_executed"],
            final_result=workflow_result["final_result"],
            intermediate_results=workflow_result["intermediate_results"],
            computational_lineage=workflow_result["computational_lineage"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tensor workflow execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )


@router.post("/index/build", response_model=VectorIndexResponse, tags=["Vector Database"])
async def build_vector_index(
    request: VectorIndexRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Build or rebuild vector index with geometric partitioning and freshness layers.
    
    Features:
    - Geometric partitioning for efficient similarity search
    - Configurable number of partitions
    - Multiple distance metrics
    - Performance optimization
    """
    try:
        index_stats = await embedding_agent.build_vector_index(
            dataset_name=request.dataset_name,
            index_type=request.index_type,
            metric=request.metric,
            num_partitions=request.num_partitions
        )
        
        if "error" in index_stats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=index_stats["error"]
            )
        
        return VectorIndexResponse(
            success=True,
            message=f"Successfully built {request.index_type} index for dataset '{request.dataset_name}'",
            index_stats=index_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector index building failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index building failed: {str(e)}"
        )


@router.get("/models", response_model=ModelListResponse, tags=["Vector Database"])
async def list_available_models(
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    List all available embedding models across providers.
    
    Returns information about supported models including:
    - Sentence Transformers models
    - OpenAI embedding models  
    - Model dimensions and capabilities
    - Cost information where available
    """
    try:
        models_by_provider = embedding_agent.list_available_models()
        
        # Convert to serializable format
        serializable_providers = {}
        total_models = 0
        
        for provider_name, models in models_by_provider.items():
            serializable_providers[provider_name] = [
                {
                    "name": model.name,
                    "provider": model.provider,
                    "dimension": model.dimension,
                    "max_tokens": model.max_tokens,
                    "description": model.description,
                    "supports_batch": model.supports_batch,
                    "cost_per_1k_tokens": model.cost_per_1k_tokens
                }
                for model in models
            ]
            total_models += len(models)
        
        return ModelListResponse(
            success=True,
            providers=serializable_providers,
            total_models=total_models
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/stats/{dataset_name}", tags=["Vector Database"])
async def get_embedding_stats(
    dataset_name: str = Path(..., description="Name of the dataset"),
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Get comprehensive statistics about embeddings and vector indexes in a dataset.
    
    Returns:
    - Total embedding count
    - Vector index statistics
    - Memory usage information
    - Performance metrics
    """
    try:
        stats = embedding_agent.get_embedding_stats(dataset_name)
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Getting embedding stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedding stats: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse, tags=["Vector Database"])
async def get_performance_metrics(
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    hybrid_search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    api_key: str = Depends(verify_api_key)
):
    """
    Get comprehensive performance metrics for the vector database system.
    
    Returns:
    - Embedding generation metrics
    - Search performance metrics
    - Computational lineage statistics
    - Cache hit rates and performance
    """
    try:
        embedding_metrics = embedding_agent.get_metrics()
        search_metrics = hybrid_search_engine.get_metrics()
        lineage_metrics = hybrid_search_engine.get_lineage_stats()
        
        return MetricsResponse(
            success=True,
            embedding_metrics=embedding_metrics,
            search_metrics=search_metrics,
            lineage_metrics=lineage_metrics
        )
        
    except Exception as e:
        logger.error(f"Getting performance metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.delete("/vectors/{dataset_name}", tags=["Vector Database"]) 
async def delete_vectors(
    dataset_name: str = Path(..., description="Dataset name"),
    vector_ids: List[str] = Query(..., description="Vector IDs to delete"),
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete specific vectors from dataset and vector indexes.
    
    This operation:
    - Removes vectors from tensor storage
    - Updates vector indexes (marked as deleted in freshness layer)
    - Maintains computational lineage for tracking
    """
    try:
        if dataset_name not in embedding_agent.vector_indexes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector index for dataset '{dataset_name}' not found"
            )
        
        # Delete from vector index
        await embedding_agent.vector_indexes[dataset_name].delete_vectors(set(vector_ids))
        
        # Delete from tensor storage
        deleted_count = 0
        for vector_id in vector_ids:
            try:
                embedding_agent.tensor_storage.delete_tensor(dataset_name, vector_id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete tensor {vector_id}: {e}")
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} vectors from dataset '{dataset_name}'",
            "deleted_count": deleted_count,
            "requested_count": len(vector_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector deletion failed: {str(e)}"
        )