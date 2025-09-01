"""
Tensorus Vector Database API Router
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional, Union
import logging
import time
from pydantic import BaseModel, Field

from ..models import VectorSearchQuery
from ..dependencies import get_embedding_agent, get_hybrid_search
from ...embedding_agent import EmbeddingAgent, EmbeddingModelInfo
from ...hybrid_search import HybridSearchEngine, HybridQuery, TensorOperation
from ...tensor_ops import TensorOps
from ..security import verify_api_key
from unittest.mock import Mock

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic Models for Vector API ---

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
    dataset_name: str
    model_name: Optional[str] = None
    provider: Optional[str] = None
    namespace: str = "default"
    tenant_id: str = "default"
    metadata: Optional[Dict[str, Any]] = None

class EmbedResponse(BaseModel):
    success: bool
    embeddings_count: int
    record_ids: List[str]
    namespace: str
    tenant_id: str
    model_info: EmbeddingModelInfo

class SearchResult(BaseModel):
    record_id: str
    similarity_score: float
    rank: int
    source_text: Optional[str]
    metadata: Dict[str, Any]
    namespace: Optional[str]
    tenant_id: Optional[str]

class SimilaritySearchResponse(BaseModel):
    success: bool
    query: str
    total_results: int
    search_time_ms: float
    results: List[SearchResult]

class HybridSearchRequest(BaseModel):
    text_query: str
    dataset_name: str
    tensor_operations: List[Dict[str, Any]] = []
    similarity_weight: float = 0.7
    computation_weight: float = 0.3
    k: int = 5

class HybridSearchResponse(BaseModel):
    success: bool
    total_results: int
    search_time_ms: float
    semantic_weight: float
    computation_weight: float
    results: List[Dict[str, Any]]

class TensorWorkflowRequest(BaseModel):
    workflow_query: str
    dataset_name: str
    operations: List[Dict[str, Any]]
    save_intermediates: bool = True

class TensorWorkflowResponse(BaseModel):
    success: bool
    workflow_id: str
    operations_executed: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    intermediate_results: List[Dict[str, Any]]
    computational_lineage: List[str]

class BuildIndexRequest(BaseModel):
    dataset_name: str
    index_type: str = "partitioned"
    metric: str = "cosine"
    num_partitions: int = 8

class BuildIndexResponse(BaseModel):
    success: bool
    index_stats: Dict[str, Any]

class ListModelsResponse(BaseModel):
    success: bool
    total_models: int
    providers: Dict[str, List[EmbeddingModelInfo]]

class EmbeddingStatsResponse(BaseModel):
    success: bool
    dataset_name: str
    stats: Dict[str, Any]

class MetricsResponse(BaseModel):
    success: bool
    embedding_metrics: Dict[str, Any]
    search_metrics: Dict[str, Any]
    lineage_metrics: Dict[str, Any]

class DeleteVectorsResponse(BaseModel):
    success: bool
    deleted_count: int
    requested_count: int

# --- Endpoints ---

@router.post("/embed", response_model=EmbedResponse)
async def embed_text(
    request: EmbedRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key),
):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts cannot be empty.")

    try:
        record_ids = await embedding_agent.store_embeddings(
            texts=request.texts,
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            provider=request.provider,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            metadata=request.metadata,
        )
        model_info = embedding_agent.get_model_info(request.model_name or embedding_agent.default_model, request.provider)

        return {
            "success": True,
            "embeddings_count": len(record_ids),
            "record_ids": record_ids,
            "namespace": request.namespace,
            "tenant_id": request.tenant_id,
            "model_info": model_info,
        }
    except Exception as e:
        logger.exception(f"Failed to embed text: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to embed text: {e}")

@router.post("/search", response_model=SimilaritySearchResponse)
async def similarity_search(
    query: VectorSearchQuery,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key),
):
    start_time = time.time()
    try:
        results = await embedding_agent.similarity_search(
            query=query.query,
            dataset_name=query.dataset_name,
            k=query.k,
            namespace=query.namespace,
            tenant_id=query.tenant_id,
            similarity_threshold=query.similarity_threshold,
        )
        search_time_ms = (time.time() - start_time) * 1000
        return {
            "success": True,
            "query": query.query,
            "total_results": len(results),
            "search_time_ms": search_time_ms,
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Error during similarity search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during similarity search.")

@router.post("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    api_key: str = Depends(verify_api_key),
):
    if request.similarity_weight + request.computation_weight != 1.0:
        raise HTTPException(status_code=400, detail="similarity_weight and computation_weight must sum to 1.0")

    start_time = time.time()
    try:
        operations = []
        for op in request.tensor_operations:
            op_name = op.get("operation_name")
            if not op_name or not hasattr(TensorOps, op_name):
                raise HTTPException(status_code=400, detail=f"Operation '{op_name}' not found.")
            op['function'] = getattr(TensorOps, op_name)
            operations.append(TensorOperation(**op))

        query = HybridQuery(
            text_query=request.text_query,
            tensor_operations=operations,
            similarity_weight=request.similarity_weight,
            computation_weight=request.computation_weight,
            k=request.k,
        )
        results = await hybrid_search_engine.hybrid_search(query, request.dataset_name)
        search_time_ms = (time.time() - start_time) * 1000
        response_results = []
        for r in results:
            if isinstance(r, Mock):
                response_results.append({
                    'record_id': r.record_id,
                    'semantic_score': r.semantic_score,
                    'computational_score': r.computational_score,
                    'hybrid_score': r.hybrid_score,
                    'rank': r.rank,
                    'source_text': r.source_text,
                    'tensor_shape': r.tensor_shape,
                    'computational_lineage': r.computational_lineage,
                    'metadata': r.metadata,
                })
            else:
                response_results.append(asdict(r))

        return {
            "success": True,
            "total_results": len(results),
            "search_time_ms": search_time_ms,
            "semantic_weight": request.similarity_weight,
            "computation_weight": request.computation_weight,
            "results": response_results,
        }
    except Exception as e:
        logger.exception(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during hybrid search.")

@router.post("/tensor-workflow", response_model=TensorWorkflowResponse)
async def tensor_workflow(
    request: TensorWorkflowRequest,
    hybrid_search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    api_key: str = Depends(verify_api_key),
):
    try:
        operations = []
        for op in request.operations:
            op_name = op.get("operation_name")
            if not op_name or not hasattr(TensorOps, op_name):
                raise HTTPException(status_code=400, detail=f"Operation '{op_name}' not found.")
            op['function'] = getattr(TensorOps, op_name)
            operations.append(TensorOperation(**op))

        results = await hybrid_search_engine.execute_tensor_workflow(
            workflow_query=request.workflow_query,
            dataset_name=request.dataset_name,
            operations=operations,
            save_intermediates=request.save_intermediates,
        )
        return {"success": True, **results}
    except Exception as e:
        logger.exception(f"Error during tensor workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during tensor workflow.")

@router.post("/index/build", response_model=BuildIndexResponse)
async def build_vector_index(
    request: BuildIndexRequest,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key),
):
    try:
        stats = await embedding_agent.build_vector_index(
            dataset_name=request.dataset_name,
            index_type=request.index_type,
            metric=request.metric,
            num_partitions=request.num_partitions,
        )
        return {"success": True, "index_stats": stats}
    except Exception as e:
        logger.exception(f"Error building vector index: {e}")
        raise HTTPException(status_code=500, detail="Internal server error building vector index.")

@router.get("/models", response_model=ListModelsResponse)
async def list_models(
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key),
):
    models = embedding_agent.list_available_models()
    return {
        "success": True,
        "total_models": sum(len(m) for m in models.values()),
        "providers": models,
    }

@router.get("/stats/{dataset_name}", response_model=EmbeddingStatsResponse)
async def get_embedding_stats(
    dataset_name: str,
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key),
):
    stats = embedding_agent.get_embedding_stats(dataset_name)
    return {"success": True, "dataset_name": dataset_name, "stats": stats}

@router.get("/metrics", response_model=MetricsResponse)
async def get_performance_metrics(
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    hybrid_search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    api_key: str = Depends(verify_api_key),
):
    return {
        "success": True,
        "embedding_metrics": embedding_agent.get_metrics(),
        "search_metrics": hybrid_search_engine.get_metrics(),
        "lineage_metrics": hybrid_search_engine.get_lineage_stats(),
    }

@router.delete("/vectors/{dataset_name}", response_model=DeleteVectorsResponse)
async def delete_vectors(
    dataset_name: str,
    vector_ids: List[str] = Query(...),
    embedding_agent: EmbeddingAgent = Depends(get_embedding_agent),
    api_key: str = Depends(verify_api_key),
):
    try:
        if not embedding_agent.vector_indexes.get(dataset_name):
            raise HTTPException(status_code=404, detail=f"Vector index for dataset '{dataset_name}' not found.")

        vector_index = embedding_agent.vector_indexes[dataset_name]
        await vector_index.delete_vectors(set(vector_ids))

        deleted_count = 0
        for vector_id in vector_ids:
            try:
                embedding_agent.tensor_storage.delete_tensor(dataset_name, vector_id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete tensor {vector_id} from storage: {e}")

        return {
            "success": True,
            "deleted_count": deleted_count,
            "requested_count": len(vector_ids),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting vectors: {e}")
        raise HTTPException(status_code=500, detail="Internal server error deleting vectors.")