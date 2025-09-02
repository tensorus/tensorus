"""
Tensorus Hybrid Search Engine - Advanced search combining tensor operations with semantic similarity.

This module implements the unique "Computational Vector Database" capability by combining:
- Semantic vector similarity search
- Mathematical tensor operations
- Complex multi-dimensional queries
- Scientific computational lineage tracking
- Advanced tensor decomposition workflows

Key Features:
- Multi-modal query processing (text + tensor operations)
- Tensor operation chaining with semantic context
- Mathematical provenance tracking
- Scientific workflow integration
- Real-time computational lineage
"""

import asyncio
import logging
import numpy as np
import torch
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from uuid import uuid4

from .tensor_ops import TensorOps
from .tensor_storage import TensorStorage
from .embedding_agent import EmbeddingAgent
from .vector_database import SearchResult, VectorMetadata
from .metadata.schemas import ComputationalMetadata, LineageMetadata

logger = logging.getLogger(__name__)


@dataclass
class TensorOperation:
    """Represents a tensor operation with metadata."""
    operation_name: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    computational_cost: float = 1.0
    

@dataclass
class HybridQuery:
    """Represents a hybrid search query combining semantic and computational elements."""
    text_query: Optional[str] = None
    tensor_operations: List[TensorOperation] = field(default_factory=list)
    similarity_weight: float = 0.7
    computation_weight: float = 0.3
    filters: Dict[str, Any] = field(default_factory=dict)
    k: int = 10
    namespace: str = "default"
    tenant_id: str = "default"
    # Pagination and parallelization parameters
    offset: int = 0
    limit: Optional[int] = None
    max_workers: int = 4
    enable_parallel_scoring: bool = True
    

@dataclass
class HybridSearchResult:
    """Result from hybrid search combining semantic similarity and computational relevance."""
    record_id: str
    semantic_score: float
    computational_score: float
    hybrid_score: float
    rank: int
    source_text: Optional[str] = None
    tensor_shape: Optional[Tuple[int, ...]] = None
    computational_lineage: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensor_data: Optional[torch.Tensor] = None


class ComputationalScorer:
    """Scores tensors based on computational relevance and mathematical properties."""
    
    def __init__(self, tensor_ops: TensorOps):
        self.tensor_ops = tensor_ops
        
    def score_tensor_properties(self, tensor: torch.Tensor, query: HybridQuery) -> float:
        """Score tensor based on mathematical properties relevant to query."""
        score = 0.0
        
        # Shape compatibility scoring
        if "preferred_shape" in query.filters:
            preferred_shape = query.filters["preferred_shape"]
            if isinstance(preferred_shape, (list, tuple)):
                shape_similarity = self._compute_shape_similarity(tensor.shape, preferred_shape)
                score += shape_similarity * 0.3
                
        # Sparsity scoring
        if "sparsity_preference" in query.filters:
            sparsity = self._compute_sparsity(tensor)
            target_sparsity = query.filters["sparsity_preference"]
            sparsity_score = 1.0 - abs(sparsity - target_sparsity)
            score += sparsity_score * 0.2
            
        # Rank scoring for matrices
        if tensor.ndim == 2 and "rank_preference" in query.filters:
            rank = torch.matrix_rank(tensor).item()
            target_rank = query.filters["rank_preference"]
            rank_score = 1.0 - abs(rank - target_rank) / max(tensor.shape)
            score += rank_score * 0.2
            
        # Norm-based scoring
        if "norm_range" in query.filters:
            norm_range = query.filters["norm_range"]
            tensor_norm = torch.norm(tensor).item()
            if norm_range[0] <= tensor_norm <= norm_range[1]:
                score += 0.3
                
        return min(score, 1.0)
        
    def score_operation_compatibility(self, tensor: torch.Tensor, operations: List[TensorOperation]) -> float:
        """Score how well tensor is suited for specified operations."""
        if not operations:
            return 0.0
            
        total_score = 0.0
        
        for op in operations:
            op_score = self._score_single_operation(tensor, op)
            total_score += op_score
            
        return min(total_score / len(operations), 1.0)
        
    def _score_single_operation(self, tensor: torch.Tensor, operation: TensorOperation) -> float:
        """Score compatibility with a single operation."""
        op_name = operation.operation_name.lower()
        
        # SVD operations prefer matrices
        if "svd" in op_name and tensor.ndim == 2:
            return 0.8
            
        # Tensor decompositions prefer 3D+ tensors
        if any(decomp in op_name for decomp in ["cp", "tucker", "tt"]) and tensor.ndim >= 3:
            return 0.9
            
        # Matrix operations prefer 2D tensors
        if any(mat_op in op_name for mat_op in ["eigenvalue", "qr", "cholesky"]) and tensor.ndim == 2:
            return 0.8
            
        # Convolution operations prefer 3D+ tensors
        if "conv" in op_name and tensor.ndim >= 3:
            return 0.7
            
        # Element-wise operations work with any tensor
        if any(elem_op in op_name for elem_op in ["add", "multiply", "normalize"]):
            return 0.5
            
        return 0.1  # Low compatibility by default
        
    def _compute_shape_similarity(self, actual_shape: Tuple[int, ...], preferred_shape: Tuple[int, ...]) -> float:
        """Compute similarity between tensor shapes."""
        if len(actual_shape) != len(preferred_shape):
            return 0.0
            
        similarities = []
        for actual_dim, preferred_dim in zip(actual_shape, preferred_shape):
            similarity = min(actual_dim, preferred_dim) / max(actual_dim, preferred_dim)
            similarities.append(similarity)
            
        return np.mean(similarities)
        
    def _compute_sparsity(self, tensor: torch.Tensor) -> float:
        """Compute sparsity ratio of tensor."""
        total_elements = tensor.numel()
        zero_elements = torch.sum(torch.abs(tensor) < 1e-8).item()
        return zero_elements / total_elements if total_elements > 0 else 0.0


class LineageTracker:
    """Tracks computational lineage for tensor operations."""
    
    def __init__(self):
        self.lineage_graph: Dict[str, List[str]] = {}
        self.operation_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def record_operation(self, 
                        input_tensor_ids: List[str], 
                        output_tensor_id: str,
                        operation: TensorOperation,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a tensor operation in the lineage graph."""
        operation_id = str(uuid4())
        
        # Update lineage graph
        if output_tensor_id not in self.lineage_graph:
            self.lineage_graph[output_tensor_id] = []
            
        self.lineage_graph[output_tensor_id].extend(input_tensor_ids)
        
        # Record operation details
        operation_record = {
            "operation_id": operation_id,
            "operation_name": operation.operation_name,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_id": output_tensor_id,
            "parameters": operation.parameters,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        if output_tensor_id not in self.operation_history:
            self.operation_history[output_tensor_id] = []
            
        self.operation_history[output_tensor_id].append(operation_record)
        
        return operation_id
        
    def get_lineage(self, tensor_id: str, depth: int = 5) -> List[str]:
        """Get computational lineage for a tensor."""
        lineage = []
        current_level = [tensor_id]
        
        for _ in range(depth):
            next_level = []
            for tid in current_level:
                if tid in self.lineage_graph:
                    parents = self.lineage_graph[tid]
                    next_level.extend(parents)
                    lineage.extend(parents)
                    
            current_level = list(set(next_level))
            if not current_level:
                break
                
        return list(set(lineage))
        
    def get_operation_history(self, tensor_id: str) -> List[Dict[str, Any]]:
        """Get operation history for a tensor."""
        return self.operation_history.get(tensor_id, [])


class HybridSearchEngine:
    """
    Advanced hybrid search engine combining semantic similarity with tensor operations.
    
    This is the core of Tensorus's "Computational Vector Database" capabilities,
    enabling complex queries that combine semantic meaning with mathematical operations.
    """
    
    def __init__(self,
                 tensor_storage: TensorStorage,
                 embedding_agent: EmbeddingAgent,
                 tensor_ops: TensorOps,
                 max_workers: int = 4):
        
        self.tensor_storage = tensor_storage
        self.embedding_agent = embedding_agent
        self.tensor_ops = tensor_ops
        self.computational_scorer = ComputationalScorer(tensor_ops)
        self.lineage_tracker = LineageTracker()
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Search performance metrics
        self.metrics = {
            "total_searches": 0,
            "average_search_time": 0.0,
            "semantic_search_ratio": 0.0,
            "computational_search_ratio": 0.0,
            "parallel_operations": 0,
            "sequential_operations": 0
        }
        
    async def hybrid_search(self, query: HybridQuery, dataset_name: str) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining semantic similarity and computational relevance.
        
        This method represents the core innovation of Tensorus - the ability to search
        for tensors based on both semantic meaning and mathematical properties.
        """
        start_time = datetime.utcnow()
        
        # Initialize results containers
        semantic_results = {}
        computational_scores = {}
        
        # Phase 1: Semantic Search (if text query provided)
        if query.text_query:
            logger.info(f"Performing semantic search for: '{query.text_query}'")
            
            semantic_search_results = await self.embedding_agent.similarity_search(
                query=query.text_query,
                dataset_name=dataset_name,
                k=query.k * 2,  # Get more candidates for computational filtering
                namespace=query.namespace,
                tenant_id=query.tenant_id
            )
            
            for result in semantic_search_results:
                semantic_results[result["record_id"]] = result["similarity_score"]
                
        # Phase 2: Computational Relevance Scoring with Pagination
        logger.info("Computing tensor operation compatibility scores")
        
        # Get candidate tensor IDs with pagination support
        candidate_tensor_ids = self._get_candidate_tensor_ids(
            semantic_results, dataset_name, query.offset, query.limit
        )
        
        # Score computational relevance (with parallel processing if enabled)
        if query.enable_parallel_scoring and len(candidate_tensor_ids) > 10:
            computational_scores = await self._score_tensors_parallel(
                candidate_tensor_ids, dataset_name, query
            )
        else:
            computational_scores = self._score_tensors_sequential(
                candidate_tensor_ids, dataset_name, query
            )
                
        # Phase 3: Hybrid Score Combination
        logger.info("Combining semantic and computational scores")
        
        hybrid_results = []
        
        # Get all unique tensor IDs
        all_tensor_ids = set(semantic_results.keys()) | set(computational_scores.keys())
        
        for tensor_id in all_tensor_ids:
            semantic_score = semantic_results.get(tensor_id, 0.0)
            computational_score = computational_scores.get(tensor_id, 0.0)
            
            # Skip if both scores are zero
            if semantic_score == 0.0 and computational_score == 0.0:
                continue
                
            # Compute hybrid score
            hybrid_score = (
                semantic_score * query.similarity_weight + 
                computational_score * query.computation_weight
            )
            
            # Get tensor metadata
            try:
                tensor_record = self.tensor_storage.get_tensor_by_id(dataset_name, tensor_id)
                tensor = tensor_record["tensor"]
                metadata = tensor_record.get("metadata", {})
                
                # Get computational lineage
                lineage = self.lineage_tracker.get_lineage(tensor_id)
                
                result = HybridSearchResult(
                    record_id=tensor_id,
                    semantic_score=semantic_score,
                    computational_score=computational_score,
                    hybrid_score=hybrid_score,
                    rank=0,  # Will be set after sorting
                    source_text=metadata.get("source_text"),
                    tensor_shape=tuple(tensor.shape),
                    computational_lineage=lineage,
                    metadata=metadata,
                    tensor_data=tensor
                )
                
                hybrid_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to create result for tensor {tensor_id}: {e}")
                
        # Phase 4: Ranking and Filtering with Pagination
        logger.info("Ranking and filtering results")
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Apply pagination to results
        total_results = len(hybrid_results)
        start_idx = query.offset
        end_idx = start_idx + query.k if query.limit is None else min(start_idx + query.k, start_idx + query.limit)
        
        paginated_results = hybrid_results[start_idx:end_idx]
        
        # Assign ranks based on overall ranking (not just paginated subset)
        for i, result in enumerate(paginated_results):
            result.rank = start_idx + i + 1
            
        # Update metrics
        search_time = (datetime.utcnow() - start_time).total_seconds()
        self._update_metrics(
            search_time, 
            bool(query.text_query), 
            bool(query.tensor_operations),
            query.enable_parallel_scoring
        )
        
        logger.info(
            f"Hybrid search completed: {len(paginated_results)} results (of {total_results} total) "
            f"in {search_time:.3f}s, offset={query.offset}"
        )
        
        return paginated_results
        
    async def execute_tensor_workflow(self, 
                                    workflow_query: str,
                                    dataset_name: str,
                                    operations: List[TensorOperation],
                                    save_intermediates: bool = True) -> Dict[str, Any]:
        """
        Execute a complex tensor workflow with semantic context tracking.
        
        This method enables scientific computing workflows with full lineage tracking.
        """
        logger.info(f"Executing tensor workflow: '{workflow_query}'")
        
        # Step 1: Find relevant tensors using semantic search
        initial_search = HybridQuery(
            text_query=workflow_query,
            tensor_operations=operations,
            k=5
        )
        
        candidate_tensors = await self.hybrid_search(initial_search, dataset_name)
        
        if not candidate_tensors:
            return {"error": "No suitable tensors found for workflow"}
            
        # Step 2: Execute operations on best candidate
        input_tensor = candidate_tensors[0].tensor_data
        input_tensor_id = candidate_tensors[0].record_id
        
        workflow_results = {
            "workflow_id": str(uuid4()),
            "input_tensor_id": input_tensor_id,
            "input_tensor_shape": tuple(input_tensor.shape),
            "operations_executed": [],
            "intermediate_results": [],
            "final_result": None,
            "computational_lineage": [input_tensor_id]
        }
        
        current_tensor = input_tensor
        current_tensor_id = input_tensor_id
        
        # Step 3: Execute each operation
        for i, operation in enumerate(operations):
            try:
                logger.info(f"Executing operation {i+1}/{len(operations)}: {operation.operation_name}")
                
                # Get the tensor operation function
                if hasattr(self.tensor_ops, operation.operation_name):
                    op_function = getattr(self.tensor_ops, operation.operation_name)
                    
                    # Execute operation
                    result_tensor = op_function(current_tensor, **operation.parameters)
                    
                    # Generate new tensor ID
                    result_tensor_id = str(uuid4())
                    
                    # Record lineage
                    operation_id = self.lineage_tracker.record_operation(
                        input_tensor_ids=[current_tensor_id],
                        output_tensor_id=result_tensor_id,
                        operation=operation,
                        metadata={
                            "workflow_id": workflow_results["workflow_id"],
                            "step": i + 1,
                            "operation_description": operation.description
                        }
                    )
                    
                    # Save intermediate result if requested
                    if save_intermediates:
                        intermediate_metadata = {
                            "workflow_id": workflow_results["workflow_id"],
                            "operation_step": i + 1,
                            "operation_name": operation.operation_name,
                            "operation_id": operation_id,
                            "parent_tensor_id": current_tensor_id,
                            "created_at": datetime.utcnow().isoformat()
                        }
                        
                        # Create intermediate dataset if needed
                        intermediate_dataset = f"{dataset_name}_workflow_intermediates"
                        if not self.tensor_storage.dataset_exists(intermediate_dataset):
                            self.tensor_storage.create_dataset(intermediate_dataset)
                            
                        self.tensor_storage.insert(
                            name=intermediate_dataset,
                            tensor=result_tensor,
                            metadata=intermediate_metadata
                        )
                        
                        workflow_results["intermediate_results"].append({
                            "step": i + 1,
                            "tensor_id": result_tensor_id,
                            "tensor_shape": tuple(result_tensor.shape),
                            "operation_name": operation.operation_name
                        })
                    
                    # Update for next iteration
                    current_tensor = result_tensor
                    current_tensor_id = result_tensor_id
                    workflow_results["computational_lineage"].append(result_tensor_id)
                    
                    workflow_results["operations_executed"].append({
                        "operation_name": operation.operation_name,
                        "parameters": operation.parameters,
                        "input_shape": tuple(input_tensor.shape if i == 0 else current_tensor.shape),
                        "output_shape": tuple(result_tensor.shape),
                        "operation_id": operation_id
                    })
                    
                else:
                    error_msg = f"Operation '{operation.operation_name}' not available"
                    logger.error(error_msg)
                    workflow_results["operations_executed"].append({
                        "operation_name": operation.operation_name,
                        "error": error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Failed to execute operation '{operation.operation_name}': {e}"
                logger.error(error_msg)
                workflow_results["operations_executed"].append({
                    "operation_name": operation.operation_name,
                    "error": error_msg
                })
                break
                
        # Step 4: Store final result
        final_tensor_id = str(uuid4())
        final_metadata = {
            "workflow_id": workflow_results["workflow_id"],
            "workflow_query": workflow_query,
            "final_result": True,
            "operations_count": len(operations),
            "computational_lineage": workflow_results["computational_lineage"],
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Create dataset if it doesn't exist
        results_dataset = f"{dataset_name}_workflow_results"
        if not self.tensor_storage.dataset_exists(results_dataset):
            self.tensor_storage.create_dataset(results_dataset)
        
        final_tensor_id = self.tensor_storage.insert(
            name=results_dataset,
            tensor=current_tensor,
            metadata=final_metadata
        )
        
        workflow_results["final_result"] = {
            "tensor_id": final_tensor_id,
            "tensor_shape": tuple(current_tensor.shape),
            "dataset": f"{dataset_name}_workflow_results"
        }
        
        logger.info(f"Workflow completed: {len(operations)} operations executed")
        
        return workflow_results
        
    def _get_candidate_tensor_ids(self, semantic_results: Dict[str, float], 
                                 dataset_name: str, offset: int, limit: Optional[int]) -> List[str]:
        """Get candidate tensor IDs with pagination support."""
        if semantic_results:
            # Use semantic search results
            sorted_results = sorted(semantic_results.items(), key=lambda x: x[1], reverse=True)
            candidate_tensor_ids = [tid for tid, _ in sorted_results]
        else:
            # Get all tensors in dataset with pagination
            if dataset_name in self.tensor_storage.datasets:
                all_metadata = self.tensor_storage.datasets[dataset_name]["metadata"]
                candidate_tensor_ids = [meta["record_id"] for meta in all_metadata]
            else:
                candidate_tensor_ids = []
                
        # Apply pagination to candidate list
        if limit is not None:
            end_idx = offset + limit
            candidate_tensor_ids = candidate_tensor_ids[offset:end_idx]
        elif offset > 0:
            candidate_tensor_ids = candidate_tensor_ids[offset:]
            
        return candidate_tensor_ids
        
    def _score_tensors_sequential(self, tensor_ids: List[str], dataset_name: str, 
                                query: HybridQuery) -> Dict[str, float]:
        """Score tensors sequentially (original method)."""
        computational_scores = {}
        
        for tensor_id in tensor_ids:
            try:
                tensor_record = self.tensor_storage.get_tensor_by_id(dataset_name, tensor_id)
                tensor = tensor_record["tensor"]
                
                # Compute property-based score
                property_score = self.computational_scorer.score_tensor_properties(tensor, query)
                
                # Compute operation compatibility score
                operation_score = self.computational_scorer.score_operation_compatibility(
                    tensor, query.tensor_operations
                )
                
                # Combined computational score
                computational_scores[tensor_id] = (property_score + operation_score) / 2
                
            except Exception as e:
                logger.warning(f"Failed to score tensor {tensor_id}: {e}")
                computational_scores[tensor_id] = 0.0
                
        return computational_scores
        
    async def _score_tensors_parallel(self, tensor_ids: List[str], dataset_name: str,
                                     query: HybridQuery) -> Dict[str, float]:
        """Score tensors in parallel for better performance."""
        computational_scores = {}
        
        def score_single_tensor(tensor_id: str) -> Tuple[str, float]:
            try:
                tensor_record = self.tensor_storage.get_tensor_by_id(dataset_name, tensor_id)
                tensor = tensor_record["tensor"]
                
                # Compute property-based score
                property_score = self.computational_scorer.score_tensor_properties(tensor, query)
                
                # Compute operation compatibility score
                operation_score = self.computational_scorer.score_operation_compatibility(
                    tensor, query.tensor_operations
                )
                
                # Combined computational score
                score = (property_score + operation_score) / 2
                return (tensor_id, score)
                
            except Exception as e:
                logger.warning(f"Failed to score tensor {tensor_id}: {e}")
                return (tensor_id, 0.0)
                
        # Submit scoring tasks to thread pool
        loop = asyncio.get_event_loop()
        futures = []
        
        for tensor_id in tensor_ids:
            future = loop.run_in_executor(self.thread_pool, score_single_tensor, tensor_id)
            futures.append(future)
            
        # Wait for all scoring to complete
        results = await asyncio.gather(*futures)
        
        # Collect results
        for tensor_id, score in results:
            computational_scores[tensor_id] = score
            
        return computational_scores
        
    async def paginated_search(self, query: HybridQuery, dataset_name: str, 
                              page_size: int = 50) -> Dict[str, Any]:
        """Perform paginated hybrid search for large result sets."""
        original_k = query.k
        original_offset = query.offset
        
        # Collect paginated results
        all_results = []
        current_offset = 0
        total_processed = 0
        
        while total_processed < original_k:
            # Update query for this page
            query.offset = current_offset
            query.k = min(page_size, original_k - total_processed)
            
            # Get page results
            page_results = await self.hybrid_search(query, dataset_name)
            
            if not page_results:
                break  # No more results
                
            all_results.extend(page_results)
            total_processed += len(page_results)
            current_offset += page_size
            
            # Break if we got fewer results than page size (indicating end of data)
            if len(page_results) < page_size:
                break
                
        # Restore original query parameters
        query.k = original_k
        query.offset = original_offset
        
        # Apply final offset and limit
        start_idx = original_offset
        end_idx = start_idx + original_k
        final_results = all_results[start_idx:end_idx]
        
        return {
            "results": final_results,
            "total_found": len(all_results),
            "page_size": page_size,
            "offset": original_offset,
            "limit": original_k,
            "has_more": len(all_results) >= end_idx
        }
        
    def _update_metrics(self, search_time: float, has_semantic: bool, 
                       has_computational: bool, used_parallel: bool = False) -> None:
        """Update search performance metrics."""
        self.metrics["total_searches"] += 1
        
        # Update average search time
        total_searches = self.metrics["total_searches"]
        current_avg = self.metrics["average_search_time"]
        self.metrics["average_search_time"] = (current_avg * (total_searches - 1) + search_time) / total_searches
        
        # Update search type ratios
        if has_semantic:
            self.metrics["semantic_search_ratio"] = (
                (self.metrics["semantic_search_ratio"] * (total_searches - 1) + 1.0) / total_searches
            )
            
        if has_computational:
            self.metrics["computational_search_ratio"] = (
                (self.metrics["computational_search_ratio"] * (total_searches - 1) + 1.0) / total_searches
            )
            
        # Update parallel vs sequential operation counts
        if used_parallel:
            self.metrics["parallel_operations"] += 1
        else:
            self.metrics["sequential_operations"] += 1
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid search performance metrics."""
        return self.metrics.copy()
        
    def get_lineage_stats(self) -> Dict[str, Any]:
        """Get computational lineage statistics."""
        return {
            "total_tensors_tracked": len(self.lineage_tracker.lineage_graph),
            "total_operations_recorded": sum(len(ops) for ops in self.lineage_tracker.operation_history.values()),
            "average_lineage_depth": np.mean([
                len(self.lineage_tracker.get_lineage(tid)) 
                for tid in self.lineage_tracker.lineage_graph.keys()
            ]) if self.lineage_tracker.lineage_graph else 0
        }
        
    def cleanup_resources(self) -> None:
        """Clean up thread pool resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            
    def __del__(self):
        """Ensure resources are cleaned up when object is destroyed."""
        try:
            self.cleanup_resources()
        except:
            pass  # Ignore errors during cleanup