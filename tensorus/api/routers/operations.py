"""Operation History and Lineage API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime

from ...operation_history import (
    OperationRecord, TensorLineage, OperationType, OperationStatus
)
from ...storage_ops_with_history import TensorStorageWithHistoryOps
from ..models import OperationHistoryRequest, OperationHistoryResponse, LineageResponse, OperationStatsResponse
from ..dependencies import get_storage_with_history
from ..security import verify_api_key

router = APIRouter()


@router.get("/operations/recent", response_model=OperationHistoryResponse)
async def get_recent_operations(
    limit: int = Query(100, description="Maximum number of operations to return", ge=1, le=1000),
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    status: Optional[str] = Query(None, description="Filter by operation status"),
    storage: TensorStorageWithHistoryOps = Depends(get_storage_with_history),
    api_key: str = Depends(verify_api_key)
):
    """Get recent operations with optional filtering."""
    try:
        operations = storage.ops.get_recent_operations(limit)
        
        # Apply filters
        if operation_type:
            try:
                op_type = OperationType(operation_type.lower())
                operations = [op for op in operations if op.operation_type == op_type]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation type: {operation_type}")
        
        if status:
            try:
                op_status = OperationStatus(status.lower())
                operations = [op for op in operations if op.status == op_status]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation status: {status}")
        
        # Convert to response format
        operation_data = []
        for op in operations:
            operation_data.append({
                "operation_id": str(op.operation_id),
                "operation_type": op.operation_type.value,
                "operation_name": op.operation_name,
                "status": op.status.value,
                "started_at": op.started_at.isoformat(),
                "completed_at": op.completed_at.isoformat() if op.completed_at else None,
                "duration_ms": op.duration_ms,
                "user_id": op.user_id,
                "session_id": op.session_id,
                "inputs": [
                    {
                        "tensor_id": str(inp.tensor_id) if inp.tensor_id else None,
                        "shape": inp.shape,
                        "dtype": inp.dtype,
                        "parameter_name": inp.parameter_name,
                        "is_tensor": inp.is_tensor,
                        "value": inp.value if not inp.is_tensor else None
                    } for inp in op.inputs
                ],
                "outputs": [
                    {
                        "tensor_id": str(out.tensor_id) if out.tensor_id else None,
                        "shape": out.shape,
                        "dtype": out.dtype,
                        "is_primary": out.is_primary
                    } for out in op.outputs
                ],
                "execution_info": {
                    "execution_time_ms": op.execution_info.execution_time_ms,
                    "memory_usage_mb": op.execution_info.memory_usage_mb,
                    "device": op.execution_info.device,
                    "hostname": op.execution_info.hostname
                } if op.execution_info else None,
                "error_message": op.error_message,
                "tags": op.tags
            })
        
        return OperationHistoryResponse(
            success=True,
            message=f"Retrieved {len(operation_data)} operations",
            count=len(operation_data),
            operations=operation_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve operations: {str(e)}")


@router.get("/operations/tensor/{tensor_id}", response_model=OperationHistoryResponse)
async def get_tensor_operations(
    tensor_id: str,
    storage: TensorStorageWithHistoryOps = Depends(get_storage_with_history),
    api_key: str = Depends(verify_api_key)
):
    """Get all operations that involved a specific tensor."""
    try:
        operations = storage.get_tensor_operation_history(tensor_id)
        
        operation_data = []
        for op in operations:
            operation_data.append({
                "operation_id": str(op.operation_id),
                "operation_type": op.operation_type.value,
                "operation_name": op.operation_name,
                "status": op.status.value,
                "started_at": op.started_at.isoformat(),
                "completed_at": op.completed_at.isoformat() if op.completed_at else None,
                "duration_ms": op.duration_ms,
                "user_id": op.user_id,
                "session_id": op.session_id,
                "inputs": [
                    {
                        "tensor_id": str(inp.tensor_id) if inp.tensor_id else None,
                        "shape": inp.shape,
                        "dtype": inp.dtype,
                        "parameter_name": inp.parameter_name,
                        "is_tensor": inp.is_tensor,
                        "value": inp.value if not inp.is_tensor else None
                    } for inp in op.inputs
                ],
                "outputs": [
                    {
                        "tensor_id": str(out.tensor_id) if out.tensor_id else None,
                        "shape": out.shape,
                        "dtype": out.dtype,
                        "is_primary": out.is_primary
                    } for out in op.outputs
                ],
                "execution_info": {
                    "execution_time_ms": op.execution_info.execution_time_ms,
                    "memory_usage_mb": op.execution_info.memory_usage_mb,
                    "device": op.execution_info.device,
                    "hostname": op.execution_info.hostname
                } if op.execution_info else None,
                "error_message": op.error_message,
                "tags": op.tags
            })
        
        return OperationHistoryResponse(
            success=True,
            message=f"Retrieved {len(operation_data)} operations for tensor {tensor_id}",
            count=len(operation_data),
            operations=operation_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tensor operations: {str(e)}")


@router.get("/lineage/tensor/{tensor_id}", response_model=LineageResponse)
async def get_tensor_lineage(
    tensor_id: str,
    include_operations: bool = Query(True, description="Include detailed operation information"),
    max_depth: Optional[int] = Query(None, description="Maximum lineage depth to retrieve"),
    storage: TensorStorageWithHistoryOps = Depends(get_storage_with_history),
    api_key: str = Depends(verify_api_key)
):
    """Get computational lineage for a tensor."""
    try:
        lineage = storage.get_tensor_lineage(tensor_id)
        
        if not lineage:
            raise HTTPException(status_code=404, detail=f"No lineage found for tensor {tensor_id}")
        
        # Convert lineage nodes
        lineage_nodes = []
        for tensor_str, node in lineage.lineage_nodes.items():
            if max_depth is None or node.depth <= max_depth:
                node_data = {
                    "tensor_id": tensor_str,
                    "operation_id": str(node.operation_id) if node.operation_id else None,
                    "parent_tensor_ids": [str(pid) for pid in node.parent_tensor_ids],
                    "created_at": node.created_at.isoformat(),
                    "depth": node.depth,
                    "is_root": node.is_root,
                    "is_leaf": node.is_leaf
                }
                lineage_nodes.append(node_data)
        
        # Convert operation records if requested
        operations = []
        if include_operations:
            for op_str, op in lineage.operation_records.items():
                if any(node["operation_id"] == op_str for node in lineage_nodes):
                    operations.append({
                        "operation_id": op_str,
                        "operation_type": op.operation_type.value,
                        "operation_name": op.operation_name,
                        "status": op.status.value,
                        "started_at": op.started_at.isoformat(),
                        "completed_at": op.completed_at.isoformat() if op.completed_at else None,
                        "duration_ms": op.duration_ms,
                        "execution_info": {
                            "execution_time_ms": op.execution_info.execution_time_ms,
                            "memory_usage_mb": op.execution_info.memory_usage_mb,
                            "device": op.execution_info.device,
                            "hostname": op.execution_info.hostname
                        } if op.execution_info else None
                    })
        
        return LineageResponse(
            success=True,
            message=f"Retrieved lineage for tensor {tensor_id}",
            tensor_id=tensor_id,
            root_tensor_ids=[str(rid) for rid in lineage.root_tensor_ids],
            max_depth=lineage.max_depth,
            total_operations=lineage.total_operations,
            lineage_nodes=lineage_nodes,
            operations=operations if include_operations else None,
            created_at=lineage.created_at.isoformat(),
            last_updated=lineage.last_updated.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tensor lineage: {str(e)}")


@router.get("/lineage/tensor/{tensor_id}/dot")
async def get_tensor_lineage_dot(
    tensor_id: str,
    storage: TensorStorageWithHistoryOps = Depends(get_storage_with_history),
    api_key: str = Depends(verify_api_key)
):
    """Get tensor lineage in DOT graph format for visualization."""
    try:
        dot_graph = storage.export_lineage_graph(tensor_id)
        
        if not dot_graph:
            raise HTTPException(status_code=404, detail=f"No lineage found for tensor {tensor_id}")
        
        return {
            "success": True,
            "message": f"Generated DOT graph for tensor {tensor_id}",
            "tensor_id": tensor_id,
            "dot_graph": dot_graph
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate DOT graph: {str(e)}")


@router.get("/lineage/tensor/{source_tensor_id}/path/{target_tensor_id}")
async def get_lineage_path(
    source_tensor_id: str,
    target_tensor_id: str,
    storage: TensorStorageWithHistoryOps = Depends(get_storage_with_history),
    api_key: str = Depends(verify_api_key)
):
    """Get the operation path between two tensors."""
    try:
        lineage = storage.get_tensor_lineage(target_tensor_id)
        
        if not lineage:
            raise HTTPException(status_code=404, detail=f"No lineage found for tensor {target_tensor_id}")
        
        try:
            source_uuid = UUID(source_tensor_id)
            target_uuid = UUID(target_tensor_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tensor ID format")
        
        path = lineage.get_operation_path(source_uuid, target_uuid)
        
        if not path:
            return {
                "success": False,
                "message": f"No path found from {source_tensor_id} to {target_tensor_id}",
                "source_tensor_id": source_tensor_id,
                "target_tensor_id": target_tensor_id,
                "path_exists": False
            }
        
        return {
            "success": True,
            "message": f"Found path from {source_tensor_id} to {target_tensor_id}",
            "source_tensor_id": source_tensor_id,
            "target_tensor_id": target_tensor_id,
            "path_exists": True,
            "path_nodes": [str(node_id) for node_id in path.path_nodes],
            "operations": [str(op_id) for op_id in path.operations],
            "total_depth": path.total_depth
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find lineage path: {str(e)}")


@router.get("/operations/statistics", response_model=OperationStatsResponse)
async def get_operation_statistics(
    storage: TensorStorageWithHistoryOps = Depends(get_storage_with_history),
    api_key: str = Depends(verify_api_key)
):
    """Get comprehensive operation statistics."""
    try:
        stats = storage.get_operation_stats()
        
        return OperationStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve operation statistics: {str(e)}")


@router.get("/operations/types")
async def get_operation_types(
    api_key: str = Depends(verify_api_key)
):
    """Get list of available operation types."""
    try:
        operation_types = [op_type.value for op_type in OperationType]
        
        return {
            "success": True,
            "message": "Retrieved available operation types",
            "operation_types": operation_types,
            "count": len(operation_types)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve operation types: {str(e)}")


@router.get("/operations/statuses")
async def get_operation_statuses(
    api_key: str = Depends(verify_api_key)
):
    """Get list of available operation statuses."""
    try:
        operation_statuses = [status.value for status in OperationStatus]
        
        return {
            "success": True,
            "message": "Retrieved available operation statuses",
            "operation_statuses": operation_statuses,
            "count": len(operation_statuses)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve operation statuses: {str(e)}")