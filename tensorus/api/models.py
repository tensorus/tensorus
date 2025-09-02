"""API Models for Tensorus."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class NQLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query string.", json_schema_extra={"example": "find image tensors from 'my_image_dataset' where metadata.source = 'web_scrape'"})


class TensorOutput(BaseModel):
    record_id: str = Field(..., description="Unique record ID assigned during ingestion.")
    shape: List[int] = Field(..., description="Shape of the retrieved tensor.")
    dtype: str = Field(..., description="Data type of the retrieved tensor.")
    data: Union[List[Any], int, float] = Field(..., description="Tensor data (nested list or scalar).")
    metadata: Dict[str, Any] = Field(..., description="Associated metadata.")


class NQLResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the query was successfully processed (syntax, execution).")
    message: str = Field(..., description="Status message (e.g., 'Query successful', 'Error parsing query').")
    count: Optional[int] = Field(None, description="Number of matching records found.")
    results: Optional[List[TensorOutput]] = Field(None, description="List of matching tensor records.")

class VectorSearchQuery(BaseModel):
    query: str = Field(..., description="The query text to search for.")
    dataset_name: str = Field(..., description="The name of the dataset to search in.")
    k: int = Field(5, description="The number of top results to return.")
    namespace: Optional[str] = Field(None, description="The namespace to search in.")
    tenant_id: Optional[str] = Field(None, description="The tenant ID to search in.")
    similarity_threshold: Optional[float] = Field(None, description="The similarity threshold.")
    include_vectors: bool = Field(False, description="Whether to include vectors in the response.")


# Operation History and Lineage Models

class OperationInputModel(BaseModel):
    tensor_id: Optional[str] = Field(None, description="ID of the input tensor")
    shape: List[int] = Field(..., description="Shape of the input tensor")
    dtype: str = Field(..., description="Data type of the input")
    parameter_name: Optional[str] = Field(None, description="Name of the parameter")
    is_tensor: bool = Field(True, description="Whether this input is a tensor")
    value: Optional[Any] = Field(None, description="Value for non-tensor parameters")


class OperationOutputModel(BaseModel):
    tensor_id: Optional[str] = Field(None, description="ID of the output tensor")
    shape: List[int] = Field(..., description="Shape of the output tensor")
    dtype: str = Field(..., description="Data type of the output")
    is_primary: bool = Field(True, description="Whether this is the primary output")


class ExecutionInfoModel(BaseModel):
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in megabytes")
    device: Optional[str] = Field(None, description="Device where operation was executed")
    hostname: Optional[str] = Field(None, description="Hostname where operation was executed")


class OperationModel(BaseModel):
    operation_id: str = Field(..., description="Unique operation identifier")
    operation_type: str = Field(..., description="Type of operation")
    operation_name: str = Field(..., description="Human-readable operation name")
    status: str = Field(..., description="Operation status")
    started_at: str = Field(..., description="Operation start time")
    completed_at: Optional[str] = Field(None, description="Operation completion time")
    duration_ms: Optional[float] = Field(None, description="Operation duration in milliseconds")
    user_id: Optional[str] = Field(None, description="User who initiated the operation")
    session_id: Optional[str] = Field(None, description="Session identifier")
    inputs: List[OperationInputModel] = Field(..., description="Operation inputs")
    outputs: List[OperationOutputModel] = Field(..., description="Operation outputs")
    execution_info: Optional[ExecutionInfoModel] = Field(None, description="Execution information")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    tags: List[str] = Field(..., description="Operation tags")


class OperationHistoryRequest(BaseModel):
    tensor_id: Optional[str] = Field(None, description="Filter by tensor ID")
    operation_type: Optional[str] = Field(None, description="Filter by operation type")
    status: Optional[str] = Field(None, description="Filter by operation status")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    start_time: Optional[str] = Field(None, description="Filter by start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Filter by end time (ISO format)")
    limit: int = Field(100, description="Maximum number of results")


class OperationHistoryResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    count: int = Field(..., description="Number of operations returned")
    operations: List[OperationModel] = Field(..., description="List of operations")


class LineageNodeModel(BaseModel):
    tensor_id: str = Field(..., description="Tensor identifier")
    operation_id: Optional[str] = Field(None, description="Operation that created this tensor")
    parent_tensor_ids: List[str] = Field(..., description="Parent tensor identifiers")
    created_at: str = Field(..., description="Creation timestamp")
    depth: int = Field(..., description="Depth in the lineage graph")
    is_root: bool = Field(..., description="Whether this is a root node")
    is_leaf: bool = Field(..., description="Whether this is a leaf node")


class LineageOperationModel(BaseModel):
    operation_id: str = Field(..., description="Operation identifier")
    operation_type: str = Field(..., description="Type of operation")
    operation_name: str = Field(..., description="Human-readable operation name")
    status: str = Field(..., description="Operation status")
    started_at: str = Field(..., description="Operation start time")
    completed_at: Optional[str] = Field(None, description="Operation completion time")
    duration_ms: Optional[float] = Field(None, description="Operation duration in milliseconds")
    execution_info: Optional[ExecutionInfoModel] = Field(None, description="Execution information")


class LineageResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    tensor_id: str = Field(..., description="Target tensor ID")
    root_tensor_ids: List[str] = Field(..., description="Root tensor identifiers")
    max_depth: int = Field(..., description="Maximum depth in the lineage")
    total_operations: int = Field(..., description="Total number of operations in lineage")
    lineage_nodes: List[LineageNodeModel] = Field(..., description="Lineage nodes")
    operations: Optional[List[LineageOperationModel]] = Field(None, description="Operations in lineage")
    created_at: str = Field(..., description="Lineage creation timestamp")
    last_updated: str = Field(..., description="Lineage last updated timestamp")


class OperationStatsResponse(BaseModel):
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    success_rate: float = Field(..., description="Success rate (0.0 to 1.0)")
    operations_by_type: Dict[str, int] = Field(..., description="Operations count by type")
    average_execution_times_ms: Dict[str, float] = Field(..., description="Average execution times by type")
    total_tensors_tracked: int = Field(..., description="Total number of tensors tracked")
    session_id: str = Field(..., description="Current session ID")
    created_at: str = Field(..., description="Statistics creation timestamp")
    last_updated: str = Field(..., description="Statistics last updated timestamp")