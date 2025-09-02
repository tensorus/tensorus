# operation_history.py
"""
Operation History and Lineage tracking for Tensorus.

This module provides comprehensive tracking of tensor operations and computational
lineage, enabling users to understand how tensors were created and transformed.
"""

from enum import Enum
from typing import List, Dict, Optional, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import json

import torch
from pydantic import BaseModel, Field, field_validator


class OperationType(str, Enum):
    """Types of operations that can be performed on tensors."""
    # Arithmetic operations
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    LOG = "log"
    
    # Matrix operations
    MATMUL = "matmul"
    DOT = "dot"
    OUTER = "outer"
    CROSS = "cross"
    
    # Reduction operations
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    
    # Reshaping operations
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    PERMUTE = "permute"
    FLATTEN = "flatten"
    SQUEEZE = "squeeze"
    UNSQUEEZE = "unsqueeze"
    
    # Concatenation operations
    CONCATENATE = "concatenate"
    STACK = "stack"
    
    # Advanced operations
    EINSUM = "einsum"
    SVD = "svd"
    QR_DECOMPOSITION = "qr_decomposition"
    EIGENDECOMPOSITION = "eigendecomposition"
    
    # Convolution operations
    CONVOLVE_1D = "convolve_1d"
    CONVOLVE_2D = "convolve_2d"
    CONVOLVE_3D = "convolve_3d"
    
    # Statistical operations
    VARIANCE = "variance"
    STD = "std"
    COVARIANCE = "covariance"
    CORRELATION = "correlation"
    
    # Norm operations
    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    FROBENIUS_NORM = "frobenius_norm"
    NUCLEAR_NORM = "nuclear_norm"
    P_NORM = "p_norm"
    
    # Storage operations
    STORE = "store"
    LOAD = "load"
    DELETE = "delete"
    
    # Custom operations
    CUSTOM = "custom"


class OperationStatus(str, Enum):
    """Status of an operation execution."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationExecutionInfo(BaseModel):
    """Information about the execution environment and performance."""
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    device: Optional[str] = None  # e.g., "cuda:0", "cpu"
    torch_version: Optional[str] = None
    python_version: Optional[str] = None
    hostname: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[str] = None
    
    @field_validator('execution_time_ms')
    def validate_execution_time(cls, v):
        if v is not None and v < 0:
            raise ValueError('Execution time cannot be negative')
        return v
    
    @field_validator('memory_usage_mb')
    def validate_memory_usage(cls, v):
        if v is not None and v < 0:
            raise ValueError('Memory usage cannot be negative')
        return v


class OperationInput(BaseModel):
    """Represents an input to an operation."""
    tensor_id: Optional[UUID] = None  # Reference to stored tensor
    shape: List[int]
    dtype: str
    device: str
    parameter_name: Optional[str] = None  # e.g., "t1", "t2", "dim"
    is_tensor: bool = True  # False for scalar parameters
    value: Optional[Any] = None  # For non-tensor parameters


class OperationOutput(BaseModel):
    """Represents an output from an operation."""
    tensor_id: Optional[UUID] = None  # Reference to stored result
    shape: List[int]
    dtype: str
    device: str
    is_primary: bool = True  # False for secondary outputs (e.g., indices from min/max)


class OperationRecord(BaseModel):
    """Complete record of a tensor operation."""
    operation_id: UUID = Field(default_factory=uuid4)
    operation_type: OperationType
    operation_name: str  # Human-readable name
    description: Optional[str] = None
    
    # Timing information
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: OperationStatus = OperationStatus.STARTED
    
    # Operation details
    inputs: List[OperationInput] = Field(default_factory=list)
    outputs: List[OperationOutput] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution context
    execution_info: Optional[OperationExecutionInfo] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_completed(self, outputs: List[OperationOutput], execution_info: Optional[OperationExecutionInfo] = None):
        """Mark the operation as completed."""
        self.completed_at = datetime.utcnow()
        self.status = OperationStatus.COMPLETED
        self.outputs = outputs
        if execution_info:
            self.execution_info = execution_info
    
    def mark_failed(self, error_message: str, error_traceback: Optional[str] = None):
        """Mark the operation as failed."""
        self.completed_at = datetime.utcnow()
        self.status = OperationStatus.FAILED
        self.error_message = error_message
        self.error_traceback = error_traceback
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate operation duration in milliseconds."""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None


class LineageNode(BaseModel):
    """Represents a node in the computational lineage graph."""
    tensor_id: UUID
    operation_id: Optional[UUID] = None  # None for root nodes (original data)
    parent_tensor_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Derived properties
    depth: int = 0  # Distance from root nodes
    is_root: bool = True  # True if no parents
    is_leaf: bool = True  # True if no children (computed dynamically)


class LineagePath(BaseModel):
    """Represents a path through the computational lineage."""
    path_id: UUID = Field(default_factory=uuid4)
    source_tensor_id: UUID
    target_tensor_id: UUID
    path_nodes: List[UUID] = Field(default_factory=list)  # Ordered list of tensor_ids
    operations: List[UUID] = Field(default_factory=list)  # Ordered list of operation_ids
    total_depth: int = 0


class TensorLineage(BaseModel):
    """Complete computational lineage for a tensor."""
    tensor_id: UUID
    lineage_nodes: Dict[str, LineageNode] = Field(default_factory=dict)  # Key: tensor_id str
    operation_records: Dict[str, OperationRecord] = Field(default_factory=dict)  # Key: operation_id str
    
    # Graph properties
    root_tensor_ids: List[UUID] = Field(default_factory=list)
    max_depth: int = 0
    total_operations: int = 0
    
    # Lineage metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def add_operation(self, operation: OperationRecord, input_tensor_ids: List[UUID], output_tensor_ids: List[UUID]):
        """Add an operation and update lineage graph."""
        # Store operation
        op_id_str = str(operation.operation_id)
        self.operation_records[op_id_str] = operation
        
        # Create nodes for output tensors
        for output_tensor_id in output_tensor_ids:
            output_tensor_str = str(output_tensor_id)
            
            # Determine depth and root status
            max_parent_depth = 0
            is_root = len(input_tensor_ids) == 0
            
            if not is_root:
                for input_tensor_id in input_tensor_ids:
                    input_tensor_str = str(input_tensor_id)
                    if input_tensor_str in self.lineage_nodes:
                        parent_depth = self.lineage_nodes[input_tensor_str].depth
                        max_parent_depth = max(max_parent_depth, parent_depth)
            
            node = LineageNode(
                tensor_id=output_tensor_id,
                operation_id=operation.operation_id,
                parent_tensor_ids=input_tensor_ids.copy(),
                depth=max_parent_depth + (0 if is_root else 1),
                is_root=is_root
            )
            
            self.lineage_nodes[output_tensor_str] = node
            
            # Update graph statistics
            if is_root and output_tensor_id not in self.root_tensor_ids:
                self.root_tensor_ids.append(output_tensor_id)
            
            self.max_depth = max(self.max_depth, node.depth)
        
        self.total_operations = len(self.operation_records)
        self.last_updated = datetime.utcnow()
    
    def get_ancestors(self, tensor_id: UUID) -> List[UUID]:
        """Get all ancestor tensor IDs for a given tensor."""
        ancestors = set()
        to_visit = [tensor_id]
        visited = set()
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            current_str = str(current_id)
            if current_str in self.lineage_nodes:
                node = self.lineage_nodes[current_str]
                for parent_id in node.parent_tensor_ids:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        to_visit.append(parent_id)
        
        return list(ancestors)
    
    def get_descendants(self, tensor_id: UUID) -> List[UUID]:
        """Get all descendant tensor IDs for a given tensor."""
        descendants = set()
        tensor_str = str(tensor_id)
        
        # Find all nodes that have this tensor as a parent
        for node_tensor_str, node in self.lineage_nodes.items():
            if tensor_id in node.parent_tensor_ids:
                descendant_id = UUID(node_tensor_str)
                descendants.add(descendant_id)
                # Recursively get descendants of this descendant
                descendants.update(self.get_descendants(descendant_id))
        
        return list(descendants)
    
    def get_operation_path(self, source_tensor_id: UUID, target_tensor_id: UUID) -> Optional[LineagePath]:
        """Get the operation path between two tensors."""
        # Simple BFS to find path
        queue = [(source_tensor_id, [source_tensor_id], [])]
        visited = {source_tensor_id}
        
        while queue:
            current_id, path, operations = queue.pop(0)
            
            if current_id == target_tensor_id:
                return LineagePath(
                    source_tensor_id=source_tensor_id,
                    target_tensor_id=target_tensor_id,
                    path_nodes=path,
                    operations=operations,
                    total_depth=len(operations)
                )
            
            # Find children of current tensor
            descendants = self.get_descendants(current_id)
            for desc_id in descendants:
                desc_str = str(desc_id)
                if desc_id not in visited and desc_str in self.lineage_nodes:
                    node = self.lineage_nodes[desc_str]
                    if current_id in node.parent_tensor_ids and node.operation_id:
                        visited.add(desc_id)
                        new_path = path + [desc_id]
                        new_operations = operations + [node.operation_id]
                        queue.append((desc_id, new_path, new_operations))
        
        return None  # No path found


class OperationHistory(BaseModel):
    """Complete operation history for the system."""
    operations: Dict[str, OperationRecord] = Field(default_factory=dict)  # Key: operation_id str
    tensor_lineages: Dict[str, TensorLineage] = Field(default_factory=dict)  # Key: tensor_id str
    
    # Global statistics
    total_operations: int = 0
    operations_by_type: Dict[OperationType, int] = Field(default_factory=dict)
    operations_by_status: Dict[OperationStatus, int] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def record_operation(self, operation: OperationRecord, input_tensor_ids: List[UUID], output_tensor_ids: List[UUID]):
        """Record a new operation in the history."""
        # Store operation
        op_id_str = str(operation.operation_id)
        self.operations[op_id_str] = operation
        
        # Update lineage for each output tensor
        for output_tensor_id in output_tensor_ids:
            output_tensor_str = str(output_tensor_id)
            
            if output_tensor_str not in self.tensor_lineages:
                self.tensor_lineages[output_tensor_str] = TensorLineage(tensor_id=output_tensor_id)
            
            lineage = self.tensor_lineages[output_tensor_str]
            lineage.add_operation(operation, input_tensor_ids, [output_tensor_id])
        
        # Update statistics
        self.total_operations = len(self.operations)
        self.operations_by_type[operation.operation_type] = self.operations_by_type.get(operation.operation_type, 0) + 1
        self.operations_by_status[operation.status] = self.operations_by_status.get(operation.status, 0) + 1
        self.last_updated = datetime.utcnow()
    
    def get_tensor_history(self, tensor_id: UUID) -> Optional[TensorLineage]:
        """Get the complete lineage for a tensor."""
        tensor_str = str(tensor_id)
        return self.tensor_lineages.get(tensor_str)
    
    def get_operations_by_type(self, operation_type: OperationType) -> List[OperationRecord]:
        """Get all operations of a specific type."""
        return [op for op in self.operations.values() if op.operation_type == operation_type]
    
    def get_operations_by_tensor(self, tensor_id: UUID) -> List[OperationRecord]:
        """Get all operations that involved a specific tensor (as input or output)."""
        tensor_str = str(tensor_id)
        result = []
        
        for operation in self.operations.values():
            # Check if tensor was an input
            tensor_involved = False
            for inp in operation.inputs:
                if inp.tensor_id and str(inp.tensor_id) == tensor_str:
                    result.append(operation)
                    tensor_involved = True
                    break
            
            if not tensor_involved:
                # Check if tensor was an output
                for out in operation.outputs:
                    if out.tensor_id and str(out.tensor_id) == tensor_str:
                        result.append(operation)
                        break
        
        return result
    
    def get_recent_operations(self, limit: int = 100) -> List[OperationRecord]:
        """Get the most recent operations."""
        operations = list(self.operations.values())
        operations.sort(key=lambda op: op.started_at, reverse=True)
        return operations[:limit]
    
    def export_lineage_dot(self, tensor_id: UUID) -> str:
        """Export tensor lineage as DOT graph format for visualization."""
        lineage = self.get_tensor_history(tensor_id)
        if not lineage:
            return ""
        
        dot_lines = ["digraph TensorLineage {", "  rankdir=TB;"]
        
        # Add nodes
        for tensor_str, node in lineage.lineage_nodes.items():
            tensor_uuid = UUID(tensor_str)
            label = f"Tensor\\n{str(tensor_uuid)[:8]}..."
            if node.is_root:
                dot_lines.append(f'  "{tensor_str}" [label="{label}" shape=ellipse color=green];')
            else:
                dot_lines.append(f'  "{tensor_str}" [label="{label}" shape=ellipse];')
        
        # Add edges with operation labels
        for tensor_str, node in lineage.lineage_nodes.items():
            if node.operation_id:
                op_str = str(node.operation_id)
                if op_str in lineage.operation_records:
                    operation = lineage.operation_records[op_str]
                    for parent_id in node.parent_tensor_ids:
                        parent_str = str(parent_id)
                        dot_lines.append(f'  "{parent_str}" -> "{tensor_str}" [label="{operation.operation_type.value}"];')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)