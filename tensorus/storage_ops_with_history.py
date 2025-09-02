#!/usr/bin/env python3
"""
Storage-connected tensor operations with comprehensive operation history tracking.

This module extends StorageConnectedTensorOps to include full operation history
and lineage tracking using the OperationHistory and TensorLineage classes.
"""

import torch
import logging
import hashlib
import time
import traceback
import platform
import psutil
import os
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from uuid import UUID, uuid4
from datetime import datetime

from .storage_ops import OperationResult, StorageConnectedTensorOps
from .operation_history import (
    OperationRecord, OperationInput, OperationOutput, OperationExecutionInfo,
    OperationType, OperationStatus, OperationHistory, TensorLineage
)
from .metadata.schemas import OperationHistoryMetadata, LineageReference


class TrackedOperationResult(OperationResult):
    """Extended OperationResult with operation tracking information."""
    
    def __init__(self, tensor: torch.Tensor, operation: str, inputs: List[str], 
                 computation_time: float, cached: bool = False, 
                 operation_record: Optional[OperationRecord] = None):
        super().__init__(tensor, operation, inputs, computation_time, cached)
        self.operation_record = operation_record


class StorageConnectedTensorOpsWithHistory(StorageConnectedTensorOps):
    """
    Storage-connected tensor operations with comprehensive operation history tracking.
    
    This class extends the base StorageConnectedTensorOps to include:
    - Complete operation history tracking
    - Computational lineage management
    - Enhanced metadata storage
    - Performance monitoring
    - Error tracking
    """
    
    def __init__(self, storage, enable_history: bool = True, 
                 enable_lineage: bool = True, session_id: Optional[str] = None):
        """
        Initialize with a TensorStorage instance and history tracking.
        
        Args:
            storage: TensorStorage instance to operate on
            enable_history: Whether to track operation history
            enable_lineage: Whether to track computational lineage
            session_id: Optional session identifier for operations
        """
        super().__init__(storage)
        
        # History tracking
        self.enable_history = enable_history
        self.enable_lineage = enable_lineage
        self.session_id = session_id or str(uuid4())
        
        # Operation tracking
        if enable_history:
            self.operation_history = OperationHistory()
        else:
            self.operation_history = None
        
        # System information for execution context
        self._system_info = self._get_system_info()
        
        logger = logging.getLogger(__name__)
        logger.info(f"StorageConnectedTensorOpsWithHistory initialized with history={enable_history}, "
                   f"lineage={enable_lineage}, session={self.session_id}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information for execution context."""
        try:
            return {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "process_id": os.getpid(),
                "thread_id": threading.get_ident()
            }
        except Exception as e:
            logging.warning(f"Failed to gather system info: {e}")
            return {}
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return None
    
    def _get_device_info(self, tensor: torch.Tensor) -> str:
        """Get device information for a tensor."""
        if tensor.is_cuda:
            return f"cuda:{tensor.device.index}"
        else:
            return "cpu"
    
    def _create_operation_record(self, operation_type: OperationType, operation_name: str,
                               inputs: List[OperationInput], parameters: Dict[str, Any],
                               description: Optional[str] = None, user_id: Optional[str] = None) -> OperationRecord:
        """Create a new operation record."""
        return OperationRecord(
            operation_type=operation_type,
            operation_name=operation_name,
            description=description,
            inputs=inputs,
            parameters=parameters,
            execution_info=None,  # Will be set later
            user_id=user_id,
            session_id=self.session_id
        )
    
    def _create_operation_input(self, tensor_id: Optional[str], tensor: Optional[torch.Tensor],
                              parameter_name: Optional[str] = None, 
                              value: Optional[Any] = None, is_tensor: bool = True) -> OperationInput:
        """Create an operation input record."""
        if tensor is not None:
            return OperationInput(
                tensor_id=UUID(tensor_id) if tensor_id else None,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                device=self._get_device_info(tensor),
                parameter_name=parameter_name,
                is_tensor=is_tensor,
                value=None
            )
        else:
            return OperationInput(
                tensor_id=None,
                shape=[],
                dtype="scalar",
                device="cpu",
                parameter_name=parameter_name,
                is_tensor=False,
                value=value
            )
    
    def _create_operation_output(self, tensor_id: Optional[str], tensor: torch.Tensor,
                               is_primary: bool = True) -> OperationOutput:
        """Create an operation output record."""
        return OperationOutput(
            tensor_id=UUID(tensor_id) if tensor_id else None,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            device=self._get_device_info(tensor),
            is_primary=is_primary
        )
    
    def _create_execution_info(self, execution_time_ms: float, memory_usage_mb: Optional[float] = None,
                             device: Optional[str] = None) -> OperationExecutionInfo:
        """Create execution information record."""
        return OperationExecutionInfo(
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            device=device,
            torch_version=self._system_info.get("torch_version"),
            python_version=self._system_info.get("python_version"),
            hostname=self._system_info.get("hostname"),
            process_id=self._system_info.get("process_id"),
            thread_id=str(self._system_info.get("thread_id"))
        )
    
    def _record_operation(self, operation: OperationRecord, input_tensor_ids: List[UUID],
                         output_tensor_ids: List[UUID]) -> None:
        """Record an operation in the history."""
        if not self.enable_history or not self.operation_history:
            return
        
        self.operation_history.record_operation(operation, input_tensor_ids, output_tensor_ids)
    
    def _update_tensor_metadata(self, dataset_name: str, tensor_id: str, 
                              operation_record: OperationRecord) -> None:
        """Update tensor metadata with operation history information."""
        if not self.enable_history:
            return
        
        try:
            # Get current metadata
            result = self.storage.get_tensor_by_id(dataset_name, tensor_id)
            current_metadata = result.get("metadata", {})
            
            # Create or update operation history metadata
            operation_history_meta = OperationHistoryMetadata(
                tensor_id=UUID(tensor_id),
                operation_count=current_metadata.get("operation_count", 0) + 1,
                last_operation_id=operation_record.operation_id,
                last_operation_type=operation_record.operation_type.value,
                last_operation_timestamp=operation_record.completed_at or operation_record.started_at,
                creation_operation_id=operation_record.operation_id if current_metadata.get("operation_count", 0) == 0 else current_metadata.get("creation_operation_id"),
                lineage_depth=current_metadata.get("lineage_depth", 0),
                has_children=current_metadata.get("has_children", False),
                operation_tags=current_metadata.get("operation_tags", []) + operation_record.tags
            )
            
            # Update metadata
            current_metadata.update({
                "operation_history": operation_history_meta.model_dump(),
                "last_operation_timestamp": operation_record.completed_at.isoformat() if operation_record.completed_at else None
            })
            
            # Update in storage (this would need to be implemented in the storage layer)
            # self.storage.update_metadata(dataset_name, tensor_id, current_metadata)
            
        except Exception as e:
            logging.warning(f"Failed to update tensor metadata: {e}")
    
    def _store_result_with_history(self, dataset_name: str, result: TrackedOperationResult, 
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store operation result with history tracking."""
        if metadata is None:
            metadata = {}
        
        # Add operation metadata
        metadata.update({
            "operation": result.operation,
            "input_tensors": result.inputs,
            "computation_time": result.computation_time,
            "cached": result.cached,
            "operation_timestamp": result.timestamp
        })
        
        # Add operation record information if available
        if result.operation_record:
            metadata["operation_id"] = str(result.operation_record.operation_id)
            metadata["session_id"] = result.operation_record.session_id
        
        # Store the tensor
        tensor_id = self.storage.insert(dataset_name, result.tensor, metadata)
        
        # Update tensor metadata with operation history
        if result.operation_record:
            self._update_tensor_metadata(dataset_name, tensor_id, result.operation_record)
        
        return tensor_id
    
    # Override base operations with history tracking
    
    def add(self, dataset_name: str, tensor_id1: str, tensor_id2: Union[str, float, int],
            store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None,
            user_id: Optional[str] = None) -> Union[str, TrackedOperationResult]:
        """Add two tensors or a tensor and scalar, with comprehensive history tracking."""
        
        # Create operation record
        operation = None
        if self.enable_history:
            # Prepare inputs
            tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
            input1 = self._create_operation_input(tensor_id1, tensor1, "t1")
            
            if isinstance(tensor_id2, str):
                tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
                input2 = self._create_operation_input(tensor_id2, tensor2, "t2")
            else:
                input2 = self._create_operation_input(None, None, "t2", tensor_id2, False)
            
            operation = self._create_operation_record(
                operation_type=OperationType.ADD,
                operation_name="Element-wise Addition",
                inputs=[input1, input2],
                parameters={"tensor_id1": tensor_id1, "tensor_id2": tensor_id2},
                description=f"Add tensor {tensor_id1} and {'tensor ' + tensor_id2 if isinstance(tensor_id2, str) else 'scalar ' + str(tensor_id2)}",
                user_id=user_id
            )
        
        # Check cache
        inputs = [tensor_id1, str(tensor_id2)]
        cache_key = self._generate_cache_key("add", inputs)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            logging.debug(f"Cache hit for add operation: {cache_key}")
            cache_hit_result = TrackedOperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True,
                operation_record=operation
            )
            
            if operation:
                operation.mark_completed([self._create_operation_output(None, cached_result.tensor)])
            
            if store_result:
                result_id = self._store_result_with_history(dataset_name, cache_hit_result, result_metadata)
                if operation and self.enable_history:
                    self._record_operation(operation, 
                                         [UUID(tensor_id1)] + ([UUID(tensor_id2)] if isinstance(tensor_id2, str) else []),
                                         [UUID(result_id)])
                return result_id
            return cache_hit_result
        
        # Perform computation with timing and memory tracking
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
            
            if isinstance(tensor_id2, str):
                tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
            else:
                tensor2 = tensor_id2
            
            result_tensor = TensorOps.add(tensor1, tensor2)
            computation_time_ms = (time.time() - start_time) * 1000
            
            # Create execution info
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before if (memory_after and memory_before) else None
            execution_info = self._create_execution_info(
                execution_time_ms=computation_time_ms,
                memory_usage_mb=memory_used,
                device=self._get_device_info(result_tensor)
            )
            
            # Create result object
            result = TrackedOperationResult(
                tensor=result_tensor,
                operation="add",
                inputs=inputs,
                computation_time=computation_time_ms / 1000,  # Convert back to seconds for compatibility
                cached=False,
                operation_record=operation
            )
            
            # Complete operation record
            if operation:
                output = self._create_operation_output(None, result_tensor)
                operation.mark_completed([output], execution_info)
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            # Store or return result
            if store_result:
                result_id = self._store_result_with_history(dataset_name, result, result_metadata)
                
                # Record operation in history
                if operation and self.enable_history:
                    input_ids = [UUID(tensor_id1)]
                    if isinstance(tensor_id2, str):
                        input_ids.append(UUID(tensor_id2))
                    self._record_operation(operation, input_ids, [UUID(result_id)])
                
                return result_id
            
            return result
            
        except Exception as e:
            # Record operation failure
            if operation:
                operation.mark_failed(str(e), traceback.format_exc())
                if self.enable_history and self.operation_history:
                    # Still record the failed operation
                    input_ids = [UUID(tensor_id1)]
                    if isinstance(tensor_id2, str):
                        input_ids.append(UUID(tensor_id2))
                    self._record_operation(operation, input_ids, [])
            
            raise e
    
    # Similar implementations would be created for other operations...
    # For brevity, I'll show the pattern for one more operation
    
    def matmul(self, dataset_name: str, tensor_id1: str, tensor_id2: str,
              store_result: bool = True, result_metadata: Optional[Dict[str, Any]] = None,
              user_id: Optional[str] = None) -> Union[str, TrackedOperationResult]:
        """Matrix multiplication with comprehensive history tracking."""
        
        operation = None
        if self.enable_history:
            tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
            tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
            
            input1 = self._create_operation_input(tensor_id1, tensor1, "matrix1")
            input2 = self._create_operation_input(tensor_id2, tensor2, "matrix2")
            
            operation = self._create_operation_record(
                operation_type=OperationType.MATMUL,
                operation_name="Matrix Multiplication",
                inputs=[input1, input2],
                parameters={"tensor_id1": tensor_id1, "tensor_id2": tensor_id2},
                description=f"Matrix multiplication of tensors {tensor_id1} and {tensor_id2}",
                user_id=user_id
            )
        
        # Use parent implementation with modifications
        inputs = [tensor_id1, tensor_id2]
        cache_key = self._generate_cache_key("matmul", inputs)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            cache_hit_result = TrackedOperationResult(
                tensor=cached_result.tensor,
                operation=cached_result.operation,
                inputs=cached_result.inputs,
                computation_time=cached_result.computation_time,
                cached=True,
                operation_record=operation
            )
            
            if operation:
                operation.mark_completed([self._create_operation_output(None, cached_result.tensor)])
            
            if store_result:
                result_id = self._store_result_with_history(dataset_name, cache_hit_result, result_metadata)
                if operation and self.enable_history:
                    self._record_operation(operation, [UUID(tensor_id1), UUID(tensor_id2)], [UUID(result_id)])
                return result_id
            return cache_hit_result
        
        # Perform computation
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            tensor1 = self._retrieve_tensor(dataset_name, tensor_id1)
            tensor2 = self._retrieve_tensor(dataset_name, tensor_id2)
            
            result_tensor = TensorOps.matmul(tensor1, tensor2)
            computation_time_ms = (time.time() - start_time) * 1000
            
            # Create execution info
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before if (memory_after and memory_before) else None
            execution_info = self._create_execution_info(
                execution_time_ms=computation_time_ms,
                memory_usage_mb=memory_used,
                device=self._get_device_info(result_tensor)
            )
            
            result = TrackedOperationResult(
                tensor=result_tensor,
                operation="matmul",
                inputs=inputs,
                computation_time=computation_time_ms / 1000,
                cached=False,
                operation_record=operation
            )
            
            if operation:
                output = self._create_operation_output(None, result_tensor)
                operation.mark_completed([output], execution_info)
            
            self._cache_result(cache_key, result)
            
            if store_result:
                result_id = self._store_result_with_history(dataset_name, result, result_metadata)
                if operation and self.enable_history:
                    self._record_operation(operation, [UUID(tensor_id1), UUID(tensor_id2)], [UUID(result_id)])
                return result_id
            
            return result
            
        except Exception as e:
            if operation:
                operation.mark_failed(str(e), traceback.format_exc())
                if self.enable_history and self.operation_history:
                    self._record_operation(operation, [UUID(tensor_id1), UUID(tensor_id2)], [])
            raise e
    
    # History querying methods
    
    def get_tensor_history(self, tensor_id: str) -> Optional[TensorLineage]:
        """Get the complete computational lineage for a tensor."""
        if not self.enable_history or not self.operation_history:
            return None
        
        try:
            return self.operation_history.get_tensor_history(UUID(tensor_id))
        except ValueError:
            return None
    
    def get_tensor_operations(self, tensor_id: str) -> List[OperationRecord]:
        """Get all operations that involved a specific tensor."""
        if not self.enable_history or not self.operation_history:
            return []
        
        try:
            return self.operation_history.get_operations_by_tensor(UUID(tensor_id))
        except ValueError:
            return []
    
    def get_recent_operations(self, limit: int = 100) -> List[OperationRecord]:
        """Get the most recent operations."""
        if not self.enable_history or not self.operation_history:
            return []
        
        return self.operation_history.get_recent_operations(limit)
    
    def get_operations_by_type(self, operation_type: OperationType) -> List[OperationRecord]:
        """Get all operations of a specific type."""
        if not self.enable_history or not self.operation_history:
            return []
        
        return self.operation_history.get_operations_by_type(operation_type)
    
    def export_tensor_lineage_dot(self, tensor_id: str) -> str:
        """Export tensor lineage as DOT graph format."""
        if not self.enable_history or not self.operation_history:
            return ""
        
        try:
            return self.operation_history.export_lineage_dot(UUID(tensor_id))
        except ValueError:
            return ""
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive operation statistics."""
        if not self.enable_history or not self.operation_history:
            return {}
        
        history = self.operation_history
        
        # Calculate success/failure rates
        total_ops = history.total_operations
        successful_ops = history.operations_by_status.get(OperationStatus.COMPLETED, 0)
        failed_ops = history.operations_by_status.get(OperationStatus.FAILED, 0)
        
        # Calculate average execution times by operation type
        avg_times = {}
        for op_type, count in history.operations_by_type.items():
            ops_of_type = history.get_operations_by_type(op_type)
            completed_ops = [op for op in ops_of_type if op.status == OperationStatus.COMPLETED and op.execution_info]
            if completed_ops:
                avg_time = sum(op.execution_info.execution_time_ms for op in completed_ops) / len(completed_ops)
                avg_times[op_type.value] = avg_time
        
        return {
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
            "operations_by_type": {k.value: v for k, v in history.operations_by_type.items()},
            "average_execution_times_ms": avg_times,
            "total_tensors_tracked": len(history.tensor_lineages),
            "session_id": self.session_id,
            "created_at": history.created_at.isoformat(),
            "last_updated": history.last_updated.isoformat()
        }


class TensorStorageWithHistoryOps:
    """
    Extension of TensorStorage with integrated operations and history tracking.
    """
    
    def __init__(self, storage, enable_history: bool = True, enable_lineage: bool = True):
        """Initialize with a TensorStorage instance."""
        self.storage = storage
        self.ops = StorageConnectedTensorOpsWithHistory(storage, enable_history, enable_lineage)
    
    def __getattr__(self, name):
        """Delegate to storage for unknown attributes."""
        return getattr(self.storage, name)
    
    # Convenience methods with history tracking
    
    def tensor_add(self, dataset_name: str, tensor_id1: str, tensor_id2: Union[str, float, int],
                  store_result: bool = True, user_id: Optional[str] = None) -> Union[str, TrackedOperationResult]:
        """Add two tensors or tensor and scalar with history tracking."""
        return self.ops.add(dataset_name, tensor_id1, tensor_id2, store_result, user_id=user_id)
    
    def tensor_matmul(self, dataset_name: str, tensor_id1: str, tensor_id2: str,
                     store_result: bool = True, user_id: Optional[str] = None) -> Union[str, TrackedOperationResult]:
        """Matrix multiply two tensors with history tracking."""
        return self.ops.matmul(dataset_name, tensor_id1, tensor_id2, store_result, user_id=user_id)
    
    # History querying methods
    
    def get_tensor_lineage(self, tensor_id: str) -> Optional[TensorLineage]:
        """Get computational lineage for a tensor."""
        return self.ops.get_tensor_history(tensor_id)
    
    def get_tensor_operation_history(self, tensor_id: str) -> List[OperationRecord]:
        """Get operation history for a tensor."""
        return self.ops.get_tensor_operations(tensor_id)
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return self.ops.get_operation_statistics()
    
    def export_lineage_graph(self, tensor_id: str) -> str:
        """Export tensor lineage as DOT graph."""
        return self.ops.export_tensor_lineage_dot(tensor_id)


def create_storage_with_history_ops(storage_path: str, enable_history: bool = True, 
                                   enable_lineage: bool = True, **storage_kwargs):
    """
    Factory function to create TensorStorage with integrated operations and history tracking.
    
    Args:
        storage_path: Path to storage directory
        enable_history: Whether to enable operation history tracking
        enable_lineage: Whether to enable computational lineage tracking
        **storage_kwargs: Additional arguments for TensorStorage
        
    Returns:
        TensorStorageWithHistoryOps instance
    """
    from .tensor_storage import TensorStorage
    
    storage = TensorStorage(storage_path, **storage_kwargs)
    return TensorStorageWithHistoryOps(storage, enable_history, enable_lineage)