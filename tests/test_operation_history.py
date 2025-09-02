#!/usr/bin/env python3
"""
Tests for operation history and lineage functionality.
"""

import pytest
import torch
import tempfile
import shutil
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path

from tensorus.operation_history import (
    OperationRecord, OperationInput, OperationOutput, OperationExecutionInfo,
    OperationType, OperationStatus, OperationHistory, TensorLineage, LineageNode
)
from tensorus.storage_ops_with_history import (
    StorageConnectedTensorOpsWithHistory, TensorStorageWithHistoryOps, TrackedOperationResult
)
from tensorus.tensor_storage import TensorStorage


class TestOperationHistory:
    """Test basic operation history functionality."""
    
    def test_operation_record_creation(self):
        """Test creating an operation record."""
        inputs = [
            OperationInput(
                tensor_id=uuid4(),
                shape=[3, 3],
                dtype="float32",
                device="cpu",
                parameter_name="matrix1",
                is_tensor=True
            )
        ]
        
        record = OperationRecord(
            operation_type=OperationType.ADD,
            operation_name="Element-wise Addition",
            inputs=inputs,
            parameters={"tensor_id1": "test_id"},
            user_id="test_user"
        )
        
        assert record.operation_type == OperationType.ADD
        assert record.operation_name == "Element-wise Addition"
        assert record.status == OperationStatus.STARTED
        assert len(record.inputs) == 1
        assert record.inputs[0].parameter_name == "matrix1"
        assert record.user_id == "test_user"
        assert record.duration_ms is None  # Not completed yet
    
    def test_operation_record_completion(self):
        """Test completing an operation record."""
        record = OperationRecord(
            operation_type=OperationType.MATMUL,
            operation_name="Matrix Multiplication",
            inputs=[],
            parameters={}
        )
        
        outputs = [
            OperationOutput(
                tensor_id=uuid4(),
                shape=[5, 5],
                dtype="float32",
                device="cpu"
            )
        ]
        
        exec_info = OperationExecutionInfo(
            execution_time_ms=150.5,
            memory_usage_mb=64.2,
            device="cpu"
        )
        
        record.mark_completed(outputs, exec_info)
        
        assert record.status == OperationStatus.COMPLETED
        assert record.completed_at is not None
        assert len(record.outputs) == 1
        assert record.execution_info is not None
        assert record.execution_info.execution_time_ms == 150.5
        assert record.duration_ms is not None
        assert record.duration_ms > 0
    
    def test_operation_record_failure(self):
        """Test marking an operation as failed."""
        record = OperationRecord(
            operation_type=OperationType.SVD,
            operation_name="Singular Value Decomposition",
            inputs=[],
            parameters={}
        )
        
        error_msg = "Matrix is not invertible"
        traceback_info = "Traceback (most recent call last):\n  File..."
        
        record.mark_failed(error_msg, traceback_info)
        
        assert record.status == OperationStatus.FAILED
        assert record.completed_at is not None
        assert record.error_message == error_msg
        assert record.error_traceback == traceback_info
        assert record.duration_ms is not None
    
    def test_operation_history_basic(self):
        """Test basic operation history functionality."""
        history = OperationHistory()
        
        # Create test operation
        tensor_id1 = uuid4()
        tensor_id2 = uuid4()
        output_id = uuid4()
        
        record = OperationRecord(
            operation_type=OperationType.ADD,
            operation_name="Addition",
            inputs=[],
            parameters={}
        )
        record.mark_completed([])
        
        # Record operation
        history.record_operation(record, [tensor_id1, tensor_id2], [output_id])
        
        assert history.total_operations == 1
        assert OperationType.ADD in history.operations_by_type
        assert history.operations_by_type[OperationType.ADD] == 1
        assert OperationStatus.COMPLETED in history.operations_by_status
        
        # Check lineage was created
        output_str = str(output_id)
        assert output_str in history.tensor_lineages
        lineage = history.tensor_lineages[output_str]
        assert output_str in lineage.lineage_nodes
        assert lineage.total_operations == 1
    
    def test_tensor_lineage_basic(self):
        """Test basic tensor lineage functionality."""
        lineage = TensorLineage(tensor_id=uuid4())
        
        # Create test operation and tensors
        input_id1 = uuid4()
        input_id2 = uuid4()
        output_id = uuid4()
        
        operation = OperationRecord(
            operation_type=OperationType.MULTIPLY,
            operation_name="Multiplication",
            inputs=[],
            parameters={}
        )
        operation.mark_completed([])
        
        # Add operation to lineage
        lineage.add_operation(operation, [input_id1, input_id2], [output_id])
        
        # Check lineage structure
        output_str = str(output_id)
        assert output_str in lineage.lineage_nodes
        
        node = lineage.lineage_nodes[output_str]
        assert node.tensor_id == output_id
        assert node.operation_id == operation.operation_id
        assert len(node.parent_tensor_ids) == 2
        assert input_id1 in node.parent_tensor_ids
        assert input_id2 in node.parent_tensor_ids
        assert not node.is_root
        assert node.depth == 1
        
        assert lineage.total_operations == 1
        assert lineage.max_depth == 1
    
    def test_lineage_ancestors_descendants(self):
        """Test getting ancestors and descendants in lineage."""
        lineage = TensorLineage(tensor_id=uuid4())
        
        # Create a chain: root -> intermediate -> final
        root_id = uuid4()
        intermediate_id = uuid4()
        final_id = uuid4()
        
        # First operation: root -> intermediate
        op1 = OperationRecord(
            operation_type=OperationType.RESHAPE,
            operation_name="Reshape",
            inputs=[], parameters={}
        )
        op1.mark_completed([])
        lineage.add_operation(op1, [root_id], [intermediate_id])
        
        # Second operation: intermediate -> final
        op2 = OperationRecord(
            operation_type=OperationType.TRANSPOSE,
            operation_name="Transpose",
            inputs=[], parameters={}
        )
        op2.mark_completed([])
        lineage.add_operation(op2, [intermediate_id], [final_id])
        
        # Test ancestors
        final_ancestors = lineage.get_ancestors(final_id)
        assert root_id in final_ancestors
        assert intermediate_id in final_ancestors
        assert len(final_ancestors) == 2
        
        intermediate_ancestors = lineage.get_ancestors(intermediate_id)
        assert root_id in intermediate_ancestors
        assert len(intermediate_ancestors) == 1
        
        # Test descendants
        root_descendants = lineage.get_descendants(root_id)
        assert intermediate_id in root_descendants
        assert final_id in root_descendants
        assert len(root_descendants) == 2
        
        intermediate_descendants = lineage.get_descendants(intermediate_id)
        assert final_id in intermediate_descendants
        assert len(intermediate_descendants) == 1


class TestStorageConnectedOpsWithHistory:
    """Test storage-connected operations with history tracking."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage_with_history(self, temp_storage_path):
        """Create storage with history tracking."""
        storage = TensorStorage(str(temp_storage_path))
        return TensorStorageWithHistoryOps(storage, enable_history=True, enable_lineage=True)
    
    def test_tracked_operation_result(self):
        """Test TrackedOperationResult creation."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        operation = OperationRecord(
            operation_type=OperationType.ADD,
            operation_name="Addition",
            inputs=[], parameters={}
        )
        
        result = TrackedOperationResult(
            tensor=tensor,
            operation="add",
            inputs=["tensor1", "tensor2"],
            computation_time=0.05,
            cached=False,
            operation_record=operation
        )
        
        assert torch.equal(result.tensor, tensor)
        assert result.operation == "add"
        assert len(result.inputs) == 2
        assert result.computation_time == 0.05
        assert not result.cached
        assert result.operation_record is operation
    
    def test_add_operation_with_history(self, storage_with_history):
        """Test add operation with history tracking."""
        # Create test dataset
        dataset_name = "test_dataset"
        storage_with_history.create_dataset(dataset_name)
        
        # Insert test tensors
        tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        
        id1 = storage_with_history.insert(dataset_name, tensor1, {"name": "tensor1"})
        id2 = storage_with_history.insert(dataset_name, tensor2, {"name": "tensor2"})
        
        # Perform add operation
        result_id = storage_with_history.tensor_add(dataset_name, id1, id2, user_id="test_user")
        
        # Check result tensor
        result_data = storage_with_history.get_tensor_by_id(dataset_name, result_id)
        expected = tensor1 + tensor2
        assert torch.allclose(result_data["tensor"], expected)
        
        # Check operation history
        tensor_ops = storage_with_history.get_tensor_operation_history(result_id)
        assert len(tensor_ops) >= 1
        
        add_ops = [op for op in tensor_ops if op.operation_type == OperationType.ADD]
        assert len(add_ops) >= 1
        
        add_op = add_ops[0]
        assert add_op.status == OperationStatus.COMPLETED
        assert add_op.user_id == "test_user"
        assert len(add_op.inputs) == 2
        assert len(add_op.outputs) >= 1
        
        # Check lineage
        lineage = storage_with_history.get_tensor_lineage(result_id)
        assert lineage is not None
        assert lineage.total_operations >= 1
        
        # The result tensor should have both input tensors as ancestors
        ancestors = lineage.get_ancestors(UUID(result_id))
        assert UUID(id1) in ancestors or UUID(id2) in ancestors
    
    def test_matmul_operation_with_history(self, storage_with_history):
        """Test matrix multiplication with history tracking."""
        dataset_name = "test_dataset"
        storage_with_history.create_dataset(dataset_name)
        
        # Create compatible matrices for multiplication
        matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        matrix2 = torch.tensor([[2.0, 0.0], [1.0, 3.0]])  # 2x2
        
        id1 = storage_with_history.insert(dataset_name, matrix1)
        id2 = storage_with_history.insert(dataset_name, matrix2)
        
        # Perform matrix multiplication
        result_id = storage_with_history.tensor_matmul(dataset_name, id1, id2, user_id="test_user")
        
        # Check result
        result_data = storage_with_history.get_tensor_by_id(dataset_name, result_id)
        expected = torch.matmul(matrix1, matrix2)
        assert torch.allclose(result_data["tensor"], expected)
        
        # Check operation history
        ops = storage_with_history.get_tensor_operation_history(result_id)
        matmul_ops = [op for op in ops if op.operation_type == OperationType.MATMUL]
        assert len(matmul_ops) >= 1
        
        matmul_op = matmul_ops[0]
        assert matmul_op.status == OperationStatus.COMPLETED
        assert matmul_op.execution_info is not None
        assert matmul_op.execution_info.execution_time_ms is not None
        assert matmul_op.execution_info.execution_time_ms > 0
    
    def test_operation_statistics(self, storage_with_history):
        """Test operation statistics collection."""
        dataset_name = "test_dataset"
        storage_with_history.create_dataset(dataset_name)
        
        # Perform several operations
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        
        id1 = storage_with_history.insert(dataset_name, tensor1)
        id2 = storage_with_history.insert(dataset_name, tensor2)
        
        # Add operation
        storage_with_history.tensor_add(dataset_name, id1, id2)
        
        # Add with scalar
        storage_with_history.ops.add(dataset_name, id1, 5.0)
        
        # Get statistics
        stats = storage_with_history.get_operation_stats()
        
        assert stats["total_operations"] >= 2
        assert stats["successful_operations"] >= 2
        assert stats["success_rate"] > 0
        assert "add" in stats["operations_by_type"]
        assert stats["operations_by_type"]["add"] >= 2
        assert stats["total_tensors_tracked"] >= 1
        assert "session_id" in stats
    
    def test_lineage_dot_export(self, storage_with_history):
        """Test DOT graph export functionality."""
        dataset_name = "test_dataset" 
        storage_with_history.create_dataset(dataset_name)
        
        # Create a chain of operations
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        id1 = storage_with_history.insert(dataset_name, tensor)
        
        # Reshape operation
        id2 = storage_with_history.ops.reshape(dataset_name, id1, (4,))
        
        # Add scalar
        id3 = storage_with_history.ops.add(dataset_name, id2, 1.0)
        
        # Export DOT graph
        dot_graph = storage_with_history.export_lineage_graph(id3)
        
        assert dot_graph != ""
        assert "digraph TensorLineage" in dot_graph
        assert "rankdir=TB" in dot_graph
        assert "reshape" in dot_graph.lower() or "add" in dot_graph.lower()
    
    def test_error_handling_in_operations(self, storage_with_history):
        """Test error handling and failed operation tracking."""
        dataset_name = "test_dataset"
        storage_with_history.create_dataset(dataset_name)
        
        # Create incompatible tensors for matrix multiplication
        tensor1 = torch.tensor([[1.0, 2.0]])  # 1x2
        tensor2 = torch.tensor([[1.0], [2.0], [3.0]])  # 3x1 - incompatible
        
        id1 = storage_with_history.insert(dataset_name, tensor1)
        id2 = storage_with_history.insert(dataset_name, tensor2)
        
        # This should fail
        with pytest.raises(Exception):
            storage_with_history.tensor_matmul(dataset_name, id1, id2)
        
        # Check that failed operation was recorded
        stats = storage_with_history.get_operation_stats()
        assert stats["failed_operations"] >= 1
    
    def test_caching_with_history(self, storage_with_history):
        """Test that caching works with history tracking."""
        dataset_name = "test_dataset"
        storage_with_history.create_dataset(dataset_name)
        
        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        
        id1 = storage_with_history.insert(dataset_name, tensor1)
        id2 = storage_with_history.insert(dataset_name, tensor2)
        
        # First operation - not cached
        result_id1 = storage_with_history.ops.add(dataset_name, id1, id2)
        
        # Second identical operation - should be cached
        result_id2 = storage_with_history.ops.add(dataset_name, id1, id2)
        
        # Results should be identical
        result1 = storage_with_history.get_tensor_by_id(dataset_name, result_id1)
        result2 = storage_with_history.get_tensor_by_id(dataset_name, result_id2)
        
        assert torch.equal(result1["tensor"], result2["tensor"])
        
        # Check cache stats
        cache_stats = storage_with_history.ops.get_cache_stats()
        assert cache_stats["enabled"]
        assert cache_stats["size"] > 0


class TestOperationValidation:
    """Test validation and edge cases for operation history."""
    
    def test_operation_input_validation(self):
        """Test validation of operation inputs."""
        # Test tensor input
        tensor_input = OperationInput(
            tensor_id=uuid4(),
            shape=[3, 3],
            dtype="float32",
            device="cpu",
            is_tensor=True
        )
        assert tensor_input.is_tensor
        assert tensor_input.value is None
        
        # Test scalar input
        scalar_input = OperationInput(
            tensor_id=None,
            shape=[],
            dtype="scalar",
            device="cpu",
            is_tensor=False,
            value=5.0
        )
        assert not scalar_input.is_tensor
        assert scalar_input.value == 5.0
    
    def test_execution_info_validation(self):
        """Test validation of execution information."""
        # Valid execution info
        exec_info = OperationExecutionInfo(
            execution_time_ms=100.5,
            memory_usage_mb=64.0,
            device="cuda:0"
        )
        assert exec_info.execution_time_ms == 100.5
        assert exec_info.memory_usage_mb == 64.0
        
        # Test negative values validation
        with pytest.raises(Exception):  # Should raise validation error
            OperationExecutionInfo(execution_time_ms=-10.0)
        
        with pytest.raises(Exception):  # Should raise validation error
            OperationExecutionInfo(memory_usage_mb=-5.0)
    
    def test_lineage_with_multiple_outputs(self):
        """Test lineage tracking for operations with multiple outputs (like SVD)."""
        lineage = TensorLineage(tensor_id=uuid4())
        
        input_id = uuid4()
        output_id1 = uuid4()  # U matrix
        output_id2 = uuid4()  # S values  
        output_id3 = uuid4()  # V^T matrix
        
        operation = OperationRecord(
            operation_type=OperationType.SVD,
            operation_name="SVD Decomposition",
            inputs=[], parameters={}
        )
        operation.mark_completed([])
        
        # Add operation with multiple outputs
        lineage.add_operation(operation, [input_id], [output_id1, output_id2, output_id3])
        
        # Check all outputs are in lineage
        for output_id in [output_id1, output_id2, output_id3]:
            output_str = str(output_id)
            assert output_str in lineage.lineage_nodes
            
            node = lineage.lineage_nodes[output_str]
            assert input_id in node.parent_tensor_ids
            assert node.operation_id == operation.operation_id
    
    def test_operation_history_filtering(self):
        """Test filtering operations in history."""
        history = OperationHistory()
        
        # Create operations of different types
        add_op = OperationRecord(
            operation_type=OperationType.ADD,
            operation_name="Addition",
            inputs=[], parameters={}
        )
        add_op.mark_completed([])
        
        mul_op = OperationRecord(
            operation_type=OperationType.MULTIPLY,
            operation_name="Multiplication", 
            inputs=[], parameters={}
        )
        mul_op.mark_completed([])
        
        failed_op = OperationRecord(
            operation_type=OperationType.SVD,
            operation_name="SVD",
            inputs=[], parameters={}
        )
        failed_op.mark_failed("Test error")
        
        # Record all operations
        dummy_id = uuid4()
        history.record_operation(add_op, [dummy_id], [uuid4()])
        history.record_operation(mul_op, [dummy_id], [uuid4()])
        history.record_operation(failed_op, [dummy_id], [])
        
        # Test filtering by type
        add_ops = history.get_operations_by_type(OperationType.ADD)
        assert len(add_ops) == 1
        assert add_ops[0].operation_type == OperationType.ADD
        
        # Test filtering by tensor
        tensor_ops = history.get_operations_by_tensor(dummy_id)
        assert len(tensor_ops) == 3  # All operations involved this tensor
        
        # Test recent operations limit
        recent_ops = history.get_recent_operations(limit=2)
        assert len(recent_ops) == 2  # Should return most recent 2