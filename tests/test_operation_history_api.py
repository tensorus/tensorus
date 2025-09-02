#!/usr/bin/env python3
"""
Tests for operation history and lineage API endpoints.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Mock the dependencies before importing the app
mock_storage = Mock()
mock_storage_with_history = Mock()

with patch('tensorus.api.dependencies.get_storage_instance', return_value=mock_storage):
    with patch('tensorus.api.dependencies.get_storage_with_history', return_value=mock_storage_with_history):
        from tensorus.api import app

from tensorus.operation_history import (
    OperationRecord, OperationType, OperationStatus, TensorLineage, OperationHistory
)
from tensorus.storage_ops_with_history import TensorStorageWithHistoryOps
from tensorus.tensor_storage import TensorStorage


class TestOperationHistoryAPI:
    """Test operation history API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def real_storage_with_history(self, temp_storage_path):
        """Create real storage with history for integration tests."""
        storage = TensorStorage(str(temp_storage_path))
        return TensorStorageWithHistoryOps(storage, enable_history=True, enable_lineage=True)
    
    def test_get_recent_operations(self, client):
        """Test getting recent operations."""
        # Mock operation records
        mock_operations = [
            Mock(
                operation_id="op1",
                operation_type=OperationType.ADD,
                operation_name="Addition",
                status=OperationStatus.COMPLETED,
                started_at="2023-01-01T12:00:00",
                completed_at="2023-01-01T12:00:01",
                duration_ms=1000.0,
                user_id="test_user",
                session_id="session1",
                inputs=[],
                outputs=[],
                execution_info=Mock(
                    execution_time_ms=1000.0,
                    memory_usage_mb=64.0,
                    device="cpu",
                    hostname="test-host"
                ),
                error_message=None,
                tags=["test"]
            )
        ]
        
        # Mock the storage dependency
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.ops.get_recent_operations.return_value = mock_operations
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get("/api/v1/operations/recent?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["operations"]) == 1
        
        operation = data["operations"][0]
        assert operation["operation_id"] == "op1"
        assert operation["operation_type"] == "add"
        assert operation["status"] == "completed"
        assert operation["user_id"] == "test_user"
    
    def test_get_recent_operations_with_filters(self, client):
        """Test getting recent operations with type filter."""
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.ops.get_recent_operations.return_value = []
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get("/api/v1/operations/recent?operation_type=add&status=completed")
        
        assert response.status_code == 200
    
    def test_get_recent_operations_invalid_filter(self, client):
        """Test getting recent operations with invalid filter."""
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get("/api/v1/operations/recent?operation_type=invalid_type")
        
        assert response.status_code == 400
        assert "Invalid operation type" in response.json()["detail"]
    
    def test_get_tensor_operations(self, client):
        """Test getting operations for a specific tensor."""
        tensor_id = "test_tensor_id"
        
        mock_operations = [
            Mock(
                operation_id="op1",
                operation_type=OperationType.RESHAPE,
                operation_name="Reshape",
                status=OperationStatus.COMPLETED,
                started_at="2023-01-01T12:00:00",
                completed_at="2023-01-01T12:00:01",
                duration_ms=500.0,
                user_id=None,
                session_id="session1",
                inputs=[],
                outputs=[],
                execution_info=None,
                error_message=None,
                tags=[]
            )
        ]
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.get_tensor_operation_history.return_value = mock_operations
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get(f"/api/v1/operations/tensor/{tensor_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["count"] == 1
        assert f"tensor {tensor_id}" in data["message"]
    
    def test_get_tensor_lineage(self, client):
        """Test getting tensor lineage."""
        tensor_id = "test_tensor_id"
        
        # Mock lineage data
        mock_lineage = Mock()
        mock_lineage.root_tensor_ids = ["root1", "root2"]
        mock_lineage.max_depth = 3
        mock_lineage.total_operations = 5
        mock_lineage.created_at = "2023-01-01T10:00:00"
        mock_lineage.last_updated = "2023-01-01T12:00:00"
        
        # Mock lineage nodes
        mock_lineage.lineage_nodes = {
            "tensor1": Mock(
                tensor_id="tensor1",
                operation_id="op1",
                parent_tensor_ids=["root1"],
                created_at="2023-01-01T11:00:00",
                depth=1,
                is_root=False,
                is_leaf=True
            )
        }
        
        # Mock operation records
        mock_lineage.operation_records = {
            "op1": Mock(
                operation_id="op1",
                operation_type=OperationType.ADD,
                operation_name="Addition",
                status=OperationStatus.COMPLETED,
                started_at="2023-01-01T11:00:00",
                completed_at="2023-01-01T11:00:01",
                duration_ms=1000.0,
                execution_info=Mock(
                    execution_time_ms=1000.0,
                    memory_usage_mb=32.0,
                    device="cpu",
                    hostname="test-host"
                )
            )
        }
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.get_tensor_lineage.return_value = mock_lineage
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get(f"/api/v1/lineage/tensor/{tensor_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["tensor_id"] == tensor_id
        assert data["max_depth"] == 3
        assert data["total_operations"] == 5
        assert len(data["lineage_nodes"]) == 1
        assert data["operations"] is not None
        assert len(data["operations"]) == 1
    
    def test_get_tensor_lineage_not_found(self, client):
        """Test getting lineage for non-existent tensor."""
        tensor_id = "nonexistent_tensor"
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.get_tensor_lineage.return_value = None
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get(f"/api/v1/lineage/tensor/{tensor_id}")
        
        assert response.status_code == 404
        assert "No lineage found" in response.json()["detail"]
    
    def test_get_tensor_lineage_dot(self, client):
        """Test getting tensor lineage in DOT format."""
        tensor_id = "test_tensor_id"
        dot_graph = """digraph TensorLineage {
  rankdir=TB;
  "tensor1" [label="Tensor\\ntensor1..." shape=ellipse];
  "tensor2" [label="Tensor\\ntensor2..." shape=ellipse];
  "tensor1" -> "tensor2" [label="add"];
}"""
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.export_lineage_graph.return_value = dot_graph
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get(f"/api/v1/lineage/tensor/{tensor_id}/dot")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["tensor_id"] == tensor_id
        assert "digraph TensorLineage" in data["dot_graph"]
    
    def test_get_lineage_path(self, client):
        """Test getting path between two tensors."""
        source_id = "source_tensor"
        target_id = "target_tensor"
        
        # Mock lineage with path
        mock_lineage = Mock()
        mock_path = Mock()
        mock_path.path_nodes = [source_id, "intermediate", target_id]
        mock_path.operations = ["op1", "op2"]
        mock_path.total_depth = 2
        
        mock_lineage.get_operation_path.return_value = mock_path
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.get_tensor_lineage.return_value = mock_lineage
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get(f"/api/v1/lineage/tensor/{source_id}/path/{target_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["path_exists"] is True
        assert data["source_tensor_id"] == source_id
        assert data["target_tensor_id"] == target_id
        assert len(data["path_nodes"]) == 3
        assert len(data["operations"]) == 2
        assert data["total_depth"] == 2
    
    def test_get_lineage_path_not_found(self, client):
        """Test getting path when no path exists."""
        source_id = "source_tensor"
        target_id = "target_tensor"
        
        mock_lineage = Mock()
        mock_lineage.get_operation_path.return_value = None
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.get_tensor_lineage.return_value = mock_lineage
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get(f"/api/v1/lineage/tensor/{source_id}/path/{target_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is False
        assert data["path_exists"] is False
    
    def test_get_operation_statistics(self, client):
        """Test getting operation statistics."""
        mock_stats = {
            "total_operations": 100,
            "successful_operations": 95,
            "failed_operations": 5,
            "success_rate": 0.95,
            "operations_by_type": {"add": 30, "multiply": 25, "matmul": 20},
            "average_execution_times_ms": {"add": 10.5, "multiply": 15.2, "matmul": 45.8},
            "total_tensors_tracked": 150,
            "session_id": "session123",
            "created_at": "2023-01-01T10:00:00",
            "last_updated": "2023-01-01T12:00:00"
        }
        
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            mock_storage = Mock()
            mock_storage.get_operation_stats.return_value = mock_stats
            mock_get_storage.return_value = mock_storage
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get("/api/v1/operations/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_operations"] == 100
        assert data["success_rate"] == 0.95
        assert data["operations_by_type"]["add"] == 30
        assert data["session_id"] == "session123"
    
    def test_get_operation_types(self, client):
        """Test getting available operation types."""
        with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
            response = client.get("/api/v1/operations/types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "operation_types" in data
        assert "add" in data["operation_types"]
        assert "multiply" in data["operation_types"]
        assert "matmul" in data["operation_types"]
        assert data["count"] > 0
    
    def test_get_operation_statuses(self, client):
        """Test getting available operation statuses."""
        with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
            response = client.get("/api/v1/operations/statuses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "operation_statuses" in data
        assert "started" in data["operation_statuses"]
        assert "completed" in data["operation_statuses"]
        assert "failed" in data["operation_statuses"]
        assert data["count"] > 0
    
    def test_api_authentication(self, client):
        """Test API authentication requirement."""
        # Test without API key
        response = client.get("/api/v1/operations/recent")
        assert response.status_code == 403  # Should require authentication
        
        # Test with invalid API key
        with patch('tensorus.api.routers.operations.verify_api_key', side_effect=Exception("Invalid API key")):
            response = client.get("/api/v1/operations/recent", headers={"Authorization": "Bearer invalid_key"})
            assert response.status_code == 500
    
    def test_error_handling(self, client):
        """Test error handling in API endpoints."""
        with patch('tensorus.api.routers.operations.get_storage_with_history') as mock_get_storage:
            # Mock storage that raises an exception
            mock_get_storage.side_effect = Exception("Storage error")
            
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                response = client.get("/api/v1/operations/recent")
        
        assert response.status_code == 500
        assert "Storage error" in response.json()["detail"]


class TestOperationHistoryAPIIntegration:
    """Integration tests with real storage."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_operation_tracking(self, client, temp_storage_path):
        """Test end-to-end operation tracking through API."""
        # Create real storage with history
        storage = TensorStorage(str(temp_storage_path))
        storage_with_history = TensorStorageWithHistoryOps(storage, enable_history=True, enable_lineage=True)
        
        # Create dataset and perform operations
        dataset_name = "test_dataset"
        storage_with_history.create_dataset(dataset_name)
        
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([4.0, 5.0, 6.0])
        
        id1 = storage_with_history.insert(dataset_name, tensor1)
        id2 = storage_with_history.insert(dataset_name, tensor2)
        
        # Perform add operation
        result_id = storage_with_history.tensor_add(dataset_name, id1, id2, user_id="integration_test")
        
        # Test API with real data
        with patch('tensorus.api.routers.operations.get_storage_with_history', return_value=storage_with_history):
            with patch('tensorus.api.routers.operations.verify_api_key', return_value="test_key"):
                # Test recent operations
                response = client.get("/api/v1/operations/recent?limit=10")
                assert response.status_code == 200
                
                data = response.json()
                assert data["success"] is True
                assert data["count"] >= 1
                
                # Find the add operation
                add_ops = [op for op in data["operations"] if op["operation_type"] == "add"]
                assert len(add_ops) >= 1
                
                add_op = add_ops[0]
                assert add_op["user_id"] == "integration_test"
                assert add_op["status"] == "completed"
                
                # Test tensor operations
                response = client.get(f"/api/v1/operations/tensor/{result_id}")
                assert response.status_code == 200
                
                data = response.json()
                assert data["success"] is True
                assert data["count"] >= 1
                
                # Test tensor lineage
                response = client.get(f"/api/v1/lineage/tensor/{result_id}")
                assert response.status_code == 200
                
                data = response.json()
                assert data["success"] is True
                assert data["tensor_id"] == result_id
                assert data["total_operations"] >= 1
                
                # Test DOT export
                response = client.get(f"/api/v1/lineage/tensor/{result_id}/dot")
                assert response.status_code == 200
                
                data = response.json()
                assert data["success"] is True
                assert "digraph TensorLineage" in data["dot_graph"]
                
                # Test statistics
                response = client.get("/api/v1/operations/statistics")
                assert response.status_code == 200
                
                data = response.json()
                assert data["total_operations"] >= 1
                assert data["successful_operations"] >= 1