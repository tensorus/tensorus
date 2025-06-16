import pytest
from fastapi.testclient import TestClient
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import json
from unittest.mock import patch, call, MagicMock  # For spying on audit log
from typing import Dict

from tensorus.api import app
from tensorus.metadata.storage import InMemoryStorage
from tensorus.metadata.storage_abc import MetadataStorage
from tensorus.metadata.schemas import (
    TensorDescriptor, SemanticMetadata, LineageMetadata, DataType, StorageFormat,
    ComputationalMetadata, QualityMetadata, RelationalMetadata, UsageMetadata # Added others
)
from tensorus.metadata.schemas_iodata import TensorusExportData, TensorusExportEntry
from tensorus.api.dependencies import get_storage_instance
from tensorus.config import settings as global_settings
from tensorus.metadata import storage_instance as global_app_storage_instance

# --- Fixtures ---
# (Re-using client_with_clean_storage and API key fixtures from test_io_api.py,
# ensure they are appropriate or redefine if needed. For now, assume they are suitable.)

API_KEY_FOR_MAIN_API_TESTS = "api_key_for_main_tests"

@pytest.fixture(autouse=True)
def setup_api_test_keys(monkeypatch):
    monkeypatch.setattr(global_settings, 'VALID_API_KEYS', [API_KEY_FOR_MAIN_API_TESTS, "test_io_key_for_real"]) # Add key for these tests
    monkeypatch.setattr(global_settings, 'API_KEY_HEADER_NAME', "X-API-KEY")
    from tensorus.api.security import api_key_header_auth as global_api_key_header_auth  # Re-import or patch instance
    monkeypatch.setattr(global_api_key_header_auth.model, 'name', "X-API-KEY")


@pytest.fixture(scope="function")
def client_with_clean_storage(monkeypatch):
    monkeypatch.setattr(global_settings, "STORAGE_BACKEND", "in_memory")
    # Ensure the global_app_storage_instance IS an InMemoryStorage and clear it
    if not isinstance(global_app_storage_instance, InMemoryStorage):
         pytest.skip("Skipping API tests: Requires InMemoryStorage for clean state.")
    global_app_storage_instance.clear_all_data()

    with TestClient(app) as c:
        yield c
    global_app_storage_instance.clear_all_data()

# Helper to create TD via API for setting up tests
def _create_td_via_api(client: TestClient, headers: Dict, **overrides) -> Dict:
    payload = { # Minimal valid payload
        "tensor_id": str(overrides.get("tensor_id", uuid4())),
        "dimensionality": overrides.get("dimensionality", 1),
        "shape": overrides.get("shape", [1]),
        "data_type": overrides.get("data_type", DataType.FLOAT32.value),
        "owner": overrides.get("owner", "test_audit_user"),
        "byte_size": overrides.get("byte_size", 4)
    }
    payload.update(overrides)
    response = client.post("/tensor_descriptors/", json=payload, headers=headers)
    assert response.status_code == 201, f"Failed to create TD: {response.text}"
    return response.json()

# --- Audit Logging Tests (Integrated into existing API tests where applicable) ---

@patch('tensorus.api.endpoints.log_audit_event') # Path to where log_audit_event is defined
def test_create_tensor_descriptor_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_id = uuid4()
    payload = {
        "tensor_id": str(td_id), "dimensionality": 1, "shape": [10],
        "data_type": "float32", "owner": "audit_log_user", "byte_size": 40
    }
    response = client_with_clean_storage.post("/tensor_descriptors/", json=payload, headers=headers)
    assert response.status_code == 201

    mock_log_audit_event.assert_called_with(
        action="CREATE_TENSOR_DESCRIPTOR",
        user=API_KEY_FOR_MAIN_API_TESTS,
        tensor_id=str(td_id),
        details={"owner": "audit_log_user", "data_type": "float32"}
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_update_tensor_descriptor_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers, owner="original_owner_audit") # Use helper
    td_id_str = td_data["tensor_id"]

    update_payload = {"owner": "updated_owner_audit", "tags": ["audited"]}
    response = client_with_clean_storage.put(f"/tensor_descriptors/{td_id_str}", json=update_payload, headers=headers)
    assert response.status_code == 200

    mock_log_audit_event.assert_called_with(
        "UPDATE_TENSOR_DESCRIPTOR",
        API_KEY_FOR_MAIN_API_TESTS,
        td_id_str,
        {"updated_fields": ["owner", "tags"]}
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_delete_tensor_descriptor_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers, owner="delete_audit_user")
    td_id_str = td_data["tensor_id"]

    response = client_with_clean_storage.delete(f"/tensor_descriptors/{td_id_str}", headers=headers)
    assert response.status_code == 200

    mock_log_audit_event.assert_called_with(
        "DELETE_TENSOR_DESCRIPTOR",
        API_KEY_FOR_MAIN_API_TESTS,
        td_id_str
    )

# Example for an extended metadata type: Lineage
@patch('tensorus.api.endpoints.log_audit_event')
def test_upsert_lineage_metadata_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers)
    td_id = UUID(td_data["tensor_id"])

    lineage_payload = {"tensor_id": str(td_id), "version": "vAudit"}
    response = client_with_clean_storage.post(f"/tensor_descriptors/{td_id}/lineage", json=lineage_payload, headers=headers)
    assert response.status_code == 201

    mock_log_audit_event.assert_called_with(
        "UPSERT_Lineage_METADATA",
        API_KEY_FOR_MAIN_API_TESTS,
        str(td_id)
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_patch_lineage_metadata_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers)
    td_id = UUID(td_data["tensor_id"])

    # First, create lineage metadata to patch
    lineage_payload = {"tensor_id": str(td_id), "version": "vInitialAudit"}
    client_with_clean_storage.post(f"/tensor_descriptors/{td_id}/lineage", json=lineage_payload, headers=headers)
    mock_log_audit_event.reset_mock() # Reset mock after initial creation

    patch_payload = {"version": "vPatchedAudit", "provenance": {"notes": "audited patch"}}
    response = client_with_clean_storage.patch(f"/tensor_descriptors/{td_id}/lineage", json=patch_payload, headers=headers)
    assert response.status_code == 200

    mock_log_audit_event.assert_called_with(
        "PATCH_Lineage_METADATA",
        API_KEY_FOR_MAIN_API_TESTS,
        str(td_id),
        {"updated_fields": list(patch_payload.keys())}
    )


@patch('tensorus.api.endpoints.log_audit_event')
def test_delete_lineage_metadata_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers)
    td_id = UUID(td_data["tensor_id"])

    lineage_payload = {"tensor_id": str(td_id), "version": "vToDeleteAudit"}
    client_with_clean_storage.post(f"/tensor_descriptors/{td_id}/lineage", json=lineage_payload, headers=headers)
    mock_log_audit_event.reset_mock()

    response = client_with_clean_storage.delete(f"/tensor_descriptors/{td_id}/lineage", headers=headers)
    assert response.status_code == 204

    mock_log_audit_event.assert_called_with(
        "DELETE_Lineage_METADATA",
        API_KEY_FOR_MAIN_API_TESTS,
        str(td_id)
    )

# Test import audit log
@patch('tensorus.api.endpoints.log_audit_event')
def test_import_data_audit_log(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_id = uuid4()
    entry = TensorusExportEntry(
        tensor_descriptor=TensorDescriptor(tensor_id=td_id, dimensionality=0, shape=[], data_type=DataType.BOOLEAN, owner="audit_import", byte_size=0)
    )
    import_payload = TensorusExportData(entries=[entry])

    response = client_with_clean_storage.post("/tensors/import?conflict_strategy=skip", json=import_payload.model_dump(mode="json"), headers=headers)
    assert response.status_code == 200
    summary = response.json()

    mock_log_audit_event.assert_called_with(
        "IMPORT_DATA",
        API_KEY_FOR_MAIN_API_TESTS,
        details={"strategy": "skip", "summary": summary}
    )

# --- Placeholder for other tests from test_api.py (ensure they are still here and pass) ---
# E.g., advanced query tests, specific versioning/lineage logic tests, semantic metadata tests, etc.
# For this subtask, the focus is on adding audit log verification.
# The existing test structure of test_api.py would be preserved, and audit checks added to relevant tests.

# Example: Abridged version of a test from previous phase, now with audit log check
@patch('tensorus.api.endpoints.log_audit_event')
def test_create_semantic_metadata_audit(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers, owner="semantic_audit_user")
    td_id = UUID(td_data["tensor_id"])
    mock_log_audit_event.reset_mock() # Reset from TD creation

    sm_payload = {"tensor_id": str(td_id), "name": "audit_sem_meta", "description": "testing audit"}
    # Assuming semantic metadata is now nested under /tensor_descriptors/{tensor_id}/semantic
    response = client_with_clean_storage.post(f"/tensor_descriptors/{td_id}/semantic/", json=sm_payload, headers=headers)
    assert response.status_code == 201

    mock_log_audit_event.assert_called_with(
        "CREATE_SEMANTIC_METADATA",
        API_KEY_FOR_MAIN_API_TESTS,
        str(td_id),
        {"name": "audit_sem_meta"}
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_update_semantic_metadata_audit(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers)
    td_id = UUID(td_data["tensor_id"])

    # Create an initial semantic metadata entry
    sm_name = "initial_semantic_name"
    sm_payload_initial = {"tensor_id": str(td_id), "name": sm_name, "description": "Initial description"}
    client_with_clean_storage.post(f"/tensor_descriptors/{td_id}/semantic/", json=sm_payload_initial, headers=headers)
    mock_log_audit_event.reset_mock()

    update_payload = {"description": "Updated description for audit"}
    response = client_with_clean_storage.put(f"/tensor_descriptors/{td_id}/semantic/{sm_name}", json=update_payload, headers=headers)
    assert response.status_code == 200

    mock_log_audit_event.assert_called_with(
        "UPDATE_SEMANTIC_METADATA",
        API_KEY_FOR_MAIN_API_TESTS,
        str(td_id),
        {"original_name": sm_name, "updated_fields": update_payload}
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_delete_semantic_metadata_audit(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    td_data = _create_td_via_api(client_with_clean_storage, headers=headers)
    td_id = UUID(td_data["tensor_id"])
    sm_name = "to_delete_semantic_audit"
    sm_payload = {"tensor_id": str(td_id), "name": sm_name, "description": "Semantic to be deleted"}
    client_with_clean_storage.post(f"/tensor_descriptors/{td_id}/semantic/", json=sm_payload, headers=headers)
    mock_log_audit_event.reset_mock()

    response = client_with_clean_storage.delete(f"/tensor_descriptors/{td_id}/semantic/{sm_name}", headers=headers)
    assert response.status_code == 204 # No content for successful delete

    mock_log_audit_event.assert_called_with(
        "DELETE_SEMANTIC_METADATA",
        API_KEY_FOR_MAIN_API_TESTS,
        str(td_id),
        {"name": sm_name}
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_create_tensor_version_audit(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    parent_td_data = _create_td_via_api(client_with_clean_storage, headers=headers, owner="version_parent_audit")
    parent_id_str = parent_td_data["tensor_id"]
    mock_log_audit_event.reset_mock() # Reset after parent TD creation

    version_payload = {"new_version_string": "vAuditTest"}
    response = client_with_clean_storage.post(f"/tensors/{parent_id_str}/versions", json=version_payload, headers=headers)
    assert response.status_code == 201
    new_version_data = response.json()
    new_version_id = new_version_data["tensor_id"]

    mock_log_audit_event.assert_called_with(
        "CREATE_TENSOR_VERSION",
        API_KEY_FOR_MAIN_API_TESTS,
        new_version_id,
        {
            "parent_id": parent_id_str,
            "version": "vAuditTest"
        }
    )

@patch('tensorus.api.endpoints.log_audit_event')
def test_create_lineage_relationship_audit(mock_log_audit_event: MagicMock, client_with_clean_storage: TestClient):
    headers = {"X-API-KEY": API_KEY_FOR_MAIN_API_TESTS}
    source_td_data = _create_td_via_api(client_with_clean_storage, headers=headers, owner="src_lineage_audit")
    target_td_data = _create_td_via_api(client_with_clean_storage, headers=headers, owner="tgt_lineage_audit")
    source_id_str = source_td_data["tensor_id"]
    target_id_str = target_td_data["tensor_id"]
    mock_log_audit_event.reset_mock() # Reset after TD creations

    relationship_payload = {
        "source_tensor_id": source_id_str,
        "target_tensor_id": target_id_str,
        "relationship_type": "derived_for_audit"
    }
    response = client_with_clean_storage.post("/lineage/relationships/", json=relationship_payload, headers=headers)
    assert response.status_code == 201

    # Details in log should match relationship_payload after being dict()
    expected_details = {
        "source_tensor_id": UUID(source_id_str),
        "target_tensor_id": UUID(target_id_str),
        "relationship_type": "derived_for_audit",
        "details": None
    }
    mock_log_audit_event.assert_called_with(
        "CREATE_LINEAGE_RELATIONSHIP",
        API_KEY_FOR_MAIN_API_TESTS,
        target_id_str,
        details=expected_details
    )

