import pytest
from fastapi.testclient import TestClient
from uuid import uuid4, UUID
from datetime import datetime
import json

from tensorus.api import app
from tensorus.metadata.storage import InMemoryStorage # To directly interact for setup/verification
from tensorus.metadata.storage_abc import MetadataStorage
from tensorus.metadata.schemas import TensorDescriptor, SemanticMetadata, LineageMetadata, DataType
from tensorus.metadata.schemas_iodata import TensorusExportData, TensorusExportEntry
from tensorus.api.dependencies import get_storage_instance # To override if needed, or use global
from tensorus.config import settings as global_settings # For API key settings
from tensorus.metadata import storage_instance as global_app_storage_instance # The actual global instance

# --- Fixtures ---

@pytest.fixture(scope="function")
def client_with_clean_storage(monkeypatch):
    """
    Provides a TestClient with a fresh InMemoryStorage instance for each test.
    This ensures test isolation for I/O operations.
    It works by ensuring the global `storage_instance` (used by `get_storage_instance` dependency)
    is an InMemoryStorage and is cleared.
    """
    # Ensure global settings point to in_memory for these tests
    monkeypatch.setattr(global_settings, "STORAGE_BACKEND", "in_memory")

    # Assert that the globally configured instance is indeed InMemoryStorage
    # (This will be true if tensorus.metadata was imported after settings were patched,
    # or if the default was already in_memory and not changed by other tests' env vars)
    # For robust testing, one might re-import tensorus.metadata or use app.dependency_overrides
    if not isinstance(global_app_storage_instance, InMemoryStorage):
        # This indicates a test setup issue or interference if another test changed the global instance type
        pytest.skip("Skipping I/O tests: Requires InMemoryStorage to be the active global backend for clearing.")

    global_app_storage_instance.clear_all_data() # Start fresh

    with TestClient(app) as c:
        yield c

    global_app_storage_instance.clear_all_data() # Clean up after test


@pytest.fixture
def sample_td_1(client_with_clean_storage: TestClient) -> TensorDescriptor:
    # Add directly to the global_app_storage_instance that the TestClient's app is using
    td = TensorDescriptor(
        tensor_id=uuid4(), dimensionality=1, shape=[10], data_type=DataType.FLOAT32,
        owner="io_test_user", byte_size=40, tags=["export_test"]
    )
    global_app_storage_instance.add_tensor_descriptor(td)
    return td

@pytest.fixture
def sample_td_2(client_with_clean_storage: TestClient) -> TensorDescriptor:
    td_id = uuid4()
    td = TensorDescriptor(
        tensor_id=td_id, dimensionality=2, shape=[3,3], data_type=DataType.INT64,
        owner="io_test_user2", byte_size=72
    )
    global_app_storage_instance.add_tensor_descriptor(td)
    global_app_storage_instance.add_semantic_metadata(SemanticMetadata(tensor_id=td_id, name="purpose", description="testing import/export"))
    global_app_storage_instance.add_lineage_metadata(LineageMetadata(tensor_id=td_id, version="v1.0-exp"))
    return td


# --- /tensors/export Endpoint Tests ---

def test_export_all_data(client_with_clean_storage: TestClient, sample_td_1: TensorDescriptor, sample_td_2: TensorDescriptor):
    response = client_with_clean_storage.get("/tensors/export")
    assert response.status_code == 200
    assert "attachment; filename=" in response.headers["content-disposition"]

    data = response.json() # FastAPI TestClient .json() handles parsing
    assert data["export_format_version"] == "1.0" # Using the renamed field
    assert len(data["entries"]) == 2

    entry_ids = {entry["tensor_descriptor"]["tensor_id"] for entry in data["entries"]}
    assert str(sample_td_1.tensor_id) in entry_ids
    assert str(sample_td_2.tensor_id) in entry_ids

    td2_entry = next(e for e in data["entries"] if e["tensor_descriptor"]["tensor_id"] == str(sample_td_2.tensor_id))
    assert td2_entry["tensor_descriptor"]["owner"] == "io_test_user2"
    assert len(td2_entry["semantic_metadata"]) == 1
    assert td2_entry["semantic_metadata"][0]["name"] == "purpose"
    assert td2_entry["lineage_metadata"]["version"] == "v1.0-exp"
    assert td2_entry["computational_metadata"] is None

def test_export_selected_tensor_ids(client_with_clean_storage: TestClient, sample_td_1: TensorDescriptor, sample_td_2: TensorDescriptor):
    response = client_with_clean_storage.get(f"/tensors/export?tensor_ids={sample_td_1.tensor_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["entries"]) == 1
    assert data["entries"][0]["tensor_descriptor"]["tensor_id"] == str(sample_td_1.tensor_id)

    response_multi = client_with_clean_storage.get(f"/tensors/export?tensor_ids={sample_td_1.tensor_id},{sample_td_2.tensor_id}")
    assert response_multi.status_code == 200
    assert len(response_multi.json()["entries"]) == 2

def test_export_non_existent_tensor_id(client_with_clean_storage: TestClient, sample_td_1: TensorDescriptor):
    non_existent_id = uuid4()
    response = client_with_clean_storage.get(f"/tensors/export?tensor_ids={non_existent_id},{sample_td_1.tensor_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["entries"]) == 1
    assert data["entries"][0]["tensor_descriptor"]["tensor_id"] == str(sample_td_1.tensor_id)

def test_export_invalid_uuid_format(client_with_clean_storage: TestClient):
    response = client_with_clean_storage.get("/tensors/export?tensor_ids=not-a-uuid,another-bad-uuid")
    assert response.status_code == 400
    assert "Invalid UUID format" in response.json()["detail"]


# --- /tensors/import Endpoint Tests ---
API_KEY = "test_io_key_for_real" # Different from security tests to avoid clash if run together

@pytest.fixture(scope="module", autouse=True)  # Apply to all tests in this module
def setup_io_api_keys_module():
    """Set API key settings for I/O API tests using a local MonkeyPatch."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(global_settings, "VALID_API_KEYS", [API_KEY])
    monkeypatch.setattr(global_settings, "API_KEY_HEADER_NAME", "X-API-KEY")
    from tensorus.api.security import api_key_header_auth as global_api_key_header_auth
    monkeypatch.setattr(global_api_key_header_auth, "name", "X-API-KEY")
    yield
    monkeypatch.undo()


def test_import_data_skip_strategy(client_with_clean_storage: TestClient, sample_td_1: TensorDescriptor):
    export_entry_td1 = TensorusExportEntry(tensor_descriptor=sample_td_1)

    new_td_id = uuid4()
    new_td_payload_model = TensorDescriptor(
        tensor_id=new_td_id, dimensionality=1, shape=[5], data_type=DataType.INT16, owner="importer", byte_size=10
    )
    export_entry_new = TensorusExportEntry(tensor_descriptor=new_td_payload_model)
    import_payload = TensorusExportData(entries=[export_entry_td1, export_entry_new])

    headers = {"X-API-KEY": API_KEY}
    response = client_with_clean_storage.post("/tensors/import?conflict_strategy=skip", json=import_payload.model_dump(mode="json"), headers=headers)

    assert response.status_code == 200
    summary = response.json()
    assert summary["imported"] == 1
    assert summary["skipped"] == 1
    assert summary["overwritten"] == 0
    assert summary["failed"] == 0

    assert client_with_clean_storage.get(f"/tensor_descriptors/{new_td_id}").status_code == 200
    td1_after_import_resp = client_with_clean_storage.get(f"/tensor_descriptors/{sample_td_1.tensor_id}")
    assert td1_after_import_resp.status_code == 200
    assert td1_after_import_resp.json()["owner"] == sample_td_1.owner

def test_import_data_overwrite_strategy(client_with_clean_storage: TestClient, sample_td_1: TensorDescriptor):
    updated_td1_model = sample_td_1.model_copy(deep=True) # Pydantic v2
    # updated_td1_model = sample_td_1.copy(deep=True) # Pydantic v1
    updated_td1_model.owner = "overwritten_owner"
    updated_td1_model.tags = ["overwritten_tag"]

    export_entry_updated_td1 = TensorusExportEntry(tensor_descriptor=updated_td1_model)
    import_payload = TensorusExportData(entries=[export_entry_updated_td1])

    headers = {"X-API-KEY": API_KEY}
    response = client_with_clean_storage.post("/tensors/import?conflict_strategy=overwrite", json=import_payload.model_dump(mode="json"), headers=headers)

    assert response.status_code == 200
    summary = response.json()
    # Overwritten is counted, imported refers to net new usually.
    # The InMemoryStorage import_data counts `overwritten +=1` and `imported` is for truly new.
    assert summary["imported"] == 0
    assert summary["skipped"] == 0
    assert summary["overwritten"] == 1
    assert summary["failed"] == 0

    td1_after_import_resp = client_with_clean_storage.get(f"/tensor_descriptors/{sample_td_1.tensor_id}")
    assert td1_after_import_resp.status_code == 200
    td1_after_import_data = td1_after_import_resp.json()
    assert td1_after_import_data["owner"] == "overwritten_owner"
    assert "overwritten_tag" in td1_after_import_data["tags"]

def test_import_data_with_all_metadata_types(client_with_clean_storage: TestClient):
    td_id = uuid4()
    full_entry = TensorusExportEntry(
        tensor_descriptor=TensorDescriptor(tensor_id=td_id, dimensionality=1, shape=[1], data_type=DataType.UINT8, owner="full_import", byte_size=1),
        semantic_metadata=[SemanticMetadata(tensor_id=td_id, name="sm_name", description="sm_desc")],
        lineage_metadata=LineageMetadata(tensor_id=td_id, version="vImport"),
    )
    import_payload = TensorusExportData(entries=[full_entry])
    headers = {"X-API-KEY": API_KEY}
    response = client_with_clean_storage.post("/tensors/import", json=import_payload.model_dump(mode="json"), headers=headers)

    assert response.status_code == 200
    summary = response.json()
    assert summary["imported"] == 1

    assert client_with_clean_storage.get(f"/tensor_descriptors/{td_id}").status_code == 200
    sm_response = client_with_clean_storage.get(f"/tensor_descriptors/{td_id}/semantic")
    assert sm_response.status_code == 200; assert len(sm_response.json()) == 1
    assert sm_response.json()[0]["name"] == "sm_name"

    lm_response = client_with_clean_storage.get(f"/tensor_descriptors/{td_id}/lineage")
    assert lm_response.status_code == 200; assert lm_response.json()["version"] == "vImport"

def test_import_data_invalid_payload(client_with_clean_storage: TestClient):
    invalid_json_payload = {"foo": "bar"}
    headers = {"X-API-KEY": API_KEY}
    response = client_with_clean_storage.post("/tensors/import", json=invalid_json_payload, headers=headers)
    assert response.status_code == 422

def test_import_data_invalid_conflict_strategy(client_with_clean_storage: TestClient):
    td_id = uuid4()
    entry = TensorusExportEntry(tensor_descriptor=TensorDescriptor(tensor_id=td_id, dimensionality=0, shape=[], data_type=DataType.BOOLEAN, owner="x", byte_size=0))
    import_payload = TensorusExportData(entries=[entry])
    headers = {"X-API-KEY": API_KEY}
    response = client_with_clean_storage.post("/tensors/import?conflict_strategy=delete_all", json=import_payload.model_dump(mode="json"), headers=headers)
    assert response.status_code == 422

def test_import_data_postgres_not_implemented(client_with_clean_storage: TestClient, monkeypatch):
    from unittest.mock import MagicMock
    # Simulate Postgres backend for this test
    mock_postgres_storage = MagicMock(spec=MetadataStorage)
    mock_postgres_storage.import_data.side_effect = NotImplementedError("Postgres import not done.")

    # Override the dependency for this specific test
    # This requires `app` to be accessible or to create a new app instance with this override.
    # `client_with_clean_storage.app` provides the app instance used by the client.
    original_dependency = app.dependency_overrides.get(get_storage_instance)
    app.dependency_overrides[get_storage_instance] = lambda: mock_postgres_storage
    try:
        td_id = uuid4()
        entry = TensorusExportEntry(
            tensor_descriptor=TensorDescriptor(
                tensor_id=td_id,
                dimensionality=0,
                shape=[],
                data_type=DataType.BOOLEAN,
                owner="x",
                byte_size=0,
            )
        )
        import_payload = TensorusExportData(entries=[entry])
        headers = {"X-API-KEY": API_KEY}

        response = client_with_clean_storage.post(
            "/tensors/import",
            json=import_payload.model_dump(mode="json"),
            headers=headers,
        )
        assert response.status_code == 501
        assert "Import functionality is not implemented" in response.json()["detail"]
    finally:
        if original_dependency:
            app.dependency_overrides[get_storage_instance] = original_dependency
        else:
            del app.dependency_overrides[get_storage_instance]
