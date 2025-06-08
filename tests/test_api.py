import pytest
from fastapi.testclient import TestClient
from uuid import uuid4, UUID
from datetime import datetime

from tensorus.metadata.schemas import TensorDescriptor, SemanticMetadata, DataType, StorageFormat
from tensorus.metadata.storage import storage_instance # To verify direct effects if needed
from tensorus.storage.connectors import mock_tensor_connector_instance # To verify calls or set up mock states

# Test client is provided by the 'client' fixture in conftest.py

# --- Helper Functions ---
def create_sample_td_payload(**overrides):
    payload = {
        "tensor_id": str(uuid4()),
        "dimensionality": 2,
        "shape": [100, 100],
        "data_type": DataType.FLOAT32.value,
        "owner": "api_test_user",
        "byte_size": 40000, # 100 * 100 * 4 (bytes for float32)
        "storage_format": StorageFormat.RAW.value,
        "tags": ["sample", "testing"],
        "metadata": {"source": "api_test"}
    }
    payload.update(overrides)
    return payload

def create_sample_sm_payload(tensor_id: UUID, **overrides):
    payload = {
        "name": "test_semantic_info",
        "description": "Semantic data created via API test.",
        "tensor_id": str(tensor_id)
    }
    payload.update(overrides)
    return payload

# --- TensorDescriptor API Tests ---

def test_create_tensor_descriptor_success(client: TestClient):
    payload = create_sample_td_payload()
    response = client.post("/tensor_descriptors/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["tensor_id"] == payload["tensor_id"]
    assert data["owner"] == payload["owner"]
    assert data["shape"] == payload["shape"]
    # Verify it's in metadata storage
    assert storage_instance.get_tensor_descriptor(UUID(payload["tensor_id"])) is not None
    # Verify it was "stored" in mock tensor connector
    assert mock_tensor_connector_instance.retrieve_tensor(UUID(payload["tensor_id"])) is not None

def test_create_tensor_descriptor_fetch_from_mock_storage(client: TestClient):
    tensor_id = uuid4()
    # "Pre-store" data in mock tensor connector with details
    mock_details = {
        "shape": [50, 50],
        "data_type": DataType.INT16.value,
        "byte_size": 50*50*2,
        "mock_info": "Pre-stored in mock connector"
    }
    mock_tensor_connector_instance.store_tensor(tensor_id, {"data": "dummy", **mock_details})

    payload = {
        "tensor_id": str(tensor_id),
        "owner": "fetch_test_user",
        # Missing: dimensionality, shape, data_type, byte_size
    }
    response = client.post("/tensor_descriptors/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["tensor_id"] == str(tensor_id)
    assert data["owner"] == payload["owner"]
    assert data["shape"] == mock_details["shape"]
    assert data["data_type"] == mock_details["data_type"]
    assert data["byte_size"] == mock_details["byte_size"]
    assert data["dimensionality"] == len(mock_details["shape"])

def test_create_tensor_descriptor_validation_error(client: TestClient):
    payload = create_sample_td_payload(dimensionality=-1) # Invalid dimensionality
    response = client.post("/tensor_descriptors/", json=payload)
    assert response.status_code == 422 # Pydantic validation error
    assert "dimensionality" in response.json()["detail"][0]["loc"]

def test_list_tensor_descriptors_empty(client: TestClient):
    response = client.get("/tensor_descriptors/")
    assert response.status_code == 200
    assert response.json() == []

def test_list_tensor_descriptors_with_items_and_query(client: TestClient):
    td1_payload = create_sample_td_payload(owner="user_a", data_type=DataType.FLOAT32.value)
    client.post("/tensor_descriptors/", json=td1_payload)

    td2_payload = create_sample_td_payload(owner="user_b", data_type=DataType.INT64.value)
    client.post("/tensor_descriptors/", json=td2_payload)

    td3_payload = create_sample_td_payload(owner="user_a", data_type=DataType.INT64.value)
    client.post("/tensor_descriptors/", json=td3_payload)

    response = client.get("/tensor_descriptors/")
    assert response.status_code == 200
    assert len(response.json()) == 3

    # Query by owner
    response_owner_a = client.get("/tensor_descriptors/?owner=user_a")
    assert response_owner_a.status_code == 200
    assert len(response_owner_a.json()) == 2
    assert all(item["owner"] == "user_a" for item in response_owner_a.json())

    # Query by data_type
    response_dt_int64 = client.get("/tensor_descriptors/?data_type=int64")
    assert response_dt_int64.status_code == 200
    assert len(response_dt_int64.json()) == 2
    assert all(item["data_type"] == "int64" for item in response_dt_int64.json())

    # Query by owner and data_type
    response_combined = client.get("/tensor_descriptors/?owner=user_a&data_type=int64")
    assert response_combined.status_code == 200
    assert len(response_combined.json()) == 1
    assert response_combined.json()[0]["owner"] == "user_a"
    assert response_combined.json()[0]["data_type"] == "int64"

def test_get_tensor_descriptor_by_id_success(client: TestClient):
    payload = create_sample_td_payload()
    post_response = client.post("/tensor_descriptors/", json=payload)
    tensor_id = post_response.json()["tensor_id"]

    get_response = client.get(f"/tensor_descriptors/{tensor_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["tensor_id"] == tensor_id
    assert data["owner"] == payload["owner"]

def test_get_tensor_descriptor_by_id_not_found(client: TestClient):
    non_existent_id = uuid4()
    response = client.get(f"/tensor_descriptors/{non_existent_id}")
    assert response.status_code == 404

def test_update_tensor_descriptor_success(client: TestClient):
    payload = create_sample_td_payload(owner="old_owner")
    post_response = client.post("/tensor_descriptors/", json=payload)
    tensor_id = post_response.json()["tensor_id"]

    update_payload = {"owner": "new_owner", "tags": ["production"]}
    put_response = client.put(f"/tensor_descriptors/{tensor_id}", json=update_payload)
    assert put_response.status_code == 200
    data = put_response.json()
    assert data["owner"] == "new_owner"
    assert data["tags"] == ["production"]
    assert data["tensor_id"] == tensor_id
    assert data["last_modified_timestamp"] > payload.get("last_modified_timestamp", data["creation_timestamp"])


def test_update_tensor_descriptor_not_found(client: TestClient):
    non_existent_id = uuid4()
    response = client.put(f"/tensor_descriptors/{non_existent_id}", json={"owner": "ghost"})
    assert response.status_code == 404

def test_update_tensor_descriptor_invalid_data(client: TestClient):
    payload = create_sample_td_payload()
    post_response = client.post("/tensor_descriptors/", json=payload)
    tensor_id = post_response.json()["tensor_id"]

    update_payload = {"byte_size": -100} # Invalid byte_size
    response = client.put(f"/tensor_descriptors/{tensor_id}", json=update_payload)
    assert response.status_code == 422 # Pydantic validation error on update

def test_delete_tensor_descriptor_success(client: TestClient):
    payload = create_sample_td_payload()
    tensor_id = UUID(payload["tensor_id"])
    # Ensure it's in mock storage first
    mock_tensor_connector_instance.store_tensor(tensor_id, {"data": "some data"})
    assert mock_tensor_connector_instance.retrieve_tensor(tensor_id) is not None

    post_response = client.post("/tensor_descriptors/", json=payload)
    assert post_response.status_code == 201

    delete_response = client.delete(f"/tensor_descriptors/{tensor_id}")
    assert delete_response.status_code == 200
    assert "deleted successfully" in delete_response.json()["message"]

    assert storage_instance.get_tensor_descriptor(tensor_id) is None
    assert mock_tensor_connector_instance.retrieve_tensor(tensor_id) is None # Check mock store

def test_delete_tensor_descriptor_not_found(client: TestClient):
    non_existent_id = uuid4()
    response = client.delete(f"/tensor_descriptors/{non_existent_id}")
    assert response.status_code == 404


# --- SemanticMetadata API Tests ---

def test_create_semantic_metadata_success(client: TestClient):
    # First, create a TensorDescriptor to link to
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    assert td_response.status_code == 201
    tensor_id = UUID(td_response.json()["tensor_id"])

    sm_payload = create_sample_sm_payload(tensor_id=tensor_id, name="label_for_td1")
    sm_response = client.post("/semantic_metadata/", json=sm_payload)
    assert sm_response.status_code == 201
    data = sm_response.json()
    assert data["name"] == sm_payload["name"]
    assert data["tensor_id"] == str(tensor_id)
    assert len(storage_instance.get_semantic_metadata(tensor_id)) == 1

def test_create_semantic_metadata_tensor_not_found(client: TestClient):
    non_existent_tensor_id = uuid4()
    sm_payload = create_sample_sm_payload(tensor_id=non_existent_tensor_id)
    response = client.post("/semantic_metadata/", json=sm_payload)
    assert response.status_code == 404 # Because linked TensorDescriptor doesn't exist
    assert f"TensorDescriptor with ID {non_existent_tensor_id} not found" in response.json()["detail"]

def test_create_semantic_metadata_invalid_data(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    sm_payload = create_sample_sm_payload(tensor_id=tensor_id, name="") # Empty name
    response = client.post("/semantic_metadata/", json=sm_payload)
    assert response.status_code == 422 # Pydantic validation for SemanticMetadata

def test_get_semantic_metadata_for_tensor(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    sm_payload1 = create_sample_sm_payload(tensor_id=tensor_id, name="sm1")
    client.post("/semantic_metadata/", json=sm_payload1)
    sm_payload2 = create_sample_sm_payload(tensor_id=tensor_id, name="sm2")
    client.post("/semantic_metadata/", json=sm_payload2)

    response = client.get(f"/semantic_metadata/{tensor_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = {item["name"] for item in data}
    assert "sm1" in names and "sm2" in names

def test_get_semantic_metadata_tensor_not_found(client: TestClient):
    non_existent_tensor_id = uuid4()
    response = client.get(f"/semantic_metadata/{non_existent_tensor_id}")
    assert response.status_code == 404 # TensorDescriptor itself not found

def test_get_semantic_metadata_empty_for_tensor(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    response = client.get(f"/semantic_metadata/{tensor_id}")
    assert response.status_code == 200
    assert response.json() == [] # Empty list, not 404

def test_update_semantic_metadata_entry(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    sm_name = "original_name"
    sm_payload = create_sample_sm_payload(tensor_id=tensor_id, name=sm_name, description="Old Description")
    client.post("/semantic_metadata/", json=sm_payload)

    update_sm_payload = {"description": "New Updated Description"}
    response = client.put(f"/semantic_metadata/{tensor_id}/{sm_name}", json=update_sm_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "New Updated Description"
    assert data["name"] == sm_name # Name should not change with this endpoint

def test_update_semantic_metadata_entry_not_found(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    response = client.put(f"/semantic_metadata/{tensor_id}/non_existent_name", json={"description": "blah"})
    assert response.status_code == 404

def test_delete_specific_semantic_metadata(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    sm_name = "to_be_deleted"
    sm_payload = create_sample_sm_payload(tensor_id=tensor_id, name=sm_name)
    client.post("/semantic_metadata/", json=sm_payload)

    assert len(storage_instance.get_semantic_metadata(tensor_id)) == 1

    response = client.delete(f"/semantic_metadata/{tensor_id}/{sm_name}")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]
    assert len(storage_instance.get_semantic_metadata(tensor_id)) == 0

def test_delete_specific_semantic_metadata_not_found(client: TestClient):
    td_payload = create_sample_td_payload()
    td_response = client.post("/tensor_descriptors/", json=td_payload)
    tensor_id = UUID(td_response.json()["tensor_id"])

    response = client.delete(f"/semantic_metadata/{tensor_id}/non_existent_name")
    assert response.status_code == 404

# Test root endpoint
def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Tensorus API"}
