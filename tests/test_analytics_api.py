import pytest
from fastapi.testclient import TestClient
from uuid import uuid4, UUID
from datetime import datetime, timedelta

# FastAPI application under test
from tensorus.api import app
from tensorus.metadata.storage import InMemoryStorage
from tensorus.metadata.storage_abc import MetadataStorage
from tensorus.metadata.schemas import (
    TensorDescriptor, LineageMetadata, UsageMetadata, DataType,
    ParentTensorLink, TransformationStep
)
from tensorus.config import settings as global_settings
from tensorus.metadata import storage_instance as global_app_storage_instance

# --- Fixtures ---

@pytest.fixture(scope="function")
def client_with_clean_storage_analytics(monkeypatch):
    """
    Provides a TestClient with a fresh InMemoryStorage for analytics tests.
    """
    monkeypatch.setattr(global_settings, "STORAGE_BACKEND", "in_memory")
    if not isinstance(global_app_storage_instance, InMemoryStorage):
        pytest.skip("Skipping Analytics API tests: Requires InMemoryStorage for clean state.")
    global_app_storage_instance.clear_all_data()

    with TestClient(app) as c:
        yield c

    global_app_storage_instance.clear_all_data()


@pytest.fixture
def analytics_setup_data(client_with_clean_storage_analytics: TestClient):
    """
    Populates storage with diverse data for testing analytics endpoints.
    Uses the global_app_storage_instance directly for setup simplicity.
    """
    storage = global_app_storage_instance

    tds_data = []
    for days_ago, owner, tags in [
        (10, "user1", ["tagA", "tagB", "tagC"]),
        (100, "user2", ["tagB", "tagC", "tagD"]),
        (5, "user1", ["tagA", "tagD", "tagE"]),
        (200, "user3", ["tagX", "tagY"]),
        (1, "user4", ["tagA", "tagB", "tagD"]),  # Recent; includes tagD for co-occurrence tests
    ]:
        ts = datetime.utcnow() - timedelta(days=days_ago)
        tds_data.append(
            {
                "tensor_id": uuid4(),
                "owner": owner,
                "tags": tags,
                "creation_timestamp": ts,
                "last_modified_timestamp": ts,
            }
        )

    created_tds = []
    for i, data in enumerate(tds_data):
        td = TensorDescriptor(
            dimensionality=1, shape=[1], data_type=DataType.FLOAT32, byte_size=4,
            **data
        )
        storage.add_tensor_descriptor(td)
        created_tds.append(td)

    # Add UsageMetadata for some
    # td1 (index 0) -> accessed recently
    storage.add_usage_metadata(UsageMetadata(tensor_id=created_tds[0].tensor_id, last_accessed_at=datetime.utcnow() - timedelta(days=1)))
    # td2 (index 1) -> last_accessed_at is older than last_modified, but still stale by last_modified
    storage.add_usage_metadata(UsageMetadata(tensor_id=created_tds[1].tensor_id, last_accessed_at=datetime.utcnow() - timedelta(days=150)))
    # td3 (index 2) -> accessed very long ago, but modified recently
    storage.add_usage_metadata(UsageMetadata(tensor_id=created_tds[2].tensor_id, last_accessed_at=datetime.utcnow() - timedelta(days=300)))


    # Add LineageMetadata for some
    # td1: 1 parent, 1 step
    storage.add_lineage_metadata(LineageMetadata(tensor_id=created_tds[0].tensor_id, parent_tensors=[ParentTensorLink(tensor_id=uuid4())], transformation_history=[TransformationStep(operation="op1")]))
    # td2: 2 parents, 3 steps
    storage.add_lineage_metadata(LineageMetadata(tensor_id=created_tds[1].tensor_id, parent_tensors=[ParentTensorLink(tensor_id=uuid4()), ParentTensorLink(tensor_id=uuid4())], transformation_history=[TransformationStep(operation="op1"), TransformationStep(operation="op2"), TransformationStep(operation="op3")]))
    # td4 (index 3): 0 parents, 5 steps
    storage.add_lineage_metadata(LineageMetadata(tensor_id=created_tds[3].tensor_id, transformation_history=[TransformationStep(operation=f"op{i}") for i in range(5)]))

    return created_tds


# --- /analytics/co_occurring_tags Tests ---

def test_get_co_occurring_tags_default_params(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    response = client_with_clean_storage_analytics.get("/analytics/co_occurring_tags")
    assert response.status_code == 200
    data = response.json()

    assert "tagA" in data
    assert "tagB" in data
    assert "tagC" in data
    assert "tagD" in data

    # Check for tagA: expects tagB (2 times with default min_co_occurrence=2)
    # td0: A,B,C -> (A,B), (A,C)
    # td2: A,D,E -> (A,D), (A,E)
    # td4: A,B   -> (A,B)
    # So, (A,B) occurs 2 times.
    tag_a_co = {item["tag"]: item["count"] for item in data.get("tagA", [])}
    assert tag_a_co.get("tagB") == 2

    tag_b_co = {item["tag"]: item["count"] for item in data.get("tagB", [])}
    assert tag_b_co.get("tagA") == 2
    assert tag_b_co.get("tagC") == 2


def test_get_co_occurring_tags_custom_params(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    response = client_with_clean_storage_analytics.get("/analytics/co_occurring_tags?min_co_occurrence=3&limit=5")
    assert response.status_code == 200
    data = response.json()
    # With min_co_occurrence=3, only pairs occurring 3+ times.
    # In setup: (tagA,tagB):2, (tagA,tagC):1, (tagA,tagD):1, (tagA,tagE):0
    # (tagB,tagC):2, (tagB,tagD):1
    # (tagC,tagD):1
    # No pairs occur 3 times with the current setup data.
    assert len(data) == 0 # Expect empty if no tags meet min_co_occurrence of 3

    # Test with min_co_occurrence=1 to get more results
    response_min1 = client_with_clean_storage_analytics.get("/analytics/co_occurring_tags?min_co_occurrence=1&limit=1")
    assert response_min1.status_code == 200
    data_min1 = response_min1.json()
    assert "tagA" in data_min1
    if data_min1.get("tagA"): # If tagA has co-occurring tags
        assert len(data_min1["tagA"]) <= 1 # Limit is 1

def test_get_co_occurring_tags_no_tags_or_no_cooccurrence(client_with_clean_storage_analytics: TestClient):
    # Storage is cleared by fixture. Add a tensor with no tags, or one tag.
    storage = global_app_storage_instance
    storage.add_tensor_descriptor(TensorDescriptor(dimensionality=1, shape=[1], data_type=DataType.FLOAT32, owner="u", byte_size=4, tags=[]))
    storage.add_tensor_descriptor(TensorDescriptor(dimensionality=1, shape=[1], data_type=DataType.FLOAT32, owner="u", byte_size=4, tags=["single"]))

    response = client_with_clean_storage_analytics.get("/analytics/co_occurring_tags")
    assert response.status_code == 200
    assert response.json() == {}


# --- /analytics/stale_tensors Tests ---

def test_get_stale_tensors_default_threshold(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    # Default threshold_days = 90
    # analytics_setup_data:
    # td[0] ("tagA", "tagB", "tagC"): modified 10d ago, accessed 1d ago -> NOT STALE
    # td[1] ("tagB", "tagC", "tagD"): modified 100d ago, accessed 150d ago -> STALE (last_relevant is 100d ago)
    # td[2] ("tagA", "tagD", "tagE"): modified 5d ago, accessed 300d ago -> NOT STALE
    # td[3] ("tagX", "tagY"): modified 200d ago, no usage data -> STALE
    # td[4] ("tagA", "tagB", "tagD"): modified 1d ago, no usage data -> NOT STALE
    response = client_with_clean_storage_analytics.get("/analytics/stale_tensors")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    stale_ids = {item["tensor_id"] for item in data}
    assert str(analytics_setup_data[1].tensor_id) in stale_ids # td[1]
    assert str(analytics_setup_data[3].tensor_id) in stale_ids # td[3]

def test_get_stale_tensors_custom_threshold(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    # Threshold 0 days (everything not touched/modified "today" is stale)
    # Depending on exact timing, this could include many. Let's use a small number like 3 days.
    response = client_with_clean_storage_analytics.get("/analytics/stale_tensors?threshold_days=3")
    assert response.status_code == 200
    data = response.json()
    # td[0]: modified 10d, accessed 1d -> NOT STALE by 3d rule
    # td[1]: modified 100d, accessed 150d -> STALE
    # td[2]: modified 5d, accessed 300d -> STALE by 3d rule (last_relevant is 5d ago)
    # td[3]: modified 200d -> STALE
    # td[4]: modified 1d -> NOT STALE
    assert len(data) == 3
    stale_ids = {item["tensor_id"] for item in data}
    assert str(analytics_setup_data[1].tensor_id) in stale_ids
    assert str(analytics_setup_data[2].tensor_id) in stale_ids
    assert str(analytics_setup_data[3].tensor_id) in stale_ids


# --- /analytics/complex_tensors Tests ---

def test_get_complex_tensors_by_parents(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    # td[0]: 1 parent
    # td[1]: 2 parents
    # td[3]: 0 parents
    response = client_with_clean_storage_analytics.get("/analytics/complex_tensors?min_parent_count=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["tensor_id"] == str(analytics_setup_data[1].tensor_id)

def test_get_complex_tensors_by_transformations(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    # td[0]: 1 step
    # td[1]: 3 steps
    # td[3]: 5 steps
    response = client_with_clean_storage_analytics.get("/analytics/complex_tensors?min_transformation_steps=4")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["tensor_id"] == str(analytics_setup_data[3].tensor_id)

def test_get_complex_tensors_by_either_criterion(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    # td[1]: 2 parents, 3 steps
    # td[3]: 0 parents, 5 steps
    response = client_with_clean_storage_analytics.get("/analytics/complex_tensors?min_parent_count=2&min_transformation_steps=4")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2 # Both td[1] (by parents) and td[3] (by steps) should match if logic is OR
    # Current InMemoryStorage logic is OR, so this is correct.
    complex_ids = {item["tensor_id"] for item in data}
    assert str(analytics_setup_data[1].tensor_id) in complex_ids
    assert str(analytics_setup_data[3].tensor_id) in complex_ids


def test_get_complex_tensors_no_criteria(client_with_clean_storage_analytics: TestClient):
    response = client_with_clean_storage_analytics.get("/analytics/complex_tensors")
    assert response.status_code == 400 # Bad Request
    assert "At least one criterion" in response.json()["detail"]

def test_get_complex_tensors_limit(client_with_clean_storage_analytics: TestClient, analytics_setup_data):
    # All of td[0], td[1], td[3] have some lineage.
    # td[0]: 1 parent, 1 step
    # td[1]: 2 parents, 3 steps
    # td[3]: 0 parents, 5 steps
    response = client_with_clean_storage_analytics.get("/analytics/complex_tensors?min_parent_count=0&limit=1") # min_parent_count=0 should include those with lineage
    assert response.status_code == 200
    assert len(response.json()) == 1

    response_steps = client_with_clean_storage_analytics.get("/analytics/complex_tensors?min_transformation_steps=1&limit=2")
    assert response_steps.status_code == 200
    assert len(response_steps.json()) == 2


# Conceptual tests for Postgres (mocking storage methods) could be added here
# if specific API error handling or data transformation for these endpoints needed testing
# independent of InMemoryStorage logic. For now, covered by storage tests.
