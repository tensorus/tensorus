import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from tensorus.api import app
from tensorus.config import settings as global_settings
from tensorus.metadata import InMemoryStorage, PostgresMetadataStorage, storage_instance as global_storage_instance
from tensorus.metadata.storage_abc import MetadataStorage

# --- Fixtures ---

@pytest.fixture
def client():
    # This client uses the globally configured storage_instance.
    # We will monkeypatch this global_storage_instance for specific test scenarios.
    with TestClient(app) as c:
        yield c

# --- /health Endpoint Tests ---

def test_health_check_in_memory(client: TestClient, monkeypatch):
    # Ensure InMemoryStorage is used
    # We can't easily swap the type of global_storage_instance after it's imported by other modules.
    # Instead, we mock the check_health method of the *actual* global_storage_instance
    # or ensure global_settings make it InMemory for this test.

    # If current global_storage_instance is InMemory, this test is direct.
    # If it could be Postgres, we need to ensure it behaves like InMemory for this test.

    # Let's assume we can control the global_storage_instance type via settings for this test run scope.
    # This is hard due to import-time evaluation. A better way is to mock the dependency's result.

    # Simplest: Directly mock the method on the already configured global_storage_instance
    # This assumes that the `get_storage_instance` dependency in the endpoint will return this global one.

    with patch.object(global_storage_instance, 'check_health', return_value=(True, "in_memory_mocked")) as mock_check:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["backend"] == "in_memory_mocked"
        mock_check.assert_called_once()

def test_health_check_postgres_ok(client: TestClient, monkeypatch):
    # Simulate Postgres backend being healthy
    with patch.object(global_storage_instance, 'check_health', return_value=(True, "postgres_mocked")) as mock_check:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["backend"] == "postgres_mocked"
        mock_check.assert_called_once()

def test_health_check_postgres_error(client: TestClient, monkeypatch):
    # Simulate Postgres backend being unhealthy
    with patch.object(global_storage_instance, 'check_health', return_value=(False, "postgres_mocked_fail")) as mock_check:
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert data["backend"] == "postgres_mocked_fail"
        assert "Storage backend connection failed" in data["detail"]
        mock_check.assert_called_once()


# --- /metrics Endpoint Tests ---

def test_metrics_endpoint_empty(client: TestClient, monkeypatch):
    # Mock the storage methods to return specific counts
    with patch.object(global_storage_instance, 'get_tensor_descriptors_count', return_value=0) as mock_td_count, \
         patch.object(global_storage_instance, 'get_extended_metadata_count') as mock_ext_count:

        # Set up side_effect for get_extended_metadata_count if it's called with different args
        mock_ext_count.return_value = 0 # Default for any call

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        expected_metrics = {
            "total_tensor_descriptors": 0,
            "semantic_metadata_count": 0,
            "lineage_metadata_count": 0,
            "computational_metadata_count": 0,
            "quality_metadata_count": 0,
            "relational_metadata_count": 0,
            "usage_metadata_count": 0
        }
        assert data == expected_metrics
        mock_td_count.assert_called_once()
        # Check that get_extended_metadata_count was called for each type
        assert mock_ext_count.call_count == 6 # For the 6 extended types listed (Semantic is one of them)


def test_metrics_endpoint_with_data(client: TestClient, monkeypatch):
    # Simulate some data
    with patch.object(global_storage_instance, 'get_tensor_descriptors_count', return_value=5) as mock_td_count, \
         patch.object(global_storage_instance, 'get_extended_metadata_count') as mock_ext_count:

        def ext_count_side_effect(model_name: str):
            counts = {
                "SemanticMetadata": 10, "LineageMetadata": 4, "ComputationalMetadata": 3,
                "QualityMetadata": 2, "RelationalMetadata": 1, "UsageMetadata": 5
            }
            return counts.get(model_name, 0)

        mock_ext_count.side_effect = ext_count_side_effect

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        expected_metrics = {
            "total_tensor_descriptors": 5,
            "semantic_metadata_count": 10,
            "lineage_metadata_count": 4,
            "computational_metadata_count": 3,
            "quality_metadata_count": 2,
            "relational_metadata_count": 1,
            "usage_metadata_count": 5
        }
        assert data == expected_metrics
        mock_td_count.assert_called_once()
        assert mock_ext_count.call_count == 6


# Note on testing with different backends:
# The `client` fixture uses the app's globally configured `storage_instance`.
# To test behavior with a specific backend (e.g., truly testing Postgres health check failure),
# one would typically:
# 1. Set environment variables (`TENSORUS_STORAGE_BACKEND="postgres"`, etc.) BEFORE the app and `settings` are loaded.
#    Pytest's `monkeypatch.setenv` can do this if applied before relevant modules are imported.
# 2. Or, have a fixture that yields a TestClient configured with a specific storage backend instance.
#    This might involve creating a new FastAPI app instance within the fixture, overriding dependencies.
#
# The current approach of mocking the methods of the *globally configured* `storage_instance`
# is a pragmatic way to test the API endpoint logic without needing to fully reconfigure
# the global state for each test, which can be tricky with Python's import system.
# This tests that the endpoint calls the right storage methods and handles their responses correctly.
# It assumes the `storage_instance` itself (InMemory or Postgres) implements those methods correctly,
# which is tested in `test_storage.py` and `test_postgres_storage.py`.
