import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import logging

# ----- Pre-test dependency check -----
missing = []
for pkg in ("torch", "fastapi", "httpx"):
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)

if missing:
    pytest.exit(
        "Missing required packages: {}. Run ./setup.sh before running tests.".format(
            ", ".join(missing)
        ),
        returncode=1,
    )


# Import the FastAPI app instance
from tensorus.api import app
# Import the metadata storage instance that the API uses
from tensorus.metadata import storage_instance as metadata_storage_instance
# Import the mock tensor connector instance that the API uses
from tensorus.storage.connectors import mock_tensor_connector_instance

# Silence verbose debug/info logs from MockTensorStorageConnector during tests
logging.getLogger("tensorus.storage.connectors").setLevel(logging.CRITICAL)

# Test API keys for authentication testing
TEST_API_KEY = "tsr_test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 48 chars after prefix (52 total)
INVALID_API_KEY = "tsr_invalid_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"  # 48 chars after prefix (52 total)

@pytest.fixture(scope="function") 
def client():
    """
    Provides a FastAPI TestClient instance for API testing.
    Clears all in-memory data stores before yielding the client
    to ensure test isolation.
    """
    # Setup test authentication
    import os
    os.environ["TENSORUS_AUTH_ENABLED"] = "true"
    os.environ["TENSORUS_API_KEYS"] = TEST_API_KEY
    
    # Force reload of settings to pick up environment changes
    from tensorus.config import settings
    # Directly set the API keys to bypass parsing issues
    settings.AUTH_ENABLED = True
    settings.API_KEYS = TEST_API_KEY
    settings.VALID_API_KEYS = [TEST_API_KEY]
    
    # Clear metadata storage
    metadata_storage_instance.clear_all_data()

    # Clear mock tensor storage
    mock_tensor_connector_instance.clear_all_data()

    # Yield the test client for the test to use
    with TestClient(app) as c:
        yield c
    
    # Cleanup environment
    os.environ.pop("TENSORUS_AUTH_ENABLED", None)
    os.environ.pop("TENSORUS_API_KEYS", None)


@pytest.fixture
def auth_headers():
    """Provide valid authentication headers for testing."""
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture  
def invalid_auth_headers():
    """Provide invalid authentication headers for testing."""
    return {"Authorization": f"Bearer {INVALID_API_KEY}"}


@pytest.fixture
def legacy_auth_headers():
    """Provide legacy authentication headers for backward compatibility testing."""
    return {"X-API-KEY": TEST_API_KEY}


@pytest.fixture
def authenticated_client(client, auth_headers):
    """Provide a client with authentication headers pre-configured."""
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def unauthenticated_client():
    """
    Provides a FastAPI TestClient instance without authentication enabled
    for testing public endpoints or auth-disabled scenarios.
    """
    # Setup with auth disabled
    import os
    os.environ["TENSORUS_AUTH_ENABLED"] = "false"
    
    # Force settings reload for auth disabled
    from tensorus.config import settings
    settings.AUTH_ENABLED = False
    
    # Clear storage
    metadata_storage_instance.clear_all_data()
    mock_tensor_connector_instance.clear_all_data()

    with TestClient(app) as c:
        yield c
    
    # Cleanup
    os.environ.pop("TENSORUS_AUTH_ENABLED", None)

    # Optional: Clean up after test if necessary, though function scope
    # and clearing at the start usually suffices.
    # metadata_storage_instance.clear_all_data()
    # if hasattr(mock_tensor_connector_instance, '_mock_db'):
    #     mock_tensor_connector_instance._mock_db.clear()
    # if hasattr(mock_tensor_connector_instance, '_mock_details_db'):
    #     mock_tensor_connector_instance._mock_details_db.clear()

# Example fixture for creating a sample TensorDescriptor through the API for reusability
# This is generally more useful if many tests need to perform the same setup steps.
# For now, individual API tests will handle their own setup.
# @pytest.fixture(scope="function")
# def created_tensor_descriptor(client: TestClient):
#     from uuid import uuid4
#     from tensorus.metadata.schemas import DataType
#     tensor_id = uuid4()
#     descriptor_data = {
#         "tensor_id": str(tensor_id),
#         "dimensionality": 2,
#         "shape": [10, 10],
#         "data_type": DataType.FLOAT32.value,
#         "owner": "api_test_user",
#         "byte_size": 400
#     }
#     response = client.post("/tensor_descriptors/", json=descriptor_data)
#     assert response.status_code == 201
#     return response.json()

# The MockTensorStorageConnector needs a clear method.
# This should ideally be in the connectors.py file.
# For the purpose of this subtask, if I can't modify connectors.py now,
# I'll rely on clearing its internal dicts directly as done above.
# If I *could* modify it, I would add:
#
# In tensorus/storage/connectors.py:
# class MockTensorStorageConnector:
#   ...
#   def clear_all_data(self):
#       self._mock_db.clear()
#       self._mock_details_db.clear()
#
# And then in the fixture:
# mock_tensor_connector_instance.clear_all_data()
