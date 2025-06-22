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

@pytest.fixture(scope="function") # "function" scope ensures this runs before each test function
def client():
    """
    Provides a FastAPI TestClient instance for API testing.
    Clears all in-memory data stores before yielding the client
    to ensure test isolation.
    """
    # Clear metadata storage
    metadata_storage_instance.clear_all_data()

    # Clear mock tensor storage (assuming it has a similar clear method)
    # The MockTensorStorageConnector needs a clear method. Let's add one.
    # If MockTensorStorageConnector._mock_db and _mock_details_db are directly accessible,
    # they could be cleared here. For encapsulation, a method is better.
    # --- This was done in the previous step, now we can use it ---
    mock_tensor_connector_instance.clear_all_data()

    # Yield the test client for the test to use
    with TestClient(app) as c:
        yield c

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
