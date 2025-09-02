import os
import importlib.util
import pytest
pytest.importorskip("torch")
from fastapi.testclient import TestClient

# Load the legacy api module which exposes dataset endpoints
spec = importlib.util.spec_from_file_location(
    "tensorus.api_legacy",
    os.path.join(os.path.dirname(__file__), "..", "tensorus", "api.py"),
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

app = api.app
storage = api.tensor_storage_instance

@pytest.fixture(autouse=True)
def clear_storage():
    storage.datasets.clear()
    yield
    storage.datasets.clear()

# Import the test API key for authentication
from tests.conftest import TEST_API_KEY
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}

# Ensure the global settings used by this module's `app` instance are configured with TEST_API_KEY
from tensorus.config import settings as global_settings
global_settings.API_KEYS = TEST_API_KEY
global_settings.AUTH_ENABLED = True # Ensure auth is also enabled


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def _ingest(client, dataset, value):
    payload = {"shape": [1], "dtype": "float32", "data": [float(value)], "metadata": {"v": value}}
    # Add AUTH_HEADERS
    ingest_resp = client.post(f"/datasets/{dataset}/ingest", json=payload, headers=AUTH_HEADERS)
    assert ingest_resp.status_code == 201, f"Failed to ingest into {dataset}: {ingest_resp.text}"
    return ingest_resp


def test_count_endpoint(client):
    ds = "count_ds"
    # Add AUTH_HEADERS
    create_resp = client.post("/datasets/create", json={"name": ds}, headers=AUTH_HEADERS)
    assert create_resp.status_code == 201, f"Failed to create dataset {ds}: {create_resp.text}"

    _ingest(client, ds, 1)
    _ingest(client, ds, 2)

    # Add AUTH_HEADERS
    resp = client.get(f"/datasets/{ds}/count", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["data"]["count"] == 2

    # Add AUTH_HEADERS, similar reasoning to test_dataset_api.py for 404
    resp_missing = client.get("/datasets/missing_ds/count", headers=AUTH_HEADERS)
    assert resp_missing.status_code == 404
