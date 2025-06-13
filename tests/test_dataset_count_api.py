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

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def _ingest(client, dataset, value):
    payload = {"shape": [1], "dtype": "float32", "data": [float(value)], "metadata": {"v": value}}
    return client.post(f"/datasets/{dataset}/ingest", json=payload)


def test_count_endpoint(client):
    ds = "count_ds"
    assert client.post("/datasets/create", json={"name": ds}).status_code == 201
    _ingest(client, ds, 1)
    _ingest(client, ds, 2)
    resp = client.get(f"/datasets/{ds}/count")
    assert resp.status_code == 200
    assert resp.json()["data"]["count"] == 2

    resp_missing = client.get("/datasets/missing_ds/count")
    assert resp_missing.status_code == 404
