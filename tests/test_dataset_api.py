import pytest
pytest.importorskip("torch")
from fastapi.testclient import TestClient

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("tensorus.api_local", Path(__file__).resolve().parents[1] / "tensorus" / "api.py")
api_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_module)
app = api_module.app
tensor_storage_instance = api_module.tensor_storage_instance

# Import the test API key for authentication
from tests.conftest import TEST_API_KEY
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}

# Ensure the global settings used by this module's `app` instance are configured with TEST_API_KEY
from tensorus.config import settings as global_settings
global_settings.API_KEYS = TEST_API_KEY
global_settings.AUTH_ENABLED = True # Ensure auth is also enabled

client = TestClient(app)
TEST_DATASETS = set()

def _cleanup():
    # Revert settings if they were changed, though pytest fixtures are better for this.
    # For module-level changes like this, it's harder to clean up perfectly without fixtures.
    # This is a simple approach; a fixture managing settings would be more robust.
    pass # No specific cleanup for settings here, assume test runner isolation or conftest handles env

    for ds in list(TEST_DATASETS):
        if tensor_storage_instance.dataset_exists(ds):
            tensor_storage_instance.delete_dataset(ds)
        TEST_DATASETS.discard(ds)

@pytest.fixture(autouse=True)
def cleanup_dataset():
    yield
    _cleanup()

def _ingest(ds: str, value: int):
    if not tensor_storage_instance.dataset_exists(ds):
        # Add AUTH_HEADERS to client.post calls
        create_resp = client.post("/datasets/create", json={"name": ds}, headers=AUTH_HEADERS)
        assert create_resp.status_code == 201, f"Failed to create dataset {ds}: {create_resp.text}"
        TEST_DATASETS.add(ds)
    payload = {"shape": [1], "dtype": "float32", "data": [float(value)], "metadata": {"v": value}}
    # Add AUTH_HEADERS to client.post calls
    resp = client.post(f"/datasets/{ds}/ingest", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 201, f"Failed to ingest into {ds}: {resp.text}"
    return resp.json()["data"]["record_id"]

def test_records_pagination():
    ds = "pag_api_ds"
    ids = [_ingest(ds, i) for i in range(5)]

    # Add AUTH_HEADERS to client.get calls
    r1 = client.get(f"/datasets/{ds}/records", params={"offset":1, "limit":2}, headers=AUTH_HEADERS)
    assert r1.status_code == 200
    data = r1.json()["data"]
    assert len(data) == 2
    assert data[0]["metadata"]["record_id"] == ids[1]

    # Add AUTH_HEADERS to client.get calls
    r2 = client.get(f"/datasets/{ds}/records", params={"offset":4, "limit":2}, headers=AUTH_HEADERS)
    assert r2.status_code == 200
    assert len(r2.json()["data"]) == 1

    # This endpoint should also be protected if it follows the pattern,
    # but 404 can be returned before auth if path doesn't match.
    # If it requires auth to even determine if it's 404, then headers are needed.
    # Assuming 404 is fine without auth for a non-existent path.
    # However, the API typically checks auth first. So, let's add headers.
    r3 = client.get("/datasets/nonexistent/records", headers=AUTH_HEADERS)
    # The actual check for 404 for a non-existent dataset *after* auth is tricky.
    # Usually, auth failure (401) would precede resource not found (404).
    # If the test expects 404, it implies that either auth is not strictly applied before path checking,
    # or the test setup for "nonexistent" is intended for an unauthenticated check that should fail.
    # Given other tests, let's assume auth is applied. If 'nonexistent' dataset doesn't exist,
    # it should still be 404 after successful auth.
    assert r3.status_code == 404


def test_ingest_shape_mismatch():
    ds = "mismatch_ds"
    if not tensor_storage_instance.dataset_exists(ds):
        # Add AUTH_HEADERS to client.post calls
        create_resp = client.post("/datasets/create", json={"name": ds}, headers=AUTH_HEADERS)
        assert create_resp.status_code == 201, f"Failed to create dataset {ds}: {create_resp.text}"
        TEST_DATASETS.add(ds)

    # Provide flat data for a 2D shape to trigger validation error
    payload = {
        "shape": [2, 2],
        "dtype": "float32",
        "data": [1.0, 2.0, 3.0, 4.0],
        "metadata": {"v": 1},
    }
    # Add AUTH_HEADERS to client.post calls
    resp = client.post(f"/datasets/{ds}/ingest", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 400
