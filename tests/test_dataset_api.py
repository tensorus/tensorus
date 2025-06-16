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

client = TestClient(app)
TEST_DATASETS = set()

def _cleanup():
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
        client.post("/datasets/create", json={"name": ds})
        TEST_DATASETS.add(ds)
    payload = {"shape": [1], "dtype": "float32", "data": [float(value)], "metadata": {"v": value}}
    resp = client.post(f"/datasets/{ds}/ingest", json=payload)
    assert resp.status_code == 201
    return resp.json()["data"]["record_id"]

def test_records_pagination():
    ds = "pag_api_ds"
    ids = [_ingest(ds, i) for i in range(5)]

    r1 = client.get(f"/datasets/{ds}/records", params={"offset":1, "limit":2})
    assert r1.status_code == 200
    data = r1.json()["data"]
    assert len(data) == 2
    assert data[0]["metadata"]["record_id"] == ids[1]

    r2 = client.get(f"/datasets/{ds}/records", params={"offset":4, "limit":2})
    assert r2.status_code == 200
    assert len(r2.json()["data"]) == 1

    r3 = client.get("/datasets/nonexistent/records")
    assert r3.status_code == 404


def test_ingest_shape_mismatch():
    ds = "mismatch_ds"
    if not tensor_storage_instance.dataset_exists(ds):
        client.post("/datasets/create", json={"name": ds})
        TEST_DATASETS.add(ds)

    # Provide flat data for a 2D shape to trigger validation error
    payload = {
        "shape": [2, 2],
        "dtype": "float32",
        "data": [1.0, 2.0, 3.0, 4.0],
        "metadata": {"v": 1},
    }
    resp = client.post(f"/datasets/{ds}/ingest", json=payload)
    assert resp.status_code == 400
