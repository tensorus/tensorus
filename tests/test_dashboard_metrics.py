import os
import importlib.util
import pytest
pytest.importorskip("torch")
import torch
import time
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location(
    "tensorus.api_legacy",
    os.path.join(os.path.dirname(__file__), "..", "tensorus", "api.py"),
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

app = api.app
storage = api.tensor_storage_instance
agent_registry = api.agent_registry

@pytest.fixture(autouse=True)
def clear_storage():
    storage.datasets.clear()
    if storage.storage_path:
        import shutil
        shutil.rmtree(storage.storage_path, ignore_errors=True)
        os.makedirs(storage.storage_path, exist_ok=True)
    yield
    storage.datasets.clear()
    if storage.storage_path:
        import shutil
        shutil.rmtree(storage.storage_path, ignore_errors=True)
        os.makedirs(storage.storage_path, exist_ok=True)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_dashboard_metrics_real(client: TestClient):
    ingestion_ds = agent_registry["ingestion"]["config"]["dataset_name"]
    rl_ds = agent_registry["rl_trainer"]["config"]["experience_dataset"]
    auto_ds = agent_registry["automl_search"]["config"]["results_dataset"]
    custom_ds = "custom_ds"

    for ds in [ingestion_ds, rl_ds, auto_ds, custom_ds]:
        if not storage.dataset_exists(ds):
            storage.create_dataset(ds)

    storage.insert(ingestion_ds, torch.tensor([1.0]), metadata={"timestamp_utc": time.time() - 30})
    storage.insert(
        rl_ds,
        torch.tensor([1.0]),
        metadata={"state_id": "s1", "action": 0, "reward": 2.5, "next_state_id": None, "done": 1},
    )
    storage.insert(auto_ds, torch.tensor([0.1]), metadata={"score": 0.42})
    storage.insert(custom_ds, torch.tensor([0.5]), metadata={})

    resp = client.get("/metrics/dashboard")
    assert resp.status_code == 200
    data = resp.json()

    assert data["dataset_count"] == 4
    assert data["total_records_est"] == 4
    assert data["rl_total_steps"] == 1
    assert data["rl_latest_reward"] == 2.5
    assert data["automl_trials_completed"] == 1
    assert data["automl_best_score"] == 0.42
    assert data["data_ingestion_rate"] > 0
    assert 0.0 <= data["system_cpu_usage_percent"] <= 100.0
    assert 0.0 <= data["system_memory_usage_percent"] <= 100.0

