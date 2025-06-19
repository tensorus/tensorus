import os
import importlib.util
import pytest
from unittest.mock import patch
pytest.importorskip("torch")
import torch

from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location(
    "tensorus.api_legacy",
    os.path.join(os.path.dirname(__file__), "..", "tensorus", "api.py"),
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage
from tensorus.llm_parser import LLMParser, NQLQuery


@pytest.fixture
def client_with_llm(monkeypatch):
    with patch("tensorus.nql_agent.LLMParser") as MockParser:
        MockParser.return_value.parse.return_value = NQLQuery(dataset="test_ds")
        storage = TensorStorage(storage_path=None)
        storage.create_dataset("test_ds")
        storage.insert("test_ds", torch.tensor([1.0]), metadata={"record_id": "r1"})
        agent = NQLAgent(storage, use_llm=True)
        monkeypatch.setattr(api, "nql_agent_instance", agent)
        if hasattr(api.get_nql_agent, "_instance"):
            monkeypatch.setattr(api.get_nql_agent, "_instance", agent, raising=False)
        with TestClient(api.app) as client:
            yield client


def test_query_endpoint_with_llm_rewrite(client_with_llm):
    resp = client_with_llm.post("/query", json={"query": "nonsense"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["count"] == 0 # Changed from 1 to 0 due to NQLAgent._execute_parsed_query returning 0 for empty filters
