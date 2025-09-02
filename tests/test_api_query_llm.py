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


from tests.conftest import TEST_API_KEY # For auth headers and settings
from tensorus.config import settings as global_settings # To patch settings

@pytest.fixture
def client_with_llm(monkeypatch):
    # Ensure global settings for the app instance have API keys configured
    original_api_keys = global_settings.API_KEYS
    original_auth_enabled = global_settings.AUTH_ENABLED
    global_settings.API_KEYS = TEST_API_KEY
    global_settings.AUTH_ENABLED = True

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
            client.headers = {"Authorization": f"Bearer {TEST_API_KEY}"} # Add auth headers
            yield client

    # Restore original settings
    global_settings.API_KEYS = original_api_keys
    global_settings.AUTH_ENABLED = original_auth_enabled


def test_query_endpoint_with_llm_rewrite(client_with_llm):
    # Client already has auth headers from the fixture
    resp = client_with_llm.post("/query", json={"query": "get all data from test_dataset"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["count"] > 0
