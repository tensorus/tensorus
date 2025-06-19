import json
import pytest
pytest.importorskip("torch")
import torch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage
import tensorus.llm_parser as llm_parser


def _patch_fake_llm(monkeypatch, response_str):
    def _fake_model(*args, **kwargs):
        return FakeListChatModel(responses=[response_str])
    monkeypatch.setattr(llm_parser, "ChatGoogleGenerativeAI", _fake_model)


@pytest.fixture
def sample_storage():
    storage = TensorStorage(storage_path=None)
    storage.create_dataset("numbers")
    storage.insert("numbers", torch.tensor([1.0]), metadata={"idx": 1})
    storage.insert("numbers", torch.tensor([2.0]), metadata={"idx": 2})

    storage.create_dataset("sensors")
    storage.insert("sensors", torch.tensor([0.0]), metadata={"id": "A", "status": "active"})
    storage.insert("sensors", torch.tensor([0.0]), metadata={"id": "B", "status": "inactive"})
    storage.insert("sensors", torch.tensor([0.0]), metadata={"id": "C", "status": "active"})

    storage.create_dataset("values")
    storage.insert("values", torch.tensor([0.1]), metadata={"id": "v1", "t0": 0.1})
    storage.insert("values", torch.tensor([0.3]), metadata={"id": "v2", "t0": 0.3})
    storage.insert("values", torch.tensor([0.05]), metadata={"id": "v3", "t0": 0.05})
    return storage


def test_llm_count_query(monkeypatch, sample_storage):
    monkeypatch.setenv("NQL_USE_LLM", "true")
    response = json.dumps({"dataset": "numbers", "filters": []})
    _patch_fake_llm(monkeypatch, response)
    agent = NQLAgent(sample_storage, use_llm=True)
    res = agent.process_query("How many items does numbers contain?")
    assert res["success"]
    assert res["count"] == 2


def test_llm_metadata_filter(monkeypatch, sample_storage):
    monkeypatch.setenv("NQL_USE_LLM", "true")
    response = json.dumps({
        "dataset": "sensors",
        "filters": [
            {
                "joiner": "AND",
                "conditions": [
                    {"key": "status", "operator": "=", "value": "active"}
                ],
            }
        ],
    })
    _patch_fake_llm(monkeypatch, response)
    agent = NQLAgent(sample_storage, use_llm=True)
    res = agent.process_query("Which sensors are active?")
    assert res["success"]
    assert res["count"] == 2


def test_llm_tensor_value_filter(monkeypatch, sample_storage):
    monkeypatch.setenv("NQL_USE_LLM", "true")
    response = json.dumps({
        "dataset": "values",
        "filters": [
            {
                "joiner": "AND",
                "conditions": [
                    {"key": "t0", "operator": ">", "value": 0.2}
                ],
            }
        ],
    })
    _patch_fake_llm(monkeypatch, response)
    agent = NQLAgent(sample_storage, use_llm=True)
    res = agent.process_query("Find records with first value above 0.2")
    assert res["success"]
    assert res["count"] == 1
    ids = [r["metadata"]["id"] for r in res["results"]]
    assert ids == ["v2"]


def test_llm_parse_failure(monkeypatch, sample_storage):
    monkeypatch.setenv("NQL_USE_LLM", "true")
    _patch_fake_llm(monkeypatch, "not json")
    agent = NQLAgent(sample_storage, use_llm=True)
    res = agent.process_query("nonsense")
    assert not res["success"]
