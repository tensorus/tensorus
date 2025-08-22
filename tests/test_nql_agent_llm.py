import pytest
from unittest.mock import patch
pytest.importorskip("torch")
import torch

from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage
from tensorus.llm_parser import LLMParser, NQLQuery

@pytest.fixture
def sample_storage():
    storage = TensorStorage(storage_path=None)
    storage.create_dataset("test_ds")
    storage.insert("test_ds", torch.tensor([1.0]), metadata={"record_id": "r1"})
    return storage


def test_process_query_uses_llm(sample_storage):
    with patch("tensorus.nql_agent.LLMParser") as MockParser:
        MockParser.return_value.parse.return_value = NQLQuery(dataset="test_ds")
        agent = NQLAgent(sample_storage, use_llm=True)
        result = agent.process_query("unknown text")
        MockParser.return_value.parse.assert_called_once()
        assert result["success"]
        assert result["count"] == 1


def test_parse_failure_returns_error(sample_storage):
    with patch("tensorus.nql_agent.LLMParser") as MockParser:
        MockParser.return_value.parse.return_value = None
        agent = NQLAgent(sample_storage, use_llm=True)
        result = agent.process_query("bad query")
        MockParser.return_value.parse.assert_called_once()
        assert not result["success"]
