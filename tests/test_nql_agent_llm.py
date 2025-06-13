import pytest
from unittest.mock import patch
pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch

from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage

@pytest.fixture
def sample_storage():
    storage = TensorStorage(storage_path=None)
    storage.create_dataset("test_ds")
    storage.insert("test_ds", torch.tensor([1.0]), metadata={"record_id": "r1"})
    return storage


def test_process_query_uses_llm_rewrite(sample_storage):
    with patch("transformers.pipeline") as mock_root, patch("transformers.pipelines.pipeline") as mock_pipe:
        mock_root.return_value = mock_pipe.return_value = lambda q: [{"generated_text": "get all data from test_ds"}]
        agent = NQLAgent(sample_storage, use_llm=True)
        with patch.object(agent, "_llm_rewrite_query", wraps=agent._llm_rewrite_query) as spy:
            result = agent.process_query("unknown text")
            assert spy.call_count == 1
            assert result["success"]
            assert result["count"] == 1


def test_rewrite_stops_after_failure(sample_storage):
    with patch("transformers.pipeline") as mock_root, patch("transformers.pipelines.pipeline") as mock_pipe:
        mock_root.return_value = mock_pipe.return_value = lambda q: [{"generated_text": "still unsupported"}]
        agent = NQLAgent(sample_storage, use_llm=True)
        with patch.object(agent, "_llm_rewrite_query", wraps=agent._llm_rewrite_query) as spy:
            result = agent.process_query("bad query")
            assert spy.call_count == 1
            assert not result["success"]
