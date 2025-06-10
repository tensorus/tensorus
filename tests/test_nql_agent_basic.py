import pytest
import torch
from unittest.mock import patch

from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage

@pytest.fixture
def simple_storage():
    storage = TensorStorage(storage_path=None)
    storage.create_dataset("ds")
    for i in range(2):
        storage.insert("ds", torch.tensor([i]), metadata={"val": i})
    return storage


def test_count_query_uses_storage_count(simple_storage):
    agent = NQLAgent(simple_storage)
    with patch.object(simple_storage, "count", wraps=simple_storage.count) as spy:
        result = agent.process_query("count records in ds")
        assert result["success"]
        assert result["count"] == 2
        spy.assert_called_once_with("ds")


def test_count_query_missing_dataset(simple_storage):
    agent = NQLAgent(simple_storage)
    result = agent.process_query("count records in missing_ds")
    assert not result["success"]
    assert result["count"] is None
