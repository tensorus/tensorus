import pytest
pytest.importorskip("torch")
from unittest.mock import patch
import torch

from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage

@pytest.fixture
def basic_storage():
    storage = TensorStorage(storage_path=None) # In-memory for tests
    storage.create_dataset("ds1")
    storage.insert("ds1", torch.tensor([1.0]), metadata={"id": "007", "name": "Sensor Alpha", "status": "active", "reading": 100, "level": 5.5, "is_calibrated": True})
    storage.insert("ds1", torch.tensor([2.0]), metadata={"id": "008", "name": "Sensor Beta", "status": "inactive", "reading": 200, "level": 5.5, "is_calibrated": False})
    storage.insert("ds1", torch.tensor([3.0]), metadata={"id": "ABC", "name": "Sensor Gamma", "status": "active", "reading": 150, "level": 6.0}) # Missing is_calibrated
    storage.insert("ds1", torch.tensor([4.0]), metadata={"id": "009", "name": "Sensor Delta", "status": "active", "reading": "N/A"}) # Reading as string

    storage.create_dataset("ds2")
    storage.insert("ds2", torch.tensor([0.1, 0.2]), metadata={"record_id": "rec1", "value": 10})
    storage.insert("ds2", torch.tensor([0.3, 0.4, 0.5]), metadata={"record_id": "rec2", "value": 20})
    storage.insert("ds2", torch.tensor([]), metadata={"record_id": "rec3", "value": 0}) # Empty tensor

    return storage


def test_count_query(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("count records in ds1")
    assert result["success"]
    assert result["count"] == 4

    result = agent.process_query("count records from ds2")
    assert result["success"]
    assert result["count"] == 3


def test_count_query_missing_dataset(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("count records in missing_ds")
    assert not result["success"]
    assert result["message"] == "Dataset 'missing_ds' not found."
    assert result["count"] is None


def test_get_all_query(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("get all data from ds1")
    assert result["success"]
    assert result["count"] == 4
    assert len(result["results"]) == 4
    assert result["results"][0]["metadata"]["id"] == "007"

    result = agent.process_query("show all tensors from ds2")
    assert result["success"]
    assert result["count"] == 3
    assert len(result["results"]) == 3


def test_get_all_query_missing_dataset(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("get all records from missing_ds")
    assert not result["success"]
    # This error might come from TensorStorage, so message could vary.
    # For now, we check success is False. A more specific message check might be needed
    # if TensorStorage guarantees specific error messages for DatasetNotFoundError.
    # Based on NQLAgent's current `process_query` for get_all, it might be a ValueError.
    # The count pattern has explicit DatasetNotFoundError handling. Get all doesn't yet.
    # Let's assume for now it's a generic message or add specific handling later.
    assert "missing_ds" in result["message"].lower() # Check if dataset name is in message


# --- Metadata Filter Tests (Focus of T1 from planning) ---

@pytest.mark.parametrize("query, expected_ids", [
    # String ID comparisons (id is string "007", "008", "ABC", "009")
    ("find records from ds1 where id = '007'", ["007"]),
    ("find records from ds1 where id == \"007\"", ["007"]), # Double quotes
    ("find records from ds1 where id = 'ABC'", ["ABC"]),
    ("find records from ds1 where id = 007", []), # Unquoted "007" becomes num 7, should not match string "007"
    ("find records from ds1 where id != '007'", ["008", "ABC", "009"]),
    ("find records from ds1 where id != 007", ["007", "008", "ABC", "009"]), # num 7 != string "007" (coercion to string then compare)

    # Numeric value comparisons (reading is number 100, 200, 150; level is float 5.5, 6.0)
    ("find records from ds1 where reading = 100", ["007"]),
    ("find records from ds1 where reading == 150", ["ABC"]),
    ("find records from ds1 where reading = '100'", ["007"]), # Query "100" (str) vs metadata 100 (int) -> coercion
    ("find records from ds1 where reading > 100", ["008", "ABC"]),
    ("find records from ds1 where reading >= 150", ["008", "ABC"]),
    ("find records from ds1 where reading < 150", ["007"]),
    ("find records from ds1 where reading <= 100", ["007"]),
    ("find records from ds1 where reading != 100", ["008", "ABC", "009"]), # "009" has reading "N/A"
    ("find records from ds1 where level = 5.5", ["007", "008"]),
    ("find records from ds1 where level > 5.5", ["ABC"]),
    ("find records from ds1 where level = '5.5'", ["007", "008"]), # Query "5.5" (str) vs metadata 5.5 (float)

    # String value comparisons (status is string "active", "inactive")
    ("find records from ds1 where status = 'active'", ["007", "ABC", "009"]),
    ("find records from ds1 where status is 'inactive'", ["008"]), # alt syntax
    ("find records from ds1 where status equals \"active\"", ["007", "ABC", "009"]), # alt syntax
    ("find records from ds1 where status eq 'active'", ["007", "ABC", "009"]), # alt syntax

    # Boolean value comparisons (is_calibrated is True, False, or missing)
    # Current _parse_operator_and_value will treat True/False as unquoted strings if not careful.
    # With the fix, unquoted True/False will be strings "True"/"False".
    # Coercion logic in query_fn_meta: if filter_value is str, actual_value becomes str.
    ("find records from ds1 where is_calibrated = 'True'", ["007"]), # actual True -> "True" vs "True"
    ("find records from ds1 where is_calibrated = 'False'", ["008"]),# actual False -> "False" vs "False"
    # ("find records from ds1 where is_calibrated = True", ["007"]), # This query would make filter_value string "True"
    # ("find records from ds1 where is_calibrated = 1", []), # is_calibrated (bool) vs 1 (int) - needs specific handling or relies on coercion

    # Handling of metadata key "reading" which is string "N/A" for one record
    ("find records from ds1 where reading = 'N/A'", ["009"]),
    ("find records from ds1 where reading != 'N/A'", ["007", "008", "ABC"]),
    ("find records from ds1 where reading > 1000", []), # "N/A" should not satisfy > numeric
    ("find records from ds1 where reading = 0", []), # "N/A" should not satisfy = numeric
])
def test_metadata_filtering(basic_storage, query, expected_ids):
    agent = NQLAgent(basic_storage)
    result = agent.process_query(query)
    assert result["success"], f"Query '{query}' failed: {result.get('message', 'No message')}"
    assert result["count"] == len(expected_ids)

    found_ids = sorted([res["metadata"]["id"] for res in result["results"]])
    assert found_ids == sorted(expected_ids)

def test_filter_meta_key_not_exist(basic_storage):
    agent = NQLAgent(basic_storage)
    query = "find records from ds1 where non_existent_key = 'value'"
    result = agent.process_query(query)
    assert result["success"]
    assert result["count"] == 0

def test_filter_meta_partially_existing_key(basic_storage):
    agent = NQLAgent(basic_storage)
    # 'is_calibrated' is missing from one record ("ABC")
    query = "find records from ds1 where is_calibrated = 'True'"
    result = agent.process_query(query)
    assert result["success"]
    assert result["count"] == 1
    assert result["results"][0]["metadata"]["id"] == "007"

# --- Tensor Filter Tests ---

@pytest.mark.parametrize("query, expected_ids_values", [
    # ds2: rec1: [0.1, 0.2], val:10; rec2: [0.3, 0.4, 0.5], val:20; rec3: [], val:0
    ("get records from ds2 where tensor[0] > 0.05", {"rec1": 0.1, "rec2": 0.3}),
    ("get records from ds2 where tensor[1] < 0.3", {"rec1": 0.2}),
    ("get records from ds2 where tensor[2] = 0.5", {"rec2": 0.5}),
    ("get records from ds2 where tensor[0] = 0.1", {"rec1": 0.1}),
    # Index-less queries (default to first element)
    ("get records from ds2 where tensor > 0.2", {"rec2": 0.3}), # rec1 fails (0.1), rec2 matches (0.3)
    ("get records from ds2 where tensor = 0.1", {"rec1": 0.1}),
    # Empty tensor record (rec3) should not match positive conditions, or error
    ("get records from ds2 where tensor[0] > 0", {"rec1": 0.1, "rec2": 0.3}), # rec3's empty tensor fails index access
    ("get records from ds2 where tensor > 0", {"rec1": 0.1, "rec2": 0.3}), # rec3's empty tensor fails first element access
])
def test_tensor_filtering(basic_storage, query, expected_ids_values):
    agent = NQLAgent(basic_storage)
    result = agent.process_query(query)
    assert result["success"], f"Query '{query}' failed: {result.get('message', 'No message')}"
    assert result["count"] == len(expected_ids_values)

    found_ids = sorted([res["metadata"]["record_id"] for res in result["results"]])
    assert found_ids == sorted(expected_ids_values.keys())
    # Further checks on actual tensor values could be added if needed

def test_filter_tensor_index_out_of_bounds(basic_storage):
    agent = NQLAgent(basic_storage)
    query = "get records from ds2 where tensor[5] > 0"
    result = agent.process_query(query)
    assert result["success"]
    assert result["count"] == 0

def test_filter_tensor_non_numeric_value_in_query(basic_storage):
    agent = NQLAgent(basic_storage)
    query = "get records from ds2 where tensor[0] > 'abc'"
    result = agent.process_query(query)
    assert not result["success"]
    assert "Tensor value filtering currently only supports numeric comparisons" in result["message"]

# --- General Error and Edge Case Tests ---
def test_filter_meta_query_missing_dataset(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("find records from missing_ds where key = 'val'")
    assert not result["success"]
    assert "missing_ds" in result["message"].lower()

def test_filter_tensor_query_missing_dataset(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("show data from missing_ds where tensor[0] > 0")
    assert not result["success"]
    assert "missing_ds" in result["message"].lower()

def test_unsupported_query_format(basic_storage):
    agent = NQLAgent(basic_storage)
    result = agent.process_query("this is not a valid query")
    assert not result["success"]
    assert "Sorry, I couldn't understand that query" in result["message"]

def test_query_with_extra_spacing(basic_storage):
    agent = NQLAgent(basic_storage)
    # Test a metadata filter query with extra spaces
    query = "find  records   from  ds1  where   id  =  '007'"
    result = agent.process_query(query)
    assert result["success"]
    assert result["count"] == 1
    assert result["results"][0]["metadata"]["id"] == "007"

    # Test a count query
    query = "count    records    in    ds1"
    result = agent.process_query(query)
    assert result["success"]
    assert result["count"] == 4
