import pytest
from unittest.mock import MagicMock, patch, call
pytest.importorskip("psycopg2")
import psycopg2
from uuid import uuid4, UUID
from datetime import datetime
import json

from tensorus.metadata.postgres_storage import PostgresMetadataStorage
from tensorus.metadata.schemas import (
    TensorDescriptor, SemanticMetadata, DataType, StorageFormat, AccessControl, CompressionInfo,
    LineageMetadata, ComputationalMetadata, QualityMetadata, RelationalMetadata, UsageMetadata,
    LineageSource, LineageSourceType, ParentTensorLink # For constructing objects
)
from tensorus.metadata.schemas_iodata import TensorusExportData, TensorusExportEntry # For type hints

# --- Mocks ---

@pytest.fixture
def mock_pool():
    """Mocks the psycopg2 connection pool."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor # For 'with ... as cur:'

    pool = MagicMock(spec=psycopg2.pool.SimpleConnectionPool)
    pool.getconn.return_value = mock_conn
    pool.putconn.return_value = None # Doesn't need to do anything
    return pool, mock_cursor # Return cursor for assertions

@pytest.fixture
def pg_storage(mock_pool):
    """Provides a PostgresMetadataStorage instance with a mocked pool."""
    pool, _ = mock_pool
    # Temporarily patch SimpleConnectionPool in the module where PostgresMetadataStorage will call it
    with patch('psycopg2.pool.SimpleConnectionPool', return_value=pool):
        storage = PostgresMetadataStorage(dsn="postgresql://mockuser:mockpass@mockhost/mockdb")
    return storage


# --- TensorDescriptor Method Tests (Conceptual SQL Generation) ---

def test_pg_add_tensor_descriptor(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    now = datetime.utcnow()
    ac = AccessControl(read=["user1"], write=["owner"])
    ci = CompressionInfo(algorithm="zstd", level=3)

    td = TensorDescriptor(
        tensor_id=td_id, dimensionality=2, shape=[10,20], data_type=DataType.FLOAT32,
        storage_format=StorageFormat.RAW, creation_timestamp=now, last_modified_timestamp=now,
        owner="test_owner", access_control=ac, byte_size=1600, checksum="chk123",
        compression_info=ci, tags=["tag1", "tag2"], metadata={"key": "value"}
    )
    pg_storage.add_tensor_descriptor(td)

    expected_query_part = "INSERT INTO tensor_descriptors" # Check a part of the query
    # Check if execute was called, and inspect its arguments
    assert mock_cursor.execute.call_count == 1
    args, _ = mock_cursor.execute.call_args
    assert expected_query_part in args[0]

    # Check some parameters (order might be an issue if not using dict params in real code)
    # The current pg_storage.add_tensor_descriptor uses tuple params.
    # For Pydantic v2, model_dump_json() is used. For v1, .json()
    expected_params = (
        td_id, 2, [10,20], 'float32', 'raw', now, now, 'test_owner',
        ac.model_dump_json(), 1600, 'chk123', ci.model_dump_json(),
        ['tag1', 'tag2'], json.dumps({"key": "value"})
    )
    assert args[1] == expected_params


def test_pg_get_tensor_descriptor(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    now = datetime.utcnow()

    # Simulate a row returned from the database
    mock_db_row = {
        'tensor_id': td_id, 'dimensionality': 1, 'shape': [5], 'data_type': 'int32',
        'storage_format': 'numpy_npz', 'creation_timestamp': now, 'last_modified_timestamp': now,
        'owner': 'fetch_user', 'access_control': {"read": ["public"]}, 'byte_size': 20,
        'checksum': None, 'compression_info': None, 'tags': ['fetched'], 'metadata': {'source': 'db'}
    }
    mock_cursor.fetchone.return_value = mock_db_row

    descriptor = pg_storage.get_tensor_descriptor(td_id)

    assert mock_cursor.execute.call_count == 1
    query_args, _ = mock_cursor.execute.call_args
    assert "SELECT * FROM tensor_descriptors WHERE tensor_id = %s;" == query_args[0]
    assert query_args[1] == (td_id,)

    assert descriptor is not None
    assert descriptor.tensor_id == td_id
    assert descriptor.owner == 'fetch_user'
    assert descriptor.data_type == DataType.INT32
    assert descriptor.access_control.read == ["public"]
    assert descriptor.tags == ['fetched']

def test_pg_delete_tensor_descriptor(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    mock_cursor.rowcount = 1 # Simulate one row deleted

    result = pg_storage.delete_tensor_descriptor(td_id)

    assert result is True
    mock_cursor.execute.assert_called_once_with("DELETE FROM tensor_descriptors WHERE tensor_id = %s;", (td_id,))


# --- SemanticMetadata Method Tests ---
def test_pg_add_semantic_metadata(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    # Simulate parent TD exists (get_tensor_descriptor is called by add_semantic_metadata)
    now = datetime.utcnow()
    mock_cursor.fetchone.return_value = {
        'tensor_id': td_id,
        'dimensionality': 1,
        'shape': [1],
        'data_type': 'float32',
        'storage_format': 'raw',
        'creation_timestamp': now,
        'last_modified_timestamp': now,
        'owner': 'owner',
        'access_control': {},
        'byte_size': 4,
        'checksum': None,
        'compression_info': None,
        'tags': [],
        'metadata': {}
    }  # Minimal TD row with required fields

    sm = SemanticMetadata(tensor_id=td_id, name="purpose", description="for science")
    pg_storage.add_semantic_metadata(sm)

    # First call to get_tensor_descriptor, then to INSERT semantic
    assert mock_cursor.execute.call_count == 2
    insert_call_args, _ = mock_cursor.execute.call_args_list[1] # Second call

    assert "INSERT INTO semantic_metadata_entries" in insert_call_args[0]
    assert insert_call_args[1] == (td_id, "purpose", "for science")


# --- JSONB Extended Metadata Tests (Example with LineageMetadata) ---
def test_pg_add_jsonb_lineage_metadata(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    # Simulate parent TD exists for the _add_jsonb_metadata check
    now = datetime.utcnow()
    mock_cursor.fetchone.return_value = {
        'tensor_id': td_id,
        'dimensionality': 1,
        'shape': [1],
        'data_type': 'float32',
        'storage_format': 'raw',
        'creation_timestamp': now,
        'last_modified_timestamp': now,
        'owner': 'owner',
        'access_control': {},
        'byte_size': 4,
        'checksum': None,
        'compression_info': None,
        'tags': [],
        'metadata': {}
    }

    lm = LineageMetadata(tensor_id=td_id, version="v1.pg")
    pg_storage.add_lineage_metadata(lm)

    assert mock_cursor.execute.call_count == 2 # 1 for get_td, 1 for insert lineage
    insert_call_args, _ = mock_cursor.execute.call_args_list[1]

    assert "INSERT INTO lineage_metadata (tensor_id, data)" in insert_call_args[0]
    # Pydantic v2: lm.model_dump_json()
    assert insert_call_args[1] == {"tensor_id": td_id, "data": lm.model_dump_json()}


# --- Test list_tensor_descriptors with filters (SQL construction) ---
def test_pg_list_tensor_descriptors_with_filters(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool

    # Test with owner and lineage_version filters
    pg_storage.list_tensor_descriptors(owner="filter_owner", lineage_version="vFilter")

    assert mock_cursor.execute.call_count == 1
    query_args, _ = mock_cursor.execute.call_args
    sql_query = query_args[0]
    params = query_args[1]

    assert "FROM tensor_descriptors td" in sql_query
    assert "LEFT JOIN lineage_metadata lm ON td.tensor_id = lm.tensor_id" in sql_query
    assert "td.owner = %(owner)s" in sql_query
    assert "lm.data->>'version' = %(lineage_version)s" in sql_query
    assert params == {"owner": "filter_owner", "lineage_version": "vFilter"}


# --- Test Lineage Parent/Child SQL Construction ---
def test_pg_get_parent_tensor_ids_sql(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    mock_cursor.fetchall.return_value = [{'parent_id': str(uuid4())}, {'parent_id': str(uuid4())}] # Simulate DB response

    pg_storage.get_parent_tensor_ids(td_id)

    assert mock_cursor.execute.call_count == 1
    query_args, _ = mock_cursor.execute.call_args
    sql_query = query_args[0]
    params = query_args[1]

    assert "jsonb_array_elements(lm.data->'parent_tensors')" in sql_query
    assert "WHERE lm.tensor_id = %(tensor_id)s" in sql_query
    assert params == {"tensor_id": td_id}

def test_pg_get_child_tensor_ids_sql(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    td_id = uuid4()
    mock_cursor.fetchall.return_value = [{'tensor_id': uuid4()}, {'tensor_id': uuid4()}]

    pg_storage.get_child_tensor_ids(td_id)

    assert mock_cursor.execute.call_count == 1
    query_args, _ = mock_cursor.execute.call_args
    sql_query = query_args[0]
    params = query_args[1]

    assert "jsonb_array_elements(lm.data->'parent_tensors') AS parent" in sql_query
    assert "parent.value->>'tensor_id' = %(target_parent_id)s" in sql_query
    assert params == {"target_parent_id": str(td_id)}


# --- Test Search SQL Construction (Simplified) ---
def test_pg_search_tensor_descriptors_sql(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool

    pg_storage.search_tensor_descriptors("test_query", ["owner", "semantic.description", "lineage.version"])

    assert mock_cursor.execute.call_count == 1
    query_args, _ = mock_cursor.execute.call_args
    sql_query = query_args[0]
    params = query_args[1]

    assert "owner ILIKE %(text_query)s" in sql_query
    assert "description ILIKE %(text_query)s" in sql_query  # Assuming semantic_metadata_entries table alias sm
    assert "lm.data->>'version' ILIKE %(text_query)s" in sql_query # Assuming lineage_metadata alias lm
    assert "LEFT JOIN semantic_metadata_entries sm ON td.tensor_id = sm.tensor_id" in sql_query
    assert "LEFT JOIN lineage_metadata lm ON td.tensor_id = lm.tensor_id" in sql_query
    assert params == {"text_query": "%test_query%"}


# --- Test Aggregation SQL Construction (Simplified for count) ---
def test_pg_aggregate_tensor_descriptors_count_sql(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    mock_cursor.fetchall.return_value = [{'data_type': 'float32', 'count': 5}]

    with pytest.raises(ValueError):
        pg_storage.aggregate_tensor_descriptors("data_type", "count")

# Note: Full implementation of export/import for Postgres is complex and marked NotImplemented.
# Tests for those would require significant mocking or a live DB and are out of scope here.
# Similarly, health check and count methods are simple SQL and tested here conceptually.

def test_pg_check_health_ok(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    mock_cursor.execute.return_value = None # Simulate successful execution

    is_healthy, backend = pg_storage.check_health()
    assert is_healthy is True
    assert backend == "postgres"
    mock_cursor.execute.assert_called_once_with("SELECT 1;", None)

def test_pg_check_health_fail(pg_storage: PostgresMetadataStorage, mock_pool):
    pool, mock_cursor = mock_pool
    mock_cursor.execute.side_effect = psycopg2.Error("Connection failed")
    # Also need to mock getconn if the error happens there
    # pool.getconn.side_effect = psycopg2.Error("Pool failed")

    is_healthy, backend = pg_storage.check_health()
    assert is_healthy is False
    assert backend == "postgres"

def test_pg_get_tensor_descriptors_count(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    mock_cursor.fetchone.return_value = {'count': 123}

    count = pg_storage.get_tensor_descriptors_count()
    assert count == 123
    mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) as count FROM tensor_descriptors;", None)

def test_pg_get_extended_metadata_count(pg_storage: PostgresMetadataStorage, mock_pool):
    _, mock_cursor = mock_pool
    mock_cursor.fetchone.return_value = {'count': 42}

    count = pg_storage.get_extended_metadata_count("LineageMetadata")
    assert count == 42
    mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) as count FROM lineage_metadata;", None)

    count_sm = pg_storage.get_extended_metadata_count("SemanticMetadata") # Uses semantic_metadata_entries
    assert count_sm == 42 # Will use the same mock_fetchone for this test
    mock_cursor.execute.assert_called_with("SELECT COUNT(*) as count FROM semantic_metadata_entries;", None)

    count_unknown = pg_storage.get_extended_metadata_count("UnknownMeta")
    assert count_unknown == 0 # Should not call execute if table name not found
    # (call_count remains same as last successful call if no new execute occurs)
    # To be precise, check no *new* call with UnknownMeta's table.
    # Current logic prints warning and returns 0.

    # Ensure execute wasn't called again for "UnknownMeta" after the SemanticMetadata call
    assert mock_cursor.execute.call_count == 2  # Lineage_count and Semantic_count calls
