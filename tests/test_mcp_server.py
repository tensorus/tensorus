import json
import pytest
import httpx

from tensorus import mcp_server

class DummyResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data
    def raise_for_status(self):
        pass


def make_mock_client(monkeypatch, method, url, payload, response, *, expected_params=None):
    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def post(self, u, json=None, params=None):
            assert method == 'post'
            assert u == url
            assert json == payload
            assert params == expected_params
            return DummyResponse(response)
        async def get(self, u, params=None):
            assert method == 'get'
            assert u == url
            assert params == expected_params
            return DummyResponse(response)
        async def put(self, u, json=None, params=None):
            assert method == 'put'
            assert u == url
            assert json == payload
            assert params == expected_params
            return DummyResponse(response)
        async def patch(self, u, json=None, params=None):
            assert method == 'patch'
            assert u == url
            assert json == payload
            assert params == expected_params
            return DummyResponse(response)
        async def delete(self, u, params=None):
            assert method == 'delete'
            assert u == url
            assert params == expected_params
            return DummyResponse(response)
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", MockAsyncClient)


def make_error_client(monkeypatch, method):
    class ErrorAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, u, json=None, params=None):
            assert method == "post"
            raise httpx.HTTPError("failed")

        async def get(self, u, params=None):
            assert method == "get"
            raise httpx.HTTPError("failed")

        async def put(self, u, json=None, params=None):
            assert method == "put"
            raise httpx.HTTPError("failed")

        async def patch(self, u, json=None, params=None):
            assert method == "patch"
            raise httpx.HTTPError("failed")

        async def delete(self, u, params=None):
            assert method == "delete"
            raise httpx.HTTPError("failed")

    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", ErrorAsyncClient)


@pytest.mark.asyncio
async def test_save_tensor(monkeypatch):
    payload = {
        "shape": [2, 2],
        "dtype": "float32",
        "data": [[1, 2], [3, 4]],
        "metadata": {"a": 1},
    }
    response = {"ok": True}
    url = f"{mcp_server.API_BASE_URL}/datasets/ds1/ingest"
    make_mock_client(monkeypatch, "post", url, payload, response)
    result = await mcp_server.save_tensor.fn("ds1", (2, 2), "float32", [[1, 2], [3, 4]], {"a": 1})
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_get_tensor(monkeypatch):
    response = {"record_id": "abc"}
    url = f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/abc"
    make_mock_client(monkeypatch, "get", url, None, response)
    result = await mcp_server.get_tensor.fn("ds1", "abc")
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_execute_nql_query(monkeypatch):
    response = {"results": []}
    url = f"{mcp_server.API_BASE_URL}/query"
    make_mock_client(monkeypatch, "post", url, {"query": "count"}, response)
    result = await mcp_server.execute_nql_query.fn("count")
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_dataset_tools(monkeypatch):
    create_resp = {"message": "ok"}
    list_resp = {"data": ["ds1"]}
    delete_resp = {"deleted": True}

    make_mock_client(
        monkeypatch,
        "post",
        f"{mcp_server.API_BASE_URL}/datasets/create",
        {"name": "ds1"},
        create_resp,
    )
    res_create = await mcp_server.tensorus_create_dataset.fn("ds1")
    assert json.loads(res_create.text) == create_resp

    make_mock_client(
        monkeypatch,
        "get",
        f"{mcp_server.API_BASE_URL}/datasets",
        None,
        list_resp,
    )
    res_list = await mcp_server.tensorus_list_datasets.fn()
    assert json.loads(res_list.text) == list_resp

    make_mock_client(
        monkeypatch,
        "delete",
        f"{mcp_server.API_BASE_URL}/datasets/ds1",
        None,
        delete_resp,
    )
    res_delete = await mcp_server.tensorus_delete_dataset.fn("ds1")
    assert json.loads(res_delete.text) == delete_resp


@pytest.mark.asyncio
async def test_tensor_tools(monkeypatch):
    ingest_payload = {
        "shape": [1],
        "dtype": "int32",
        "data": [1],
        "metadata": None,
    }
    ingest_resp = {"record_id": "r1"}
    make_mock_client(
        monkeypatch,
        "post",
        f"{mcp_server.API_BASE_URL}/datasets/ds1/ingest",
        ingest_payload,
        ingest_resp,
    )
    res_ingest = await mcp_server.tensorus_ingest_tensor.fn("ds1", [1], "int32", [1])
    assert json.loads(res_ingest.text) == ingest_resp

    details_resp = {"record_id": "r1", "data": [1]}
    make_mock_client(
        monkeypatch,
        "get",
        f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1",
        None,
        details_resp,
    )
    res_details = await mcp_server.tensorus_get_tensor_details.fn("ds1", "r1")
    assert json.loads(res_details.text) == details_resp

    delete_resp = {"deleted": True}
    make_mock_client(
        monkeypatch,
        "delete",
        f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1",
        None,
        delete_resp,
    )
    res_delete = await mcp_server.tensorus_delete_tensor.fn("ds1", "r1")
    assert json.loads(res_delete.text) == delete_resp

    update_payload = {"new_metadata": {"x": 1}}
    update_resp = {"updated": True}
    make_mock_client(
        monkeypatch,
        "put",
        f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1/metadata",
        update_payload,
        update_resp,
    )
    res_update = await mcp_server.tensorus_update_tensor_metadata.fn("ds1", "r1", {"x": 1})
    assert json.loads(res_update.text) == update_resp


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "func,operation,payload",
    [
        (mcp_server.tensorus_apply_unary_operation, "log", {"a": 1}),
        (mcp_server.tensorus_apply_binary_operation, "add", {"b": 2}),
        (mcp_server.tensorus_apply_list_operation, "concatenate", {"c": 3}),
    ],
)
async def test_tensor_ops(monkeypatch, func, operation, payload):
    resp = {"result": 0}
    make_mock_client(
        monkeypatch,
        "post",
        f"{mcp_server.API_BASE_URL}/ops/{operation}",
        payload,
        resp,
    )
    res = await func.fn(operation, payload)
    assert json.loads(res.text) == resp


@pytest.mark.asyncio
async def test_tensor_ops_einsum(monkeypatch):
    resp = {"result": 1}
    payload = {"equation": "i,i->", "operands": [1, 2]}
    make_mock_client(
        monkeypatch,
        "post",
        f"{mcp_server.API_BASE_URL}/ops/einsum",
        payload,
        resp,
    )
    res = await mcp_server.tensorus_apply_einsum.fn(payload)
    assert json.loads(res.text) == resp


@pytest.mark.asyncio
async def test_http_error_returns_textcontent(monkeypatch):
    make_error_client(monkeypatch, "post")
    res = await mcp_server.save_tensor.fn("ds1", [1], "int32", [1])
    assert json.loads(res.text) == {"error": "failed"}


# --- Tensor Descriptor Tools Tests ---

@pytest.mark.asyncio
async def test_create_tensor_descriptor(monkeypatch):
    payload = {"name": "test_tensor", "description": "A test tensor descriptor"}
    response = {"id": "tensor123", **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.create_tensor_descriptor.fn(descriptor_data=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Analytics Tools Tests ---

@pytest.mark.asyncio
async def test_analytics_get_co_occurring_tags_defaults(monkeypatch):
    response = [{"tag_group": ["tagA", "tagB"], "co_occurrence_count": 5}]
    url = f"{mcp_server.API_BASE_URL}/analytics/co_occurring_tags"
    # Default params for the tool are min_co_occurrence=2, limit=10
    expected_params = {"min_co_occurrence": 2, "limit": 10}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_co_occurring_tags.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_analytics_get_co_occurring_tags_custom(monkeypatch):
    min_co = 3
    limit_val = 5
    response = [{"tag_group": ["tagC", "tagD"], "co_occurrence_count": 4}]
    url = f"{mcp_server.API_BASE_URL}/analytics/co_occurring_tags"
    expected_params = {"min_co_occurrence": min_co, "limit": limit_val}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_co_occurring_tags.fn(min_co_occurrence=min_co, limit=limit_val)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_analytics_get_stale_tensors_defaults(monkeypatch):
    response = [{"tensor_id": "stale1", "last_accessed": "2022-01-01"}]
    url = f"{mcp_server.API_BASE_URL}/analytics/stale_tensors"
    # Default params threshold_days=90, limit=100
    expected_params = {"threshold_days": 90, "limit": 100}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_stale_tensors.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_analytics_get_stale_tensors_custom(monkeypatch):
    threshold = 30
    limit_val = 50
    response = [{"tensor_id": "stale2", "last_accessed": "2023-10-01"}]
    url = f"{mcp_server.API_BASE_URL}/analytics/stale_tensors"
    expected_params = {"threshold_days": threshold, "limit": limit_val}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_stale_tensors.fn(threshold_days=threshold, limit=limit_val)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_analytics_get_complex_tensors_defaults(monkeypatch):
    response = [{"tensor_id": "complex1", "parent_count": 3}]
    url = f"{mcp_server.API_BASE_URL}/analytics/complex_tensors"
    # Default params min_parent_count=None, min_transformation_steps=None, limit=100
    # API should receive only limit if others are None
    expected_params = {"limit": 100}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_complex_tensors.fn() # Call with defaults
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_analytics_get_complex_tensors_custom(monkeypatch):
    min_p_count = 2
    min_t_steps = 5
    limit_val = 10
    response = [{"tensor_id": "complex2", "parent_count": min_p_count, "transformation_steps": min_t_steps}]
    url = f"{mcp_server.API_BASE_URL}/analytics/complex_tensors"
    expected_params = {
        "min_parent_count": min_p_count,
        "min_transformation_steps": min_t_steps,
        "limit": limit_val
    }
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_complex_tensors.fn(
        min_parent_count=min_p_count, min_transformation_steps=min_t_steps, limit=limit_val
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_analytics_get_complex_tensors_some_none(monkeypatch):
    min_p_count = 3
    # min_transformation_steps is None
    limit_val = 20
    response = [{"tensor_id": "complex3"}]
    url = f"{mcp_server.API_BASE_URL}/analytics/complex_tensors"
    expected_params = {
        "min_parent_count": min_p_count,
        "limit": limit_val
        # min_transformation_steps should not be in params
    }
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.analytics_get_complex_tensors.fn(
        min_parent_count=min_p_count, limit=limit_val # min_transformation_steps defaults to None
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Management Tools Tests ---

@pytest.mark.asyncio
async def test_management_health_check(monkeypatch):
    response = {"status": "healthy"}
    url = f"{mcp_server.API_BASE_URL}/health"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.management_health_check.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_management_get_metrics(monkeypatch):
    response = {"metrics": {"active_connections": 10, "uptime_seconds": 3600}}
    url = f"{mcp_server.API_BASE_URL}/metrics"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.management_get_metrics.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Import/Export Tools Tests ---

@pytest.mark.asyncio
async def test_export_tensor_metadata_no_ids(monkeypatch):
    response = [{"id": "tensor1"}, {"id": "tensor2"}] # Example export data
    url = f"{mcp_server.API_BASE_URL}/tensors/export"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params={})
    result = await mcp_server.export_tensor_metadata.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_export_tensor_metadata_with_ids(monkeypatch):
    tensor_ids_str = "id1,id2,id3"
    response = [{"id": "id1"}, {"id": "id2"}] # Example filtered export data
    url = f"{mcp_server.API_BASE_URL}/tensors/export"
    expected_params = {"tensor_ids": tensor_ids_str}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.export_tensor_metadata.fn(tensor_ids_str=tensor_ids_str)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_import_tensor_metadata_default_strategy(monkeypatch):
    payload = {"metadata": [{"id": "tensor1", "name": "imported"}]}
    response = {"imported_count": 1, "skipped_count": 0, "errors": []}
    url = f"{mcp_server.API_BASE_URL}/tensors/import"
    expected_params = {"conflict_strategy": "skip"} # Default strategy
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=expected_params)
    result = await mcp_server.import_tensor_metadata.fn(import_data_payload=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_import_tensor_metadata_overwrite_strategy(monkeypatch):
    payload = {"metadata": [{"id": "tensor1", "name": "imported_overwrite"}]}
    conflict_strategy = "overwrite"
    response = {"imported_count": 1, "overwritten_count": 1, "errors": []}
    url = f"{mcp_server.API_BASE_URL}/tensors/import"
    expected_params = {"conflict_strategy": conflict_strategy}
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=expected_params)
    result = await mcp_server.import_tensor_metadata.fn(
        import_data_payload=payload, conflict_strategy=conflict_strategy
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Versioning and Lineage Tools Tests ---

@pytest.mark.asyncio
async def test_create_tensor_version(monkeypatch):
    tensor_id = "tensor123"
    payload = {"version_tag": "v2.0", "description": "New version"}
    response = {"tensor_id": tensor_id, "version_id": "version_abc", **payload}
    url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.create_tensor_version.fn(tensor_id=tensor_id, version_request=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_list_tensor_versions(monkeypatch):
    tensor_id = "tensor123"
    response = [{"version_id": "v_abc", "version_tag": "v1.0"}, {"version_id": "v_def", "version_tag": "v2.0"}]
    url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.list_tensor_versions.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_create_lineage_relationship(monkeypatch):
    payload = {
        "source_tensor_id": "source_id",
        "target_tensor_id": "target_id",
        "relationship_type": "derived_from"
    }
    response = {"id": "rel_123", **payload}
    url = f"{mcp_server.API_BASE_URL}/lineage/relationships/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.create_lineage_relationship.fn(relationship_request=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_parent_tensors(monkeypatch):
    tensor_id = "target_id"
    response = [{"tensor_id": "source_id", "relationship_type": "derived_from"}]
    url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/lineage/parents"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_parent_tensors.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_child_tensors(monkeypatch):
    tensor_id = "source_id"
    response = [{"tensor_id": "target_id", "relationship_type": "derived_from"}]
    url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/lineage/children"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_child_tensors.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Search and Aggregation Tools Tests ---

@pytest.mark.asyncio
async def test_search_tensors_no_fields(monkeypatch):
    text_query = "find important tensors"
    response = [{"id": "tensor1", "score": 0.9}, {"id": "tensor2", "score": 0.8}]
    url = f"{mcp_server.API_BASE_URL}/search/tensors/"
    expected_params = {"text_query": text_query}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.search_tensors.fn(text_query=text_query)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_search_tensors_with_fields(monkeypatch):
    text_query = "find in name"
    fields_to_search = "name,description"
    response = [{"id": "tensor3", "name": "target tensor", "score": 0.95}]
    url = f"{mcp_server.API_BASE_URL}/search/tensors/"
    expected_params = {"text_query": text_query, "fields_to_search": fields_to_search}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.search_tensors.fn(text_query=text_query, fields_to_search=fields_to_search)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_aggregate_tensors_no_agg_field(monkeypatch):
    group_by_field = "owner"
    agg_function = "count"
    response = [{"owner": "user1", "count": 10}, {"owner": "user2", "count": 5}]
    url = f"{mcp_server.API_BASE_URL}/aggregate/tensors/"
    expected_params = {"group_by_field": group_by_field, "agg_function": agg_function}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.aggregate_tensors.fn(group_by_field=group_by_field, agg_function=agg_function)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_aggregate_tensors_with_agg_field(monkeypatch):
    group_by_field = "data_type"
    agg_function = "avg"
    agg_field = "size_bytes"
    response = [{"data_type": "float32", "avg_size_bytes": 1024.5}, {"data_type": "int64", "avg_size_bytes": 2048.0}]
    url = f"{mcp_server.API_BASE_URL}/aggregate/tensors/"
    expected_params = {
        "group_by_field": group_by_field,
        "agg_function": agg_function,
        "agg_field": agg_field,
    }
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=expected_params)
    result = await mcp_server.aggregate_tensors.fn(
        group_by_field=group_by_field, agg_function=agg_function, agg_field=agg_field
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Extended Metadata Tools Tests ---

# Helper for Extended Metadata tests
EXTENDED_METADATA_TYPES = ["lineage", "computational", "quality", "relational", "usage"]

# Lineage Metadata Tests
@pytest.mark.asyncio
async def test_upsert_lineage_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"source": "test_source", "version": "v1"}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.upsert_lineage_metadata.fn(tensor_id=tensor_id, metadata_in=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_lineage_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"tensor_descriptor_id": tensor_id, "data": {"source": "test_source"}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_lineage_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_patch_lineage_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"version": "v2"}
    response = {"tensor_descriptor_id": tensor_id, "data": {"source": "test_source", "version": "v2"}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(monkeypatch, "patch", url, payload, response, expected_params=None)
    result = await mcp_server.patch_lineage_metadata.fn(tensor_id=tensor_id, updates=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_lineage_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"message": "Lineage metadata deleted successfully."} # Example message
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_lineage_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

# Computational Metadata Tests
@pytest.mark.asyncio
async def test_upsert_computational_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"algorithm": "test_algo"}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.upsert_computational_metadata.fn(tensor_id=tensor_id, metadata_in=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_computational_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"tensor_descriptor_id": tensor_id, "data": {"algorithm": "test_algo"}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_computational_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_patch_computational_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"framework": "PyTorch"}
    response = {"tensor_descriptor_id": tensor_id, "data": {"algorithm": "test_algo", "framework": "PyTorch"}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(monkeypatch, "patch", url, payload, response, expected_params=None)
    result = await mcp_server.patch_computational_metadata.fn(tensor_id=tensor_id, updates=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_computational_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"message": "Computational metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_computational_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

# Quality Metadata Tests
@pytest.mark.asyncio
async def test_upsert_quality_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"score": 0.99}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.upsert_quality_metadata.fn(tensor_id=tensor_id, metadata_in=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_quality_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"tensor_descriptor_id": tensor_id, "data": {"score": 0.99}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_quality_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_patch_quality_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"validated": True}
    response = {"tensor_descriptor_id": tensor_id, "data": {"score": 0.99, "validated": True}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(monkeypatch, "patch", url, payload, response, expected_params=None)
    result = await mcp_server.patch_quality_metadata.fn(tensor_id=tensor_id, updates=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_quality_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"message": "Quality metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_quality_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

# Relational Metadata Tests
@pytest.mark.asyncio
async def test_upsert_relational_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"collection_name": "coll1"}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.upsert_relational_metadata.fn(tensor_id=tensor_id, metadata_in=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_relational_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"tensor_descriptor_id": tensor_id, "data": {"collection_name": "coll1"}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_relational_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_patch_relational_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"related_ids": ["id1", "id2"]}
    response = {"tensor_descriptor_id": tensor_id, "data": {"collection_name": "coll1", "related_ids": ["id1", "id2"]}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(monkeypatch, "patch", url, payload, response, expected_params=None)
    result = await mcp_server.patch_relational_metadata.fn(tensor_id=tensor_id, updates=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_relational_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"message": "Relational metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_relational_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

# Usage Metadata Tests
@pytest.mark.asyncio
async def test_upsert_usage_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"access_count": 10}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.upsert_usage_metadata.fn(tensor_id=tensor_id, metadata_in=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_usage_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"tensor_descriptor_id": tensor_id, "data": {"access_count": 10}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_usage_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_patch_usage_metadata(monkeypatch):
    tensor_id = "tensor123"
    payload = {"last_accessed_by": "user_x"}
    response = {"tensor_descriptor_id": tensor_id, "data": {"access_count": 10, "last_accessed_by": "user_x"}}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(monkeypatch, "patch", url, payload, response, expected_params=None)
    result = await mcp_server.patch_usage_metadata.fn(tensor_id=tensor_id, updates=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_usage_metadata(monkeypatch):
    tensor_id = "tensor123"
    response = {"message": "Usage metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_usage_metadata.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_list_tensor_descriptors_no_params(monkeypatch):
    response = [{"id": "tensor123"}, {"id": "tensor456"}]
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params={}) # Empty dict for no params
    result = await mcp_server.list_tensor_descriptors.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_list_tensor_descriptors_with_params(monkeypatch):
    response = [{"id": "tensor123", "owner": "user1", "data_type": "float32"}]
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    params_to_send = {
        "owner": "user1",
        "data_type": "float32",
        "tags_contain": "tag1,tag2",
        "lineage.version": "v1.0",
        "lineage.source.type": "manual",
        "computational.algorithm": "algo1",
        "computational.hardware_info.gpu_model": "NVIDIA A100",
        "quality.confidence_score_gt": 0.95,
        "quality.noise_level_lt": 0.05,
        "relational.collection": "coll1",
        "relational.has_related_tensor_id": "uuid_str_123",
        "usage.last_accessed_before": "2023-01-01T00:00:00Z",
        "usage.used_by_app": "app_x",
        "name": "tensor_x",
        "description": "some desc",
        "min_dimensions": 3
    }
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=params_to_send)
    result = await mcp_server.list_tensor_descriptors.fn(
        owner="user1",
        data_type="float32",
        tags_contain="tag1,tag2",
        lineage_version="v1.0",
        lineage_source_type="manual",
        comp_algorithm="algo1",
        comp_gpu_model="NVIDIA A100",
        quality_confidence_gt=0.95,
        quality_noise_lt=0.05,
        rel_collection="coll1",
        rel_has_related_tensor_id="uuid_str_123",
        usage_last_accessed_before="2023-01-01T00:00:00Z",
        usage_used_by_app="app_x",
        name="tensor_x",
        description="some desc",
        min_dimensions=3
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_list_tensor_descriptors_new_params(monkeypatch):
    response = [{"id": "tensor999"}]
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    params = {"name": "t1", "description": "d1", "min_dimensions": 2}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=params)
    result = await mcp_server.list_tensor_descriptors.fn(name="t1", description="d1", min_dimensions=2)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_tensor_descriptor(monkeypatch):
    tensor_id = "tensor123"
    response = {"id": tensor_id, "name": "test_tensor"}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_tensor_descriptor.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_update_tensor_descriptor(monkeypatch):
    tensor_id = "tensor123"
    payload = {"description": "Updated description"}
    response = {"id": tensor_id, **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
    make_mock_client(monkeypatch, "put", url, payload, response, expected_params=None)
    result = await mcp_server.update_tensor_descriptor.fn(tensor_id=tensor_id, updates=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_tensor_descriptor(monkeypatch):
    tensor_id = "tensor123"
    response = {"message": f"Tensor descriptor {tensor_id} deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_tensor_descriptor.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

# --- Semantic Metadata Tools Tests ---

@pytest.mark.asyncio
async def test_create_semantic_metadata_for_tensor(monkeypatch):
    tensor_id = "tensor123"
    payload = {"name": "sem_meta_1", "value": "test_value", "type": "string"}
    response = {"id": "meta_id_1", "tensor_descriptor_id": tensor_id, **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
    make_mock_client(monkeypatch, "post", url, payload, response, expected_params=None)
    result = await mcp_server.create_semantic_metadata_for_tensor.fn(tensor_id=tensor_id, metadata_in=payload)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_get_all_semantic_metadata_for_tensor(monkeypatch):
    tensor_id = "tensor123"
    response = [
        {"id": "meta_id_1", "name": "sem_meta_1", "value": "test_value"},
        {"id": "meta_id_2", "name": "sem_meta_2", "value": 123},
    ]
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.get_all_semantic_metadata_for_tensor.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_update_named_semantic_metadata_for_tensor(monkeypatch):
    tensor_id = "tensor123"
    current_name = "sem_meta_1"
    payload = {"value": "updated_value"}
    response = {"id": "meta_id_1", "tensor_descriptor_id": tensor_id, "name": current_name, **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/{current_name}"
    make_mock_client(monkeypatch, "put", url, payload, response, expected_params=None)
    result = await mcp_server.update_named_semantic_metadata_for_tensor.fn(
        tensor_id=tensor_id, current_name=current_name, updates=payload
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response

@pytest.mark.asyncio
async def test_delete_named_semantic_metadata_for_tensor(monkeypatch):
    tensor_id = "tensor123"
    name = "sem_meta_1"
    response = {"message": f"Semantic metadata '{name}' for tensor descriptor '{tensor_id}' deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/{name}"
    make_mock_client(monkeypatch, "delete", url, None, response, expected_params=None)
    result = await mcp_server.delete_named_semantic_metadata_for_tensor.fn(tensor_id=tensor_id, name=name)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response
