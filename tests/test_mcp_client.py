import json
import pytest

from tensorus import mcp_client
from tensorus.mcp_client import (
    TensorusMCPClient, TextContent, MCPResponseError, FastMCPError,
    DatasetListResponse, IngestTensorResponse, TensorDetailsResponse, SemanticMetadataResponse,
    MCP_AVAILABLE
)

pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP dependencies (fastmcp, mcp) not available")


class DummyFastClient:
    def __init__(self):
        self.calls = []
        self.mock_tool_responses = {}
        # Default response for tools not specifically mocked
        self.default_response_data = {"ok": True, "message": "Default dummy response"}

    def set_mock_response(self, tool_name: str, response_config: dict):
        """Helper to set mock response for a specific tool."""
        self.mock_tool_responses[tool_name] = response_config

    def clear_mock_responses(self):
        self.mock_tool_responses = {}
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))

        mock_response_config = self.mock_tool_responses.get(name)

        if mock_response_config:
            if mock_response_config.get("raise_error"):
                raise mock_response_config["raise_error"]
            if "return_value" in mock_response_config: # Can be None or non-TextContent
                return mock_response_config["return_value"]
            if "json_text" in mock_response_config:
                return [TextContent(type="text", text=mock_response_config["json_text"])]
            if "response_data" in mock_response_config: # shortcut for valid json data
                 return [TextContent(type="text", text=json.dumps(mock_response_config["response_data"]))]

        # Fallback to original behavior for existing tests if no mock is set
        # This helps maintain compatibility with tests not yet updated to use set_mock_response
        response_data = {}
        if name == "tensorus_create_dataset":
            response_data = {"success": True, "message": "Dataset ds1 created"}
        elif name == "tensorus_ingest_tensor":
            # This specific response is used by test_ingest_tensor and test_call_json_fallback_ingest_tensor
            # if not overridden by a more specific mock.
            response_data = {"id": "tensor_id_123", "status": "ingested"}
        elif name == "execute_nql_query":
            response_data = {"results": ["result1", "result2"]}
        elif name == "create_tensor_descriptor":
            response_data = {"id": "td1", "name": "desc"}
        elif name == "get_lineage_metadata":
            response_data = {"tensor_descriptor_id": "tid", "data": {"x": 1}}
        elif name == "search_tensors":
            response_data = [{"id": "t1"}]
        elif name == "create_tensor_version":
            response_data = {"tensor_id": "tid", "version_id": "v1"}
        elif name == "export_tensor_metadata":
            response_data = [{"id": "t1"}]
        elif name == "analytics_get_complex_tensors":
            response_data = [{"tensor_id": "t2"}]
        else: # Default for any other unmocked tool
            response_data = self.default_response_data.copy()
            response_data["tool_name"] = name # Add tool name to default response
        return [TextContent(type="text", text=json.dumps(response_data))]


@pytest.fixture
def dummy_fast_client(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    return dummy


def _patch_transport(monkeypatch, capture_dict):
    class DummyTransport:
        def __init__(self, url, headers=None, **kwargs):
            capture_dict["url"] = url
            capture_dict["headers"] = headers

    monkeypatch.setattr(mcp_client, "StreamableHttpTransport", DummyTransport)


def test_from_http_with_token(dummy_fast_client, monkeypatch):
    captured = {}
    _patch_transport(monkeypatch, captured)
    TensorusMCPClient.from_http("https://example.com", token="sek")
    assert captured["headers"] == {"Authorization": "Bearer sek"}


def test_from_http_without_token(dummy_fast_client, monkeypatch):
    captured = {}
    _patch_transport(monkeypatch, captured)
    TensorusMCPClient.from_http("https://example.com")
    assert captured["headers"] is None

@pytest.mark.asyncio
async def test_create_dataset(dummy_fast_client):
    async with TensorusMCPClient("dummy") as client:
        result = await client.create_dataset("ds1")
    assert dummy_fast_client.calls == [("tensorus_create_dataset", {"dataset_name": "ds1"})]
    assert result.success is True
    assert result.message == "Dataset ds1 created"

@pytest.mark.asyncio
async def test_create_dataset_with_api_key(dummy_fast_client):
    async with TensorusMCPClient("dummy") as client:
        result = await client.create_dataset("ds1", api_key="KEY")
    assert dummy_fast_client.calls[-1] == ("tensorus_create_dataset", {"dataset_name": "ds1", "api_key": "KEY"})
    assert result.success is True


@pytest.mark.asyncio
async def test_ingest_tensor(dummy_fast_client):
    # This test will use the default behavior of DummyFastClient for "tensorus_ingest_tensor"
    # if no specific mock is set by other tests that might run before it.
    # The default provides: {"id": "tensor_id_123", "status": "ingested"}
    async with TensorusMCPClient("dummy") as client:
        res = await client.ingest_tensor("ds", [1, 2], "float32", [1, 2], {"x": 1})
    assert dummy_fast_client.calls[0][0] == "tensorus_ingest_tensor"
    assert res.id == "tensor_id_123"
    assert res.status == "ingested"

@pytest.mark.asyncio
async def test_ingest_tensor_with_api_key(dummy_fast_client):
    async with TensorusMCPClient("dummy") as client:
        res = await client.ingest_tensor("ds", [1,2], "float32", [1,2], {"x":1}, api_key="KEY")
    assert dummy_fast_client.calls[-1][1]["api_key"] == "KEY"
    assert res.id == "tensor_id_123"


@pytest.mark.asyncio
async def test_execute_nql_query(dummy_fast_client):
    async with TensorusMCPClient("dummy") as client:
        res = await client.execute_nql_query("count")
    assert dummy_fast_client.calls == [("execute_nql_query", {"query": "count"})]
    assert res.results == ["result1", "result2"]


@pytest.mark.asyncio
async def test_additional_methods(dummy_fast_client):
    async with TensorusMCPClient("dummy") as client:
        td = await client.create_tensor_descriptor({"name": "desc"})
        lm = await client.get_lineage_metadata("tid")
        search = await client.search_tensors("q")
        version = await client.create_tensor_version("tid", {"tag": "v1"})
        export = await client.export_tensor_metadata()
        analytics = await client.analytics_get_complex_tensors()

    assert td.id == "td1"
    assert lm.tensor_descriptor_id == "tid"
    assert search == [{"id": "t1"}] # search_tensors returns list[dict]
    assert version["version_id"] == "v1" # create_tensor_version returns dict
    assert export == [{"id": "t1"}] # export_tensor_metadata returns list[dict]
    assert analytics == [{"tensor_id": "t2"}] # analytics_get_complex_tensors returns list[dict]

    called = [c[0] for c in dummy_fast_client.calls]
    assert called == [
        "create_tensor_descriptor",
        "get_lineage_metadata",
        "search_tensors",
        "create_tensor_version",
        "export_tensor_metadata",
        "analytics_get_complex_tensors",
    ]

# --- New tests for list_datasets ---
@pytest.mark.asyncio
async def test_list_datasets_success(dummy_fast_client):
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"response_data": {"datasets": ["ds1", "ds2"]}})
    async with TensorusMCPClient("dummy") as client:
        result = await client.list_datasets()
    assert result == ["ds1", "ds2"]

@pytest.mark.asyncio
async def test_list_datasets_fallback_success(dummy_fast_client):
    # This tests the fallback logic in _call_json for DatasetListResponse
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"response_data": {"data": ["ds1", "ds2"], "other_key": "value"}})
    async with TensorusMCPClient("dummy") as client:
        result = await client.list_datasets()
    assert result == ["ds1", "ds2"]

@pytest.mark.asyncio
async def test_list_datasets_returns_none(dummy_fast_client):
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"return_value": None})
    async with TensorusMCPClient("dummy") as client:
        result = await client.list_datasets()
    assert result == []

@pytest.mark.asyncio
async def test_list_datasets_invalid_json_model(dummy_fast_client):
    # This response does not match DatasetListResponse schema or the fallback structure
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"response_data": {"items": ["ds1"]}})
    async with TensorusMCPClient("dummy") as client:
        with pytest.raises(MCPResponseError) as excinfo:
            await client.list_datasets()
        assert "Failed to parse response" in str(excinfo.value)
        assert "Validation Error" in str(excinfo.value)

@pytest.mark.asyncio
async def test_list_datasets_empty_response(dummy_fast_client):
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"response_data": {"datasets": []}})
    async with TensorusMCPClient("dummy") as client:
        result = await client.list_datasets()
    assert result == []

# --- New tests for get_all_semantic_metadata_for_tensor ---
@pytest.mark.asyncio
async def test_get_all_semantic_metadata_success(dummy_fast_client):
    mock_data = [{"id": "m1", "tensor_descriptor_id": "t1"}, {"id": "m2", "tensor_descriptor_id": "t1"}]
    dummy_fast_client.set_mock_response("get_all_semantic_metadata_for_tensor", {"response_data": mock_data})
    async with TensorusMCPClient("dummy") as client:
        result = await client.get_all_semantic_metadata_for_tensor("t1")
    assert len(result) == 2
    assert isinstance(result[0], SemanticMetadataResponse)
    assert result[0].id == "m1"
    assert result[1].id == "m2"

@pytest.mark.asyncio
async def test_get_all_semantic_metadata_empty_list(dummy_fast_client):
    dummy_fast_client.set_mock_response("get_all_semantic_metadata_for_tensor", {"response_data": []})
    async with TensorusMCPClient("dummy") as client:
        result = await client.get_all_semantic_metadata_for_tensor("t1")
    assert result == []

@pytest.mark.asyncio
async def test_get_all_semantic_metadata_returns_none(dummy_fast_client):
    dummy_fast_client.set_mock_response("get_all_semantic_metadata_for_tensor", {"return_value": None})
    async with TensorusMCPClient("dummy") as client:
        result = await client.get_all_semantic_metadata_for_tensor("t1")
    assert result == []

@pytest.mark.asyncio
async def test_get_all_semantic_metadata_contains_invalid_item(dummy_fast_client, caplog):
    mock_data = [{"id": "m1", "tensor_descriptor_id": "t1"}, "not_a_dict", {"invalid_field": "val", "id": "m3", "tensor_descriptor_id": "t1"}]
    dummy_fast_client.set_mock_response("get_all_semantic_metadata_for_tensor", {"response_data": mock_data})
    async with TensorusMCPClient("dummy") as client:
        result = await client.get_all_semantic_metadata_for_tensor("t1")

    assert len(result) == 2 # m1 and m3 are valid (m3 has extra field but required fields are present)
    assert result[0].id == "m1"
    assert result[1].id == "m3"

    # Check logs for warnings/errors about invalid items
    assert "Item in list for get_all_semantic_metadata_for_tensor is not a dict" in caplog.text


@pytest.mark.asyncio
async def test_get_all_semantic_metadata_not_a_list(dummy_fast_client, caplog):
    dummy_fast_client.set_mock_response("get_all_semantic_metadata_for_tensor", {"response_data": {"oops": "not a list"}})
    async with TensorusMCPClient("dummy") as client:
        result = await client.get_all_semantic_metadata_for_tensor("t1")
    assert result == []
    assert "get_all_semantic_metadata_for_tensor for t1 received unexpected data type: <class 'dict'>. Expected list." in caplog.text


# --- New tests for _call_json error handling (via various client methods) ---
@pytest.mark.asyncio
async def test_call_json_json_decode_error(dummy_fast_client):
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"json_text": "invalid json"})
    async with TensorusMCPClient("dummy") as client:
        with pytest.raises(MCPResponseError) as excinfo:
            await client.list_datasets()
    assert "Failed to decode JSON response" in str(excinfo.value)

@pytest.mark.asyncio
async def test_call_json_unexpected_content_type(dummy_fast_client):
    # Return a list containing a tuple, not TextContent
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"return_value": [("not", "text_content")]})
    async with TensorusMCPClient("dummy") as client:
        with pytest.raises(MCPResponseError) as excinfo:
            await client.list_datasets()
    assert "Unexpected content type: <class 'tuple'>" in str(excinfo.value)

@pytest.mark.asyncio
async def test_call_json_fastmcp_error(dummy_fast_client):
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"raise_error": FastMCPError("some MCP error")})
    async with TensorusMCPClient("dummy") as client:
        with pytest.raises(MCPResponseError) as excinfo:
            await client.list_datasets()
    assert "Tool call failed" in str(excinfo.value)
    assert "some MCP error" in str(excinfo.value)

@pytest.mark.asyncio
async def test_call_json_fastmcp_error_auth_hint(dummy_fast_client, caplog):
    # Test the error path and implicitly the logging change (though log content check is tricky here)
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"raise_error": FastMCPError("unauthorized access")})
    async with TensorusMCPClient("dummy") as client:
        with pytest.raises(MCPResponseError) as excinfo:
            await client.list_datasets()
    assert "Tool call failed" in str(excinfo.value)
    assert "unauthorized access" in str(excinfo.value)
    # Check if the hint was logged (best effort)
    assert "Hint: This might be related to missing or incorrect API keys" in caplog.text


@pytest.mark.asyncio
async def test_call_json_fallback_ingest_tensor(dummy_fast_client):
    # This tests the specific fallback for IngestTensorResponse in _call_json
    mock_api_response = {"success": True, "data": {"record_id": "xyz123"}}
    dummy_fast_client.set_mock_response("tensorus_ingest_tensor", {"response_data": mock_api_response})
    async with TensorusMCPClient("dummy") as client:
        result = await client.ingest_tensor("dataset_name", [1], "f32", [1.0])
    assert isinstance(result, IngestTensorResponse)
    assert result.id == "xyz123"
    assert result.status == "ingested"

@pytest.mark.asyncio
async def test_call_json_fallback_tensor_details(dummy_fast_client):
    # This tests the specific fallback for TensorDetailsResponse in _call_json
    mock_api_response = {
        "record_id": "abc",
        "shape": [1,2],
        "dtype": "f32",
        "data": [1.0, 2.0],
        "metadata": {"info":"test"}
    }
    dummy_fast_client.set_mock_response("tensorus_get_tensor_details", {"response_data": mock_api_response})
    async with TensorusMCPClient("dummy") as client:
        result = await client.get_tensor_details("dataset_name", "abc")
    assert isinstance(result, TensorDetailsResponse)
    assert result.id == "abc"
    assert result.shape == [1,2]
    assert result.dtype == "f32"
    assert result.data == [1.0, 2.0]
    assert result.metadata == {"info":"test"}

@pytest.mark.asyncio
async def test_call_json_validation_error_after_fallbacks(dummy_fast_client):
    # Test case where DatasetListResponse fallback is attempted but fails validation
    # The structure `{"data": "not_a_list"}` will make `isinstance(original_data_for_specific_parse['data'], list)` fail
    # inside the DatasetListResponse specific handler. This should then re-raise primary_validation_error.
    dummy_fast_client.set_mock_response("tensorus_list_datasets", {"response_data": {"data": "not_a_list"}})
    async with TensorusMCPClient("dummy") as client:
        with pytest.raises(MCPResponseError) as excinfo:
            await client.list_datasets()
    assert "Failed to parse response for tool 'tensorus_list_datasets'" in str(excinfo.value)
    assert "Validation Error" in str(excinfo.value)
    # The error message from Pydantic might mention 'datasets' field not being a valid list or similar
    # For example: "1 validation error for DatasetListResponse\ndatasets\n  value is not a valid list (type=type_error.list)"
    # or if the fallback itself fails on `{"datasets": data['data']}` it might be:
    # "value is not a valid list" referring to the `datasets` field of `DatasetListResponse`
    assert "datasets" in str(excinfo.value).lower() # Check that the error is about the 'datasets' field

# Ensure existing tests that might rely on old DummyFastClient behavior still pass or are adapted.
# test_ingest_tensor, for example, relied on a simple hardcoded response.
# The new DummyFastClient's default behavior for "tensorus_ingest_tensor" still provides
# {"id": "tensor_id_123", "status": "ingested"}, so test_ingest_tensor should still pass.
