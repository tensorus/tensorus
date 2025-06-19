import json
import pytest

from tensorus import mcp_client
from tensorus.mcp_client import TensorusMCPClient, TextContent

TEST_API_KEY = "client_test_key_456"

class DummyFastClient:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        response_data = {}
        if name == "tensorus_create_dataset":
            response_data = {"success": True, "message": "Dataset ds1 created"}
        elif name == "tensorus_ingest_tensor":
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
        # Fallback for other tools if any are called by this dummy client in other tests
        else:
            response_data = {"ok": True, "tool_name": name}
        return [TextContent(type="text", text=json.dumps(response_data))]


@pytest.mark.asyncio
async def test_create_dataset(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy", api_key=TEST_API_KEY) as client:
        result = await client.create_dataset("ds1")
    assert dummy.calls == [("tensorus_create_dataset", {"dataset_name": "ds1", "api_key": TEST_API_KEY})]
    assert result.success is True
    assert result.message == "Dataset ds1 created"


@pytest.mark.asyncio
async def test_ingest_tensor(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy", api_key=TEST_API_KEY) as client:
        res = await client.ingest_tensor("ds", [1, 2], "float32", [1, 2], {"x": 1})

    expected_args = {
        "dataset_name": "ds",
        "tensor_shape": [1, 2],
        "tensor_dtype": "float32",
        "tensor_data": [1, 2],
        "metadata": {"x": 1},
        "api_key": TEST_API_KEY
    }
    assert dummy.calls == [("tensorus_ingest_tensor", expected_args)]
    assert res.id == "tensor_id_123"
    assert res.status == "ingested"


@pytest.mark.asyncio
async def test_execute_nql_query(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy", api_key=TEST_API_KEY) as client:
        res = await client.execute_nql_query("count")
    assert dummy.calls == [("execute_nql_query", {"query": "count", "api_key": TEST_API_KEY})]
    assert res.results == ["result1", "result2"]


@pytest.mark.asyncio
async def test_create_dataset_no_api_key(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    # Instantiate client without API key
    async with TensorusMCPClient("dummy") as client:
        result = await client.create_dataset("ds2")
    # Assert that api_key is NOT in the arguments
    assert dummy.calls == [("tensorus_create_dataset", {"dataset_name": "ds2"})]
    assert result.success is True


@pytest.mark.asyncio
async def test_additional_methods(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy", api_key=TEST_API_KEY) as client:
        td = await client.create_tensor_descriptor({"name": "desc"})
        lm = await client.get_lineage_metadata("tid")
        search = await client.search_tensors("q")
        version = await client.create_tensor_version("tid", {"tag": "v1"})
        export = await client.export_tensor_metadata() # No specific args other than api_key
        analytics = await client.analytics_get_complex_tensors() # Default args + api_key

    assert td.id == "td1"
    assert lm.tensor_descriptor_id == "tid"
    assert search == [{"id": "t1"}] # Assuming search_tensors returns list directly
    assert version["version_id"] == "v1"
    assert export == [{"id": "t1"}] # Assuming export_tensor_metadata returns list
    assert analytics == [{"tensor_id": "t2"}] # Assuming analytics_get_complex_tensors returns list

    expected_calls = [
        ("create_tensor_descriptor", {"descriptor_data": {"name": "desc"}, "api_key": TEST_API_KEY}),
        ("get_lineage_metadata", {"tensor_id": "tid", "api_key": TEST_API_KEY}),
        ("search_tensors", {"text_query": "q", "api_key": TEST_API_KEY}),
        ("create_tensor_version", {"tensor_id": "tid", "version_request": {"tag": "v1"}, "api_key": TEST_API_KEY}),
        ("export_tensor_metadata", {"api_key": TEST_API_KEY}),
        ("analytics_get_complex_tensors", {"limit": 100, "api_key": TEST_API_KEY}), # limit is default
    ]
    assert dummy.calls == expected_calls
