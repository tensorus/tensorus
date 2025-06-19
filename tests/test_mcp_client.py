import json
import pytest
from unittest.mock import patch

from tensorus import mcp_client
from tensorus.mcp_client import TensorusMCPClient, TextContent, DEFAULT_MCP_URL


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
    async with TensorusMCPClient("dummy") as client:
        result = await client.create_dataset("ds1")
    assert dummy.calls == [("tensorus_create_dataset", {"dataset_name": "ds1"})]
    assert result.success is True
    assert result.message == "Dataset ds1 created"


@pytest.mark.asyncio
async def test_ingest_tensor(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy") as client:
        res = await client.ingest_tensor("ds", [1, 2], "float32", [1, 2], {"x": 1})
    assert dummy.calls[0][0] == "tensorus_ingest_tensor"
    assert res.id == "tensor_id_123"
    assert res.status == "ingested"


@pytest.mark.asyncio
async def test_execute_nql_query(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy") as client:
        res = await client.execute_nql_query("count")
    assert dummy.calls == [("execute_nql_query", {"query": "count"})]
    assert res.results == ["result1", "result2"]


@pytest.mark.asyncio
async def test_additional_methods(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy") as client:
        td = await client.create_tensor_descriptor({"name": "desc"})
        lm = await client.get_lineage_metadata("tid")
        search = await client.search_tensors("q")
        version = await client.create_tensor_version("tid", {"tag": "v1"})
        export = await client.export_tensor_metadata()
        analytics = await client.analytics_get_complex_tensors()

    assert td.id == "td1"
    assert lm.tensor_descriptor_id == "tid"
    assert search == [{"id": "t1"}]
    assert version["version_id"] == "v1"
    assert export == [{"id": "t1"}]
    assert analytics == [{"tensor_id": "t2"}]

    called = [c[0] for c in dummy.calls]
    assert called == [
        "create_tensor_descriptor",
        "get_lineage_metadata",
        "search_tensors",
        "create_tensor_version",
        "export_tensor_metadata",
        "analytics_get_complex_tensors",
    ]


@patch('tensorus.mcp_client.StreamableHttpTransport')
async def test_from_http_auth_headers(mock_streamable_http_transport):
    test_url_no_slash = "http://fake-mcp.com/api"
    test_url_with_slash = "http://fake-mcp.com/api/" # Expected URL after processing
    test_token = "mysecrettoken"
    custom_header_name = "X-My-Token"

    # Test with custom token and header name
    TensorusMCPClient.from_http(
        url=test_url_no_slash, # Use URL without trailing slash for input
        auth_token=test_token,
        auth_header_name=custom_header_name
    )
    mock_streamable_http_transport.assert_called_with(
        url=test_url_with_slash,
        headers={custom_header_name: test_token}
    )

    # Reset mock for the next call
    mock_streamable_http_transport.reset_mock()

    # Test with token and default header name ("X-API-KEY")
    TensorusMCPClient.from_http(
        url=test_url_no_slash, # Use URL without trailing slash for input
        auth_token=test_token
        # auth_header_name defaults to "X-API-KEY"
    )
    mock_streamable_http_transport.assert_called_with(
        url=test_url_with_slash,
        headers={"X-API-KEY": test_token}
    )

    # Reset mock for the next call
    mock_streamable_http_transport.reset_mock()

    # Test without token (headers should be None)
    TensorusMCPClient.from_http(url=test_url_no_slash) # Use URL without trailing slash for input
    mock_streamable_http_transport.assert_called_with(
        url=test_url_with_slash,
        headers=None
    )

    # Test with default URL and no token
    mock_streamable_http_transport.reset_mock()
    TensorusMCPClient.from_http() # Uses DEFAULT_MCP_URL
    expected_default_url = DEFAULT_MCP_URL.rstrip("/") + "/"
    mock_streamable_http_transport.assert_called_with(
        url=expected_default_url,
        headers=None
    )
