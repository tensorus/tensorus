import json
import pytest

from tensorus import mcp_client
from tensorus.mcp_client import TensorusMCPClient, TextContent


class DummyFastClient:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return [TextContent(type="text", text=json.dumps({"ok": True, "name": name}))]


@pytest.mark.asyncio
async def test_create_dataset(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy") as client:
        result = await client.create_dataset("ds1")
    assert dummy.calls == [("tensorus_create_dataset", {"dataset_name": "ds1"})]
    assert result == {"ok": True, "name": "tensorus_create_dataset"}


@pytest.mark.asyncio
async def test_ingest_tensor(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy") as client:
        res = await client.ingest_tensor("ds", [1, 2], "float32", [1, 2], {"x": 1})
    assert dummy.calls[0][0] == "tensorus_ingest_tensor"
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_execute_nql_query(monkeypatch):
    dummy = DummyFastClient()
    monkeypatch.setattr(mcp_client, "FastMCPClient", lambda transport: dummy)
    async with TensorusMCPClient("dummy") as client:
        res = await client.execute_nql_query("count")
    assert dummy.calls == [("execute_nql_query", {"query": "count"})]
    assert res["name"] == "execute_nql_query"
