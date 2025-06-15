import json
import pytest

from tensorus import mcp_server

class DummyResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data
    def raise_for_status(self):
        pass


def make_mock_client(monkeypatch, method, url, payload, response):
    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def post(self, u, json=None):
            assert method == 'post'
            assert u == url
            assert json == payload
            return DummyResponse(response)
        async def get(self, u):
            assert method == 'get'
            assert u == url
            return DummyResponse(response)
        async def put(self, u, json=None):
            assert method == 'put'
            assert u == url
            assert json == payload
            return DummyResponse(response)
        async def delete(self, u):
            assert method == 'delete'
            assert u == url
            return DummyResponse(response)
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", MockAsyncClient)


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
