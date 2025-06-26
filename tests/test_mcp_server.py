import json
import sys
import pytest
import httpx
from typing import Optional  # Added for type hinting

from tensorus import mcp_server
from tensorus.mcp_server import MCP_AVAILABLE
from tensorus.config import settings  # Added import

pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE, reason="MCP dependencies (fastmcp, mcp) not available"
)


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def make_mock_client(
    monkeypatch,
    method,
    url,
    payload,
    response,
    *,
    expected_params=None,
    expected_headers: Optional[dict] = None,
    expected_timeout: float = mcp_server.HTTP_TIMEOUT,
):  # Added expected_headers and timeout
    class MockAsyncClient:
        def __init__(self, *, timeout=None, **kwargs):
            assert timeout == expected_timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(
            self, u, json=None, params=None, headers=None
        ):  # Added headers param
            assert method == "post"
            assert u == url
            assert json == payload
            # Handle None vs {} comparison for params
            if expected_params is None:
                assert params is None or params == {}
            else:
                assert params == expected_params
            # Handle None vs {} comparison for headers
            if expected_headers is None:
                assert headers is None or headers == {}
            else:
                assert headers == expected_headers
            return DummyResponse(response)

        async def get(self, u, params=None, headers=None):  # Added headers param
            assert method == "get"
            assert u == url
            # Handle None vs {} comparison for params
            if expected_params is None:
                assert params is None or params == {}
            else:
                assert params == expected_params
            # Handle None vs {} comparison for headers
            if expected_headers is None:
                assert headers is None or headers == {}
            else:
                assert headers == expected_headers
            return DummyResponse(response)

        async def put(
            self, u, json=None, params=None, headers=None
        ):  # Added headers param
            assert method == "put"
            assert u == url
            assert json == payload
            # Handle None vs {} comparison for params
            if expected_params is None:
                assert params is None or params == {}
            else:
                assert params == expected_params
            # Handle None vs {} comparison for headers
            if expected_headers is None:
                assert headers is None or headers == {}
            else:
                assert headers == expected_headers
            return DummyResponse(response)

        async def patch(
            self, u, json=None, params=None, headers=None
        ):  # Added headers param
            assert method == "patch"
            assert u == url
            assert json == payload
            # Handle None vs {} comparison for params
            if expected_params is None:
                assert params is None or params == {}
            else:
                assert params == expected_params
            # Handle None vs {} comparison for headers
            if expected_headers is None:
                assert headers is None or headers == {}
            else:
                assert headers == expected_headers
            return DummyResponse(response)

        async def delete(self, u, params=None, headers=None):  # Added headers param
            assert method == "delete"
            assert u == url
            # Handle None vs {} comparison for params
            if expected_params is None:
                assert params is None or params == {}
            else:
                assert params == expected_params
            # Handle None vs {} comparison for headers
            if expected_headers is None:
                assert headers is None or headers == {}
            else:
                assert headers == expected_headers
            return DummyResponse(response)

    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", MockAsyncClient)


def make_error_client(
    monkeypatch, method, expected_timeout: float = mcp_server.HTTP_TIMEOUT
):  # This function also needs to handle headers if we want to test errors with headers
    class ErrorAsyncClient:
        def __init__(self, *, timeout=None, **kwargs):
            assert timeout == expected_timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, u, json=None, params=None, headers=None):  # Added headers
            assert method == "post"
            raise httpx.HTTPError("failed")

        async def get(self, u, params=None, headers=None):  # Added headers
            assert method == "get"
            raise httpx.HTTPError("failed")

        async def put(self, u, json=None, params=None, headers=None):  # Added headers
            assert method == "put"
            raise httpx.HTTPError("failed")

        async def patch(self, u, json=None, params=None, headers=None):  # Added headers
            assert method == "patch"
            raise httpx.HTTPError("failed")

        async def delete(self, u, params=None, headers=None):  # Added headers
            assert method == "delete"
            raise httpx.HTTPError("failed")

    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", ErrorAsyncClient)


def make_status_client(monkeypatch, method, status_code):
    class StatusAsyncClient:
        def __init__(self, *, timeout=None, **kwargs):
            assert timeout == mcp_server.HTTP_TIMEOUT

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def _raise(self, u, json=None, params=None, headers=None):
            request = httpx.Request(method.upper(), u)
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError("err", request=request, response=response)

        async def post(self, u, json=None, params=None, headers=None):
            assert method == "post"
            await self._raise(u, json, params, headers)

        async def get(self, u, params=None, headers=None):
            assert method == "get"
            await self._raise(u, None, params, headers)

        async def put(self, u, json=None, params=None, headers=None):
            assert method == "put"
            await self._raise(u, json, params, headers)

        async def patch(self, u, json=None, params=None, headers=None):
            assert method == "patch"
            await self._raise(u, json, params, headers)

        async def delete(self, u, params=None, headers=None):
            assert method == "delete"
            await self._raise(u, None, params, headers)

    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", StatusAsyncClient)


@pytest.mark.asyncio
async def test_save_tensor_with_api_key(monkeypatch):
    test_api_key = "key123"
    payload = {
        "shape": [2, 2],
        "dtype": "float32",
        "data": [[1, 2], [3, 4]],
        "metadata": {"a": 1},
    }
    response = {"ok": True}
    url = f"{mcp_server.API_BASE_URL}/datasets/ds1/ingest"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.save_tensor.fn(
        "ds1", (2, 2), "float32", [[1, 2], [3, 4]], {"a": 1}, api_key=test_api_key
    )
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_save_tensor_with_global_key(monkeypatch):
    global_key = "globalKey789"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        payload = {
            "shape": [2, 2],
            "dtype": "float32",
            "data": [[1, 2], [3, 4]],
            "metadata": {"a": 1},
        }
        response = {"ok": True}
        url = f"{mcp_server.API_BASE_URL}/datasets/ds1/ingest"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.save_tensor.fn(
            "ds1", (2, 2), "float32", [[1, 2], [3, 4]], {"a": 1}
        )
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_save_tensor_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        payload = {
            "shape": [2, 2],
            "dtype": "float32",
            "data": [[1, 2], [3, 4]],
            "metadata": {"a": 1},
        }
        response = {"ok": True}
        url = f"{mcp_server.API_BASE_URL}/datasets/ds1/ingest"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.save_tensor.fn(
            "ds1", (2, 2), "float32", [[1, 2], [3, 4]], {"a": 1}
        )
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_tensor(
    monkeypatch,
):  # Assumes get_tensor does not require API key and none is globally set
    response = {"record_id": "abc"}
    url = f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/abc"
    make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
    result = await mcp_server.get_tensor.fn("ds1", "abc")
    assert json.loads(result.text) == response


# This duplicate test_get_tensor was removed by the previous partial application or should be removed.
# If it's still here, this diff will remove it. Assuming it was the one without expected_headers.


@pytest.mark.asyncio
async def test_execute_nql_query_with_api_key(monkeypatch):
    test_api_key = "nql_key"
    response = {"results": []}
    url = f"{mcp_server.API_BASE_URL}/query"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        {"query": "count"},
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.execute_nql_query.fn("count", api_key=test_api_key)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_execute_nql_query_with_global_key(monkeypatch):
    global_key = "nql_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        response = {"results": []}
        url = f"{mcp_server.API_BASE_URL}/query"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            {"query": "count"},
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.execute_nql_query.fn("count")
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_execute_nql_query_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        response = {"results": []}
        url = f"{mcp_server.API_BASE_URL}/query"
        make_mock_client(
            monkeypatch, "post", url, {"query": "count"}, response, expected_headers={}
        )
        result = await mcp_server.execute_nql_query.fn("count")
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_dataset_tools_with_api_keys(monkeypatch):
    test_api_key = "dataset_key"
    global_key = "dataset_global_key"

    create_resp = {"message": "ok"}
    list_resp = {"data": ["ds1"]}  # list_datasets is GET, not taking api_key directly
    delete_resp = {"deleted": True}

    # Test tensorus_create_dataset with direct API key
    make_mock_client(
        monkeypatch,
        "post",
        f"{mcp_server.API_BASE_URL}/datasets/create",
        {"name": "ds1"},
        create_resp,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    res_create = await mcp_server.tensorus_create_dataset.fn(
        "ds1", api_key=test_api_key
    )
    assert json.loads(res_create.text) == create_resp

    # Test tensorus_list_datasets (no API key, no global key)
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        make_mock_client(
            monkeypatch,
            "get",
            f"{mcp_server.API_BASE_URL}/datasets",
            None,
            list_resp,
            expected_headers={},  # Expect no headers
        )
        res_list = await mcp_server.tensorus_list_datasets.fn()
        assert json.loads(res_list.text) == list_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # Test tensorus_list_datasets (no API key, WITH global key)
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        make_mock_client(
            monkeypatch,
            "get",
            f"{mcp_server.API_BASE_URL}/datasets",
            None,
            list_resp,
            expected_headers={
                settings.API_KEY_HEADER_NAME: global_key
            },  # Expect global header
        )
        res_list_global = await mcp_server.tensorus_list_datasets.fn()
        assert json.loads(res_list_global.text) == list_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # Test tensorus_delete_dataset with global API key
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        make_mock_client(
            monkeypatch,
            "delete",
            f"{mcp_server.API_BASE_URL}/datasets/ds1",
            None,
            delete_resp,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        res_delete = await mcp_server.tensorus_delete_dataset.fn("ds1")  # No direct key
        assert json.loads(res_delete.text) == delete_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # Test tensorus_delete_dataset with no key (global is None)
    mcp_server.GLOBAL_API_KEY = None
    try:
        make_mock_client(
            monkeypatch,
            "delete",
            f"{mcp_server.API_BASE_URL}/datasets/ds2",  # Using ds2 to avoid collision if backend state was real
            None,
            delete_resp,
            expected_headers={},
        )
        res_delete_no_key = await mcp_server.tensorus_delete_dataset.fn("ds2")
        assert json.loads(res_delete_no_key.text) == delete_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_tensor_tools_with_api_keys(monkeypatch):
    test_api_key = "tensor_tool_key"
    global_key = "tensor_tool_global"

    ingest_payload = {
        "shape": [1],
        "dtype": "int32",
        "data": [1],
        "metadata": None,
    }
    ingest_resp = {"record_id": "r1"}
    details_resp = {"record_id": "r1", "data": [1]}  # get_tensor_details is GET
    delete_resp = {"deleted": True}
    update_payload = {"new_metadata": {"x": 1}}
    update_resp = {"updated": True}

    # tensorus_ingest_tensor with direct key
    make_mock_client(
        monkeypatch,
        "post",
        f"{mcp_server.API_BASE_URL}/datasets/ds1/ingest",
        ingest_payload,
        ingest_resp,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    res_ingest = await mcp_server.tensorus_ingest_tensor.fn(
        "ds1", [1], "int32", [1], api_key=test_api_key
    )
    assert json.loads(res_ingest.text) == ingest_resp

    # tensorus_get_tensor_details (GET, no direct key, no global key)
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        make_mock_client(
            monkeypatch,
            "get",
            f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1",
            None,
            details_resp,
            expected_headers={},
        )
        res_details = await mcp_server.tensorus_get_tensor_details.fn("ds1", "r1")
        assert json.loads(res_details.text) == details_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # tensorus_get_tensor_details (GET, no direct key, WITH global key)
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        make_mock_client(
            monkeypatch,
            "get",
            f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1",
            None,
            details_resp,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        res_details_global = await mcp_server.tensorus_get_tensor_details.fn(
            "ds1", "r1"
        )
        assert json.loads(res_details_global.text) == details_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # tensorus_delete_tensor with global key
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        make_mock_client(
            monkeypatch,
            "delete",
            f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1",
            None,
            delete_resp,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        res_delete = await mcp_server.tensorus_delete_tensor.fn(
            "ds1", "r1"
        )  # No direct key
        assert json.loads(res_delete.text) == delete_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # tensorus_update_tensor_metadata with no key (global is None)
    mcp_server.GLOBAL_API_KEY = None
    try:
        make_mock_client(
            monkeypatch,
            "put",
            f"{mcp_server.API_BASE_URL}/datasets/ds1/tensors/r1/metadata",
            update_payload,
            update_resp,
            expected_headers={},
        )
        res_update = await mcp_server.tensorus_update_tensor_metadata.fn(
            "ds1", "r1", {"x": 1}
        )  # No direct key
        assert json.loads(res_update.text) == update_resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "func_name,operation,payload",
    [
        ("tensorus_apply_unary_operation", "log", {"a": 1}),
        ("tensorus_apply_binary_operation", "add", {"b": 2}),
        ("tensorus_apply_list_operation", "concatenate", {"c": 3}),
    ],
)
async def test_tensor_ops_variants(monkeypatch, func_name, operation, payload):
    func_to_test = getattr(mcp_server, func_name)
    test_api_key = f"{operation}_key"
    global_key = f"{operation}_global_key"
    resp = {"result": 0}
    url = f"{mcp_server.API_BASE_URL}/ops/{operation}"

    # With direct API key
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        resp,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    res = await func_to_test.fn(operation, payload, api_key=test_api_key)
    assert json.loads(res.text) == resp

    # With global API key
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            resp,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        res_global = await func_to_test.fn(operation, payload)
        assert json.loads(res_global.text) == resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # No API key
    mcp_server.GLOBAL_API_KEY = None  # Ensure global is also None
    try:
        make_mock_client(monkeypatch, "post", url, payload, resp, expected_headers={})
        res_no_key = await func_to_test.fn(operation, payload)
        assert json.loads(res_no_key.text) == resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key  # Restore if it was something else before this specific test case


@pytest.mark.asyncio
async def test_tensor_ops_einsum_variants(monkeypatch):
    test_api_key = "einsum_key"
    global_key = "einsum_global_key"
    resp = {"result": 1}
    payload = {"equation": "i,i->", "operands": [1, 2]}
    url = f"{mcp_server.API_BASE_URL}/ops/einsum"

    # With direct API key
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        resp,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    res = await mcp_server.tensorus_apply_einsum.fn(payload, api_key=test_api_key)
    assert json.loads(res.text) == resp

    # With global API key
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            resp,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        res_global = await mcp_server.tensorus_apply_einsum.fn(payload)
        assert json.loads(res_global.text) == resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # No API key
    mcp_server.GLOBAL_API_KEY = None  # Ensure global is also None
    try:
        make_mock_client(monkeypatch, "post", url, payload, resp, expected_headers={})
        res_no_key = await mcp_server.tensorus_apply_einsum.fn(payload)
        assert json.loads(res_no_key.text) == resp
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_http_error_returns_textcontent(
    monkeypatch,
):  # Assumes no key needed for this error check or one is provided if error happens after auth
    make_error_client(monkeypatch, "post")
    # If save_tensor requires a key, this test might need to provide one,
    # or ensure global key is set, depending on when the error is raised.
    # For now, assuming error can happen even with empty headers for simplicity of this test.
    res = await mcp_server.save_tensor.fn("ds1", [1], "int32", [1])  # No key passed
    expected_text = 'Backend service is unreachable. Response: {"error": "Network error", "message": "failed"}'
    assert res.text == expected_text


@pytest.mark.asyncio
async def test_custom_http_timeout(monkeypatch):
    custom = 5.0
    original = mcp_server.HTTP_TIMEOUT
    mcp_server.HTTP_TIMEOUT = custom
    try:
        response = {"status": "ok"}
        url = f"{mcp_server.API_BASE_URL}/health"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={},
            expected_timeout=custom,
        )
        res = await mcp_server.management_health_check.fn()
        assert json.loads(res.text) == response
    finally:
        mcp_server.HTTP_TIMEOUT = original


def test_api_base_url_from_env(monkeypatch):
    env_url = "http://env.example"
    original = mcp_server.API_BASE_URL
    monkeypatch.setenv("TENSORUS_API_BASE_URL", env_url)
    monkeypatch.setattr(mcp_server.server, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["mcp_server.py"])
    try:
        mcp_server.main()
        assert mcp_server.API_BASE_URL == env_url
    finally:
        mcp_server.API_BASE_URL = original
        monkeypatch.delenv("TENSORUS_API_BASE_URL", raising=False)


def test_cli_overrides_env(monkeypatch):
    env_url = "http://env.example"
    cli_url = "http://cli.example"
    original = mcp_server.API_BASE_URL
    monkeypatch.setenv("TENSORUS_API_BASE_URL", env_url)
    monkeypatch.setattr(mcp_server.server, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["mcp_server.py", "--api-url", cli_url])
    try:
        mcp_server.main()
        assert mcp_server.API_BASE_URL == cli_url
    finally:
        mcp_server.API_BASE_URL = original
        monkeypatch.delenv("TENSORUS_API_BASE_URL", raising=False)


# --- Tensor Descriptor Tools Tests ---


@pytest.mark.asyncio
async def test_create_tensor_descriptor_with_api_key(monkeypatch):
    test_api_key = "desc_create_key"
    payload = {"name": "test_tensor", "description": "A test tensor descriptor"}
    response = {"id": "tensor123", **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_params=None,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.create_tensor_descriptor.fn(
        descriptor_data=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# The following tests are removed as they are now covered by _with_api_keys or _variants versions:
# - test_dataset_tools
# - test_tensor_tools
# - test_tensor_ops
# - test_tensor_ops_einsum


@pytest.mark.asyncio
async def test_post_unauthorized(monkeypatch):
    make_status_client(monkeypatch, "post", 401)
    res = await mcp_server.save_tensor.fn("ds1", [1], "int32", [1])
    # Extract JSON from wrapped response text
    json_part = res.text.split("Response: ", 1)[1]
    assert json.loads(json_part) == {"error": "API key required"}


@pytest.mark.asyncio
async def test_delete_forbidden(monkeypatch):
    make_status_client(monkeypatch, "delete", 403)
    res = await mcp_server.tensorus_delete_dataset.fn("ds1")
    # Extract JSON from wrapped response text
    json_part = res.text.split("Response: ", 1)[1]
    assert json.loads(json_part) == {"error": "Access forbidden"}


# --- Tensor Descriptor Tools Tests ---

# The old test_create_tensor_descriptor is removed by this diff as it's covered by _with_api_key, _with_global_key, _no_key variants.

# --- Analytics Tools Tests ---


@pytest.mark.asyncio
async def test_analytics_get_co_occurring_tags_defaults(monkeypatch):
    response = [{"tag_group": ["tagA", "tagB"], "co_occurrence_count": 5}]
    url = f"{mcp_server.API_BASE_URL}/analytics/co_occurring_tags"
    # Default params for the tool are min_co_occurrence=2, limit=10
    expected_params = {"min_co_occurrence": 2, "limit": 10}
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
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
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.analytics_get_co_occurring_tags.fn(
        min_co_occurrence=min_co, limit=limit_val
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_analytics_get_stale_tensors_defaults(monkeypatch):
    response = [{"tensor_id": "stale1", "last_accessed": "2022-01-01"}]
    url = f"{mcp_server.API_BASE_URL}/analytics/stale_tensors"
    # Default params threshold_days=90, limit=100
    expected_params = {"threshold_days": 90, "limit": 100}
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
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
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.analytics_get_stale_tensors.fn(
        threshold_days=threshold, limit=limit_val
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_analytics_get_complex_tensors_defaults(monkeypatch):
    response = [{"tensor_id": "complex1", "parent_count": 3}]
    url = f"{mcp_server.API_BASE_URL}/analytics/complex_tensors"
    # Default params min_parent_count=None, min_transformation_steps=None, limit=100
    # API should receive only limit if others are None
    expected_params = {"limit": 100}
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.analytics_get_complex_tensors.fn()  # Call with defaults
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_analytics_get_complex_tensors_custom(monkeypatch):
    min_p_count = 2
    min_t_steps = 5
    limit_val = 10
    response = [
        {
            "tensor_id": "complex2",
            "parent_count": min_p_count,
            "transformation_steps": min_t_steps,
        }
    ]
    url = f"{mcp_server.API_BASE_URL}/analytics/complex_tensors"
    expected_params = {
        "min_parent_count": min_p_count,
        "min_transformation_steps": min_t_steps,
        "limit": limit_val,
    }
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.analytics_get_complex_tensors.fn(
        min_parent_count=min_p_count,
        min_transformation_steps=min_t_steps,
        limit=limit_val,
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
        "limit": limit_val,
        # min_transformation_steps should not be in params
    }
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.analytics_get_complex_tensors.fn(
        min_parent_count=min_p_count,
        limit=limit_val,  # min_transformation_steps defaults to None
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Management Tools Tests ---


@pytest.mark.asyncio
async def test_management_health_check(monkeypatch):
    response = {"status": "healthy"}
    url = f"{mcp_server.API_BASE_URL}/health"
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=None,
        expected_headers={},
    )
    result = await mcp_server.management_health_check.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_management_get_metrics(monkeypatch):
    response = {"metrics": {"active_connections": 10, "uptime_seconds": 3600}}
    url = f"{mcp_server.API_BASE_URL}/metrics"
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=None,
        expected_headers={},
    )
    result = await mcp_server.management_get_metrics.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# --- Import/Export Tools Tests ---


@pytest.mark.asyncio
async def test_export_tensor_metadata_no_ids(monkeypatch):
    response = [{"id": "tensor1"}, {"id": "tensor2"}]  # Example export data
    url = f"{mcp_server.API_BASE_URL}/tensors/export"
    make_mock_client(
        monkeypatch, "get", url, None, response, expected_params={}, expected_headers={}
    )
    result = await mcp_server.export_tensor_metadata.fn()
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_export_tensor_metadata_with_ids(monkeypatch):
    tensor_ids_str = "id1,id2,id3"
    response = [{"id": "id1"}, {"id": "id2"}]  # Example filtered export data
    url = f"{mcp_server.API_BASE_URL}/tensors/export"
    expected_params = {"tensor_ids": tensor_ids_str}
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.export_tensor_metadata.fn(tensor_ids_str=tensor_ids_str)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_import_tensor_metadata_default_strategy(monkeypatch):
    payload = {"metadata": [{"id": "tensor1", "name": "imported"}]}
    response = {"imported_count": 1, "skipped_count": 0, "errors": []}
    url = f"{mcp_server.API_BASE_URL}/tensors/import"
    expected_params = {"conflict_strategy": "skip"}  # Default strategy
    # No key variant
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_params=expected_params,
            expected_headers={},
        )
        result = await mcp_server.import_tensor_metadata.fn(import_data_payload=payload)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_import_tensor_metadata_overwrite_strategy_with_api_key(monkeypatch):
    test_api_key = "import_key"
    payload = {"metadata": [{"id": "tensor1", "name": "imported_overwrite"}]}
    conflict_strategy = "overwrite"
    response = {"imported_count": 1, "overwritten_count": 1, "errors": []}
    url = f"{mcp_server.API_BASE_URL}/tensors/import"
    expected_params = {"conflict_strategy": conflict_strategy}
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_params=expected_params,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.import_tensor_metadata.fn(
        import_data_payload=payload,
        conflict_strategy=conflict_strategy,
        api_key=test_api_key,
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_import_tensor_metadata_skip_strategy_with_global_key(monkeypatch):
    global_key = "import_global_key"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        payload = {"metadata": [{"id": "tensor2", "name": "imported_skip_global"}]}
        conflict_strategy = "skip"
        response = {"imported_count": 1, "skipped_count": 0, "errors": []}
        url = f"{mcp_server.API_BASE_URL}/tensors/import"
        expected_params = {"conflict_strategy": conflict_strategy}
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_params=expected_params,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.import_tensor_metadata.fn(
            import_data_payload=payload, conflict_strategy=conflict_strategy
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


# --- Versioning and Lineage Tools Tests ---


@pytest.mark.asyncio
async def test_create_tensor_version_with_api_key(monkeypatch):
    test_api_key = "version_create_key"
    tensor_id = "tensor123"
    payload = {"version_tag": "v2.0", "description": "New version"}
    response = {"tensor_id": tensor_id, "version_id": "version_abc", **payload}
    url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_params=None,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.create_tensor_version.fn(
        tensor_id=tensor_id, version_request=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_create_tensor_version_with_global_key(monkeypatch):
    global_key = "version_create_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"version_tag": "v2.0_global", "description": "New global version"}
        response = {
            "tensor_id": tensor_id,
            "version_id": "version_abc_global",
            **payload,
        }
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.create_tensor_version.fn(
            tensor_id=tensor_id, version_request=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_create_tensor_version_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"version_tag": "v2.0_no_key", "description": "New no_key version"}
        response = {
            "tensor_id": tensor_id,
            "version_id": "version_abc_no_key",
            **payload,
        }
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.create_tensor_version.fn(
            tensor_id=tensor_id, version_request=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_list_tensor_versions_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = [
            {"version_id": "v_abc", "version_tag": "v1.0"},
            {"version_id": "v_def", "version_tag": "v2.0"},
        ]
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.list_tensor_versions.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_list_tensor_versions_with_global_key(monkeypatch):
    global_key = "list_versions_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = [{"version_id": "v_abc_global", "version_tag": "v1.0_global"}]
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.list_tensor_versions.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_create_lineage_relationship_with_api_key(monkeypatch):
    test_api_key = "lineage_create_key"
    payload = {
        "source_tensor_id": "source_id",
        "target_tensor_id": "target_id",
        "relationship_type": "derived_from",
    }
    response = {"id": "rel_123", **payload}
    url = f"{mcp_server.API_BASE_URL}/lineage/relationships/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_params=None,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.create_lineage_relationship.fn(
        relationship_request=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_list_tensor_versions(monkeypatch):
    tensor_id = "tensor123"
    response = [
        {"version_id": "v_abc", "version_tag": "v1.0"},
        {"version_id": "v_def", "version_tag": "v2.0"},
    ]
    url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/versions"
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=None)
    result = await mcp_server.list_tensor_versions.fn(tensor_id=tensor_id)
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_create_lineage_relationship_with_api_key(
    monkeypatch,
):  # Was already applied, keeping for context, will be identical if already there
    test_api_key = "lineage_create_key"
    payload = {
        "source_tensor_id": "source_id",
        "target_tensor_id": "target_id",
        "relationship_type": "derived_from",
    }
    response = {"id": "rel_123", **payload}
    url = f"{mcp_server.API_BASE_URL}/lineage/relationships/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_params=None,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.create_lineage_relationship.fn(
        relationship_request=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_create_lineage_relationship_with_global_key(monkeypatch):
    global_key = "lineage_create_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        payload = {
            "source_tensor_id": "source_global",
            "target_tensor_id": "target_global",
            "relationship_type": "derived_from_global",
        }
        response = {"id": "rel_global", **payload}
        url = f"{mcp_server.API_BASE_URL}/lineage/relationships/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.create_lineage_relationship.fn(
            relationship_request=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_create_lineage_relationship_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        payload = {
            "source_tensor_id": "source_no_key",
            "target_tensor_id": "target_no_key",
            "relationship_type": "derived_from_no_key",
        }
        response = {"id": "rel_no_key", **payload}
        url = f"{mcp_server.API_BASE_URL}/lineage/relationships/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.create_lineage_relationship.fn(
            relationship_request=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_parent_tensors_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "target_id"
        response = [{"tensor_id": "source_id", "relationship_type": "derived_from"}]
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/lineage/parents"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.get_parent_tensors.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_parent_tensors_with_global_key(monkeypatch):
    global_key = "parent_tensors_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "target_id_global"
        response = [
            {
                "tensor_id": "source_id_global",
                "relationship_type": "derived_from_global",
            }
        ]
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/lineage/parents"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_parent_tensors.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_child_tensors_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "source_id"
        response = [{"tensor_id": "target_id", "relationship_type": "derived_from"}]
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/lineage/children"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.get_child_tensors.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_child_tensors_with_global_key(monkeypatch):
    global_key = "child_tensors_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "source_id_global"
        response = [
            {
                "tensor_id": "target_id_global",
                "relationship_type": "derived_from_global",
            }
        ]
        url = f"{mcp_server.API_BASE_URL}/tensors/{tensor_id}/lineage/children"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_child_tensors.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key

    # --- Search and Aggregation Tools Tests ---
    assert json.loads(result.text) == response


# --- Search and Aggregation Tools Tests ---


@pytest.mark.asyncio
async def test_search_tensors_no_fields(monkeypatch):
    text_query = "find important tensors"
    response = [{"id": "tensor1", "score": 0.9}, {"id": "tensor2", "score": 0.8}]
    url = f"{mcp_server.API_BASE_URL}/search/tensors/"
    expected_params = {"text_query": text_query}
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
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
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.search_tensors.fn(
        text_query=text_query, fields_to_search=fields_to_search
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_aggregate_tensors_no_agg_field(monkeypatch):
    group_by_field = "owner"
    agg_function = "count"
    response = [{"owner": "user1", "count": 10}, {"owner": "user2", "count": 5}]
    url = f"{mcp_server.API_BASE_URL}/aggregate/tensors/"
    expected_params = {"group_by_field": group_by_field, "agg_function": agg_function}
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
    result = await mcp_server.aggregate_tensors.fn(
        group_by_field=group_by_field, agg_function=agg_function
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_aggregate_tensors_with_agg_field(monkeypatch):
    group_by_field = "data_type"
    agg_function = "avg"
    agg_field = "size_bytes"
    response = [
        {"data_type": "float32", "avg_size_bytes": 1024.5},
        {"data_type": "int64", "avg_size_bytes": 2048.0},
    ]
    url = f"{mcp_server.API_BASE_URL}/aggregate/tensors/"
    expected_params = {
        "group_by_field": group_by_field,
        "agg_function": agg_function,
        "agg_field": agg_field,
    }
    make_mock_client(
        monkeypatch,
        "get",
        url,
        None,
        response,
        expected_params=expected_params,
        expected_headers={},
    )
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
async def test_upsert_lineage_metadata_with_api_key(monkeypatch):
    test_api_key = "upsert_lineage_key"
    tensor_id = "tensor123"
    payload = {"source": "test_source", "version": "v1"}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.upsert_lineage_metadata.fn(
        tensor_id=tensor_id, metadata_in=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_upsert_lineage_metadata_with_global_key(monkeypatch):
    global_key = "upsert_lineage_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"source": "test_source_global", "version": "v1_global"}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.upsert_lineage_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_upsert_lineage_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"source": "test_source_no_key", "version": "v1_no_key"}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.upsert_lineage_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_lineage_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"source": "test_source"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
        make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
        result = await mcp_server.get_lineage_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_lineage_metadata_with_global_key(monkeypatch):
    global_key = "get_lineage_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"source": "test_source_global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_lineage_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_lineage_metadata_with_api_key(monkeypatch):
    test_api_key = "patch_lineage_key"
    tensor_id = "tensor123"
    payload = {"version": "v2"}
    response = {
        "tensor_descriptor_id": tensor_id,
        "data": {"source": "test_source", "version": "v2"},
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(
        monkeypatch,
        "patch",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.patch_lineage_metadata.fn(
        tensor_id=tensor_id, updates=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_patch_lineage_metadata_with_global_key(monkeypatch):
    global_key = "patch_lineage_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"version": "v2_global"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"source": "test_source", "version": "v2_global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
        make_mock_client(
            monkeypatch,
            "patch",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.patch_lineage_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_lineage_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"version": "v2_no_key"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"source": "test_source", "version": "v2_no_key"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
        make_mock_client(
            monkeypatch, "patch", url, payload, response, expected_headers={}
        )
        result = await mcp_server.patch_lineage_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_lineage_metadata_with_api_key(monkeypatch):
    test_api_key = "delete_lineage_key"
    tensor_id = "tensor123"
    response = {"message": "Lineage metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/lineage/"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_lineage_metadata.fn(
        tensor_id=tensor_id, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# Computational Metadata Tests
@pytest.mark.asyncio
async def test_upsert_computational_metadata_with_api_key(monkeypatch):
    test_api_key = "upsert_comp_key"
    tensor_id = "tensor123"
    payload = {"algorithm": "test_algo"}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.upsert_computational_metadata.fn(
        tensor_id=tensor_id, metadata_in=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_upsert_computational_metadata_with_global_key(monkeypatch):
    global_key = "upsert_comp_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"algorithm": "test_algo_global"}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.upsert_computational_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_upsert_computational_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"algorithm": "test_algo_no_key"}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.upsert_computational_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_computational_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"algorithm": "test_algo"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
        make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
        result = await mcp_server.get_computational_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_computational_metadata_with_global_key(monkeypatch):
    global_key = "get_comp_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"algorithm": "test_algo_global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_computational_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_computational_metadata_with_api_key(monkeypatch):
    test_api_key = "patch_comp_key"
    tensor_id = "tensor123"
    payload = {"framework": "PyTorch"}
    response = {
        "tensor_descriptor_id": tensor_id,
        "data": {"algorithm": "test_algo", "framework": "PyTorch"},
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(
        monkeypatch,
        "patch",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.patch_computational_metadata.fn(
        tensor_id=tensor_id, updates=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_patch_computational_metadata_with_global_key(monkeypatch):
    global_key = "patch_comp_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"framework": "PyTorch_global"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"algorithm": "test_algo", "framework": "PyTorch_global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
        make_mock_client(
            monkeypatch,
            "patch",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.patch_computational_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_computational_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"framework": "PyTorch_no_key"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"algorithm": "test_algo", "framework": "PyTorch_no_key"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
        make_mock_client(
            monkeypatch, "patch", url, payload, response, expected_headers={}
        )
        result = await mcp_server.patch_computational_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_computational_metadata_with_api_key(monkeypatch):
    test_api_key = "delete_comp_key"
    tensor_id = "tensor123"
    response = {"message": "Computational metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/computational/"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_computational_metadata.fn(
        tensor_id=tensor_id, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# Quality Metadata Tests
@pytest.mark.asyncio
async def test_upsert_quality_metadata_with_api_key(monkeypatch):
    test_api_key = "upsert_quality_key"
    tensor_id = "tensor123"
    payload = {"score": 0.99}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.upsert_quality_metadata.fn(
        tensor_id=tensor_id, metadata_in=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_upsert_quality_metadata_with_global_key(monkeypatch):
    global_key = "upsert_quality_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"score": 0.98}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.upsert_quality_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_upsert_quality_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"score": 0.97}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.upsert_quality_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_quality_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = {"tensor_descriptor_id": tensor_id, "data": {"score": 0.99}}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
        make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
        result = await mcp_server.get_quality_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_quality_metadata_with_global_key(monkeypatch):
    global_key = "get_quality_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"score": 0.99, "source": "global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_quality_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_quality_metadata_with_api_key(monkeypatch):
    test_api_key = "patch_quality_key"
    tensor_id = "tensor123"
    payload = {"validated": True}
    response = {
        "tensor_descriptor_id": tensor_id,
        "data": {"score": 0.99, "validated": True},
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(
        monkeypatch,
        "patch",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.patch_quality_metadata.fn(
        tensor_id=tensor_id, updates=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_patch_quality_metadata_with_global_key(monkeypatch):
    global_key = "patch_quality_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"validated": True, "source": "global"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"score": 0.99, **payload},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
        make_mock_client(
            monkeypatch,
            "patch",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.patch_quality_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_quality_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"validated": False}  # Different payload for distinction
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"score": 0.99, **payload},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
        make_mock_client(
            monkeypatch, "patch", url, payload, response, expected_headers={}
        )
        result = await mcp_server.patch_quality_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_quality_metadata_with_api_key(monkeypatch):
    test_api_key = "delete_quality_key"
    tensor_id = "tensor123"
    response = {"message": "Quality metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/quality/"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_quality_metadata.fn(
        tensor_id=tensor_id, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# Relational Metadata Tests
@pytest.mark.asyncio
async def test_upsert_relational_metadata_with_api_key(monkeypatch):
    test_api_key = "upsert_rel_key"
    tensor_id = "tensor123"
    payload = {"collection_name": "coll1"}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.upsert_relational_metadata.fn(
        tensor_id=tensor_id, metadata_in=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_upsert_relational_metadata_with_global_key(monkeypatch):
    global_key = "upsert_rel_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"collection_name": "coll1_global"}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.upsert_relational_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_upsert_relational_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"collection_name": "coll1_no_key"}
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.upsert_relational_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_relational_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"collection_name": "coll1"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
        make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
        result = await mcp_server.get_relational_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_relational_metadata_with_global_key(monkeypatch):
    global_key = "get_rel_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"collection_name": "coll1_global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_relational_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_relational_metadata_with_api_key(monkeypatch):
    test_api_key = "patch_rel_key"
    tensor_id = "tensor123"
    payload = {"related_ids": ["id1", "id2"]}
    response = {
        "tensor_descriptor_id": tensor_id,
        "data": {"collection_name": "coll1", "related_ids": ["id1", "id2"]},
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(
        monkeypatch,
        "patch",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.patch_relational_metadata.fn(
        tensor_id=tensor_id, updates=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_patch_relational_metadata_with_global_key(monkeypatch):
    global_key = "patch_rel_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"related_ids": ["id_global_1", "id_global_2"]}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"collection_name": "coll1", **payload},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
        make_mock_client(
            monkeypatch,
            "patch",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.patch_relational_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_relational_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"related_ids": ["id_no_key_1", "id_no_key_2"]}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"collection_name": "coll1", **payload},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
        make_mock_client(
            monkeypatch, "patch", url, payload, response, expected_headers={}
        )
        result = await mcp_server.patch_relational_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_relational_metadata_with_api_key(monkeypatch):
    test_api_key = "delete_rel_key"
    tensor_id = "tensor123"
    response = {"message": "Relational metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/relational/"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_relational_metadata.fn(
        tensor_id=tensor_id, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


# Usage Metadata Tests
@pytest.mark.asyncio
async def test_upsert_usage_metadata_with_api_key(monkeypatch):
    test_api_key = "upsert_usage_key"
    tensor_id = "tensor123"
    payload = {"access_count": 10}
    response = {"tensor_descriptor_id": tensor_id, "data": payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.upsert_usage_metadata.fn(
        tensor_id=tensor_id, metadata_in=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_upsert_usage_metadata_with_global_key(monkeypatch):
    global_key = "upsert_usage_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"access_count": 20}  # Different data
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.upsert_usage_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_upsert_usage_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"access_count": 30}  # Different data
        response = {"tensor_descriptor_id": tensor_id, "data": payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.upsert_usage_metadata.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_usage_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = {"tensor_descriptor_id": tensor_id, "data": {"access_count": 10}}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
        make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
        result = await mcp_server.get_usage_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_usage_metadata_with_global_key(monkeypatch):
    global_key = "get_usage_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"access_count": 10, "source": "global"},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_usage_metadata.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_usage_metadata_with_api_key(monkeypatch):
    test_api_key = "patch_usage_key"
    tensor_id = "tensor123"
    payload = {"last_accessed_by": "user_x"}
    response = {
        "tensor_descriptor_id": tensor_id,
        "data": {"access_count": 10, "last_accessed_by": "user_x"},
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(
        monkeypatch,
        "patch",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.patch_usage_metadata.fn(
        tensor_id=tensor_id, updates=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_patch_usage_metadata_with_global_key(monkeypatch):
    global_key = "patch_usage_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"last_accessed_by": "user_global"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"access_count": 10, **payload},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
        make_mock_client(
            monkeypatch,
            "patch",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.patch_usage_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_patch_usage_metadata_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"last_accessed_by": "user_no_key"}
        response = {
            "tensor_descriptor_id": tensor_id,
            "data": {"access_count": 10, **payload},
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
        make_mock_client(
            monkeypatch, "patch", url, payload, response, expected_headers={}
        )
        result = await mcp_server.patch_usage_metadata.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_usage_metadata_with_api_key(monkeypatch):
    test_api_key = "delete_usage_key"
    tensor_id = "tensor123"
    response = {"message": "Usage metadata deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/usage/"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_usage_metadata.fn(
        tensor_id=tensor_id, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_list_tensor_descriptors_no_params(monkeypatch):
    response = [{"id": "tensor123"}, {"id": "tensor456"}]
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    make_mock_client(
        monkeypatch, "get", url, None, response, expected_params={}
    )  # Empty dict for no params
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
        "min_dimensions": 3,
    }
    make_mock_client(
        monkeypatch, "get", url, None, response, expected_params=params_to_send
    )
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
        min_dimensions=3,
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_list_tensor_descriptors_new_params(monkeypatch):
    response = [{"id": "tensor999"}]
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/"
    params = {"name": "t1", "description": "d1", "min_dimensions": 2}
    make_mock_client(monkeypatch, "get", url, None, response, expected_params=params)
    result = await mcp_server.list_tensor_descriptors.fn(
        name="t1", description="d1", min_dimensions=2
    )
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
async def test_update_tensor_descriptor_with_api_key(monkeypatch):  # Renamed
    test_api_key = "update_desc_key"
    tensor_id = "tensor123"
    payload = {"description": "Updated description with API key"}
    response = {"id": tensor_id, **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
    make_mock_client(
        monkeypatch,
        "put",
        url,
        payload,
        response,
        expected_params=None,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.update_tensor_descriptor.fn(
        tensor_id=tensor_id, updates=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_update_tensor_descriptor_with_global_key(monkeypatch):
    global_key = "update_desc_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"description": "Updated description with Global Key"}
        response = {"id": tensor_id, **payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
        make_mock_client(
            monkeypatch,
            "put",
            url,
            payload,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.update_tensor_descriptor.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_update_tensor_descriptor_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"description": "Updated description no key"}
        response = {"id": tensor_id, **payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
        make_mock_client(
            monkeypatch,
            "put",
            url,
            payload,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.update_tensor_descriptor.fn(
            tensor_id=tensor_id, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_tensor_descriptor_with_api_key(monkeypatch):  # Renamed
    test_api_key = "delete_desc_key"
    tensor_id = "tensor123"
    response = {"message": f"Tensor descriptor {tensor_id} deleted successfully."}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_params=None,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_tensor_descriptor.fn(
        tensor_id=tensor_id, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_delete_tensor_descriptor_with_global_key(monkeypatch):
    global_key = "delete_desc_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123_global_delete"
        response = {"message": f"Tensor descriptor {tensor_id} deleted successfully."}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
        make_mock_client(
            monkeypatch,
            "delete",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.delete_tensor_descriptor.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_tensor_descriptor_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123_no_key_delete"
        response = {"message": f"Tensor descriptor {tensor_id} deleted successfully."}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}"
        make_mock_client(
            monkeypatch,
            "delete",
            url,
            None,
            response,
            expected_params=None,
            expected_headers={},
        )
        result = await mcp_server.delete_tensor_descriptor.fn(tensor_id=tensor_id)
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


# --- Semantic Metadata Tools Tests ---


@pytest.mark.asyncio
async def test_create_semantic_metadata_for_tensor_with_api_key(
    monkeypatch,
):  # Name kept from previous successful apply
    test_api_key = "sem_create_key"
    tensor_id = "tensor123"
    payload = {"name": "sem_meta_1", "value": "test_value", "type": "string"}
    response = {"id": "meta_id_1", "tensor_descriptor_id": tensor_id, **payload}
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
    make_mock_client(
        monkeypatch,
        "post",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.create_semantic_metadata_for_tensor.fn(
        tensor_id=tensor_id, metadata_in=payload, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_create_semantic_metadata_for_tensor_with_global_key(
    monkeypatch,
):  # Name kept from previous successful apply
    global_key = "sem_create_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        payload = {"name": "sem_meta_global", "value": "global_value"}
        response = {"id": "meta_id_global", **payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
        make_mock_client(
            monkeypatch,
            "post",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.create_semantic_metadata_for_tensor.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_create_semantic_metadata_for_tensor_no_key(
    monkeypatch,
):  # Name kept from previous successful apply
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        payload = {"name": "sem_meta_no_key", "value": "no_key_value"}
        response = {"id": "meta_id_no_key", **payload}
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
        make_mock_client(
            monkeypatch, "post", url, payload, response, expected_headers={}
        )
        result = await mcp_server.create_semantic_metadata_for_tensor.fn(
            tensor_id=tensor_id, metadata_in=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_all_semantic_metadata_for_tensor_no_key(
    monkeypatch,
):  # Name kept from previous successful apply
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        response = [
            {"id": "meta_id_1", "name": "sem_meta_1", "value": "test_value"},
            {"id": "meta_id_2", "name": "sem_meta_2", "value": 123},
        ]
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
        make_mock_client(monkeypatch, "get", url, None, response, expected_headers={})
        result = await mcp_server.get_all_semantic_metadata_for_tensor.fn(
            tensor_id=tensor_id
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_get_all_semantic_metadata_for_tensor_with_global_key(
    monkeypatch,
):  # Name kept from previous successful apply
    global_key = "sem_get_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        response = [{"id": "meta_id_1"}]
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/"
        make_mock_client(
            monkeypatch,
            "get",
            url,
            None,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.get_all_semantic_metadata_for_tensor.fn(
            tensor_id=tensor_id
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_update_named_semantic_metadata_for_tensor_with_api_key(monkeypatch):
    test_api_key = "update_named_sem_key"
    tensor_id = "tensor123"
    current_name = "sem_meta_1"
    payload = {"value": "updated_value_apikey"}
    response = {
        "id": "meta_id_1",
        "tensor_descriptor_id": tensor_id,
        "name": current_name,
        **payload,
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/{current_name}"
    make_mock_client(
        monkeypatch,
        "put",
        url,
        payload,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.update_named_semantic_metadata_for_tensor.fn(
        tensor_id=tensor_id,
        current_name=current_name,
        updates=payload,
        api_key=test_api_key,
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_update_named_semantic_metadata_for_tensor_with_global_key(monkeypatch):
    global_key = "update_named_sem_global"
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = global_key
    try:
        tensor_id = "tensor123"
        current_name = "sem_meta_1_global"
        payload = {"value": "updated_value_global"}
        response = {
            "id": "meta_id_global",
            "tensor_descriptor_id": tensor_id,
            "name": current_name,
            **payload,
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/{current_name}"
        make_mock_client(
            monkeypatch,
            "put",
            url,
            payload,
            response,
            expected_headers={settings.API_KEY_HEADER_NAME: global_key},
        )
        result = await mcp_server.update_named_semantic_metadata_for_tensor.fn(
            tensor_id=tensor_id, current_name=current_name, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_update_named_semantic_metadata_for_tensor_no_key(monkeypatch):
    original_global_key = mcp_server.GLOBAL_API_KEY
    mcp_server.GLOBAL_API_KEY = None
    try:
        tensor_id = "tensor123"
        current_name = "sem_meta_1_no_key"
        payload = {"value": "updated_value_no_key"}
        response = {
            "id": "meta_id_no_key",
            "tensor_descriptor_id": tensor_id,
            "name": current_name,
            **payload,
        }
        url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/{current_name}"
        make_mock_client(
            monkeypatch, "put", url, payload, response, expected_headers={}
        )
        result = await mcp_server.update_named_semantic_metadata_for_tensor.fn(
            tensor_id=tensor_id, current_name=current_name, updates=payload
        )
        assert isinstance(result, mcp_server.TextContent)
        assert json.loads(result.text) == response
    finally:
        mcp_server.GLOBAL_API_KEY = original_global_key


@pytest.mark.asyncio
async def test_delete_named_semantic_metadata_for_tensor_with_api_key(monkeypatch):
    test_api_key = "delete_named_sem_key"
    tensor_id = "tensor123"
    name = "sem_meta_1"
    response = {
        "message": f"Semantic metadata '{name}' for tensor descriptor '{tensor_id}' deleted successfully."
    }
    url = f"{mcp_server.API_BASE_URL}/tensor_descriptors/{tensor_id}/semantic/{name}"
    make_mock_client(
        monkeypatch,
        "delete",
        url,
        None,
        response,
        expected_headers={settings.API_KEY_HEADER_NAME: test_api_key},
    )
    result = await mcp_server.delete_named_semantic_metadata_for_tensor.fn(
        tensor_id=tensor_id, name=name, api_key=test_api_key
    )
    assert isinstance(result, mcp_server.TextContent)
    assert json.loads(result.text) == response


@pytest.mark.asyncio
async def test_prompt_functions(monkeypatch):
    assert (
        await mcp_server.ask_about_topic.fn("AI")
        == "Can you explain the concept of 'AI'?"
    )
    assert (
        await mcp_server.summarize_text.fn(text="abc", max_length=5)
        == "Summarize the following in 5 words:\n\nabc"
    )
    assert (
        await mcp_server.data_analysis_prompt.fn(data_uri="uri")
        == "Analyze the data at uri and report key insights."
    )

    dynamic_resp = {"info": 1}
    url = f"{mcp_server.API_BASE_URL}/tensors/xyz/metadata"
    # Assuming this test runs in a context where GLOBAL_API_KEY is None
    make_mock_client(monkeypatch, "get", url, None, dynamic_resp, expected_headers={})
    result = await mcp_server.dynamic_prompt.fn(record_id="xyz")
    assert "info" in result
