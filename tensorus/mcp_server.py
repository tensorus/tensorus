"""FastMCP server exposing Tensorus API endpoints as tools.

This module registers a set of MCP tools that proxy to the Tensorus FastAPI
backend.  Tools mirror the ones documented in the README under "Available
Tools" and return results as :class:`TextContent` objects.
"""

import argparse
import json
from typing import Any, Optional, Sequence

import httpx
from fastmcp import FastMCP
try:
    from fastmcp.tools import TextContent
except ImportError:  # pragma: no cover - support older fastmcp versions
    from dataclasses import dataclass

    @dataclass
    class TextContent:  # minimal fallback for tests
        type: str
        text: str

API_BASE_URL = "http://127.0.0.1:8000"

server = FastMCP(name="Tensorus FastMCP")


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


async def _get(path: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def _put(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{API_BASE_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


async def _delete(path: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


@server.tool()
async def save_tensor(
    dataset_name: str,
    tensor_shape: Sequence[int],
    tensor_dtype: str,
    tensor_data: Any,
    metadata: Optional[dict] = None,
) -> TextContent:
    """Save a tensor to a dataset."""
    payload = {
        "shape": list(tensor_shape),
        "dtype": tensor_dtype,
        "data": tensor_data,
        "metadata": metadata,
    }
    result = await _post(f"/datasets/{dataset_name}/ingest", payload)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def get_tensor(dataset_name: str, record_id: str) -> TextContent:
    """Retrieve a tensor by record ID."""
    result = await _get(f"/datasets/{dataset_name}/tensors/{record_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def execute_nql_query(query: str) -> TextContent:
    """Execute a Natural Query Language query."""
    result = await _post("/query", {"query": query})
    return TextContent(type="text", text=json.dumps(result))


# --- Dataset Management Tools ---

@server.tool(name="tensorus_list_datasets")
async def tensorus_list_datasets() -> TextContent:
    """List all available datasets."""
    result = await _get("/datasets")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_create_dataset")
async def tensorus_create_dataset(dataset_name: str) -> TextContent:
    """Create a new dataset."""
    result = await _post("/datasets/create", {"name": dataset_name})
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_delete_dataset")
async def tensorus_delete_dataset(dataset_name: str) -> TextContent:
    """Delete an existing dataset."""
    result = await _delete(f"/datasets/{dataset_name}")
    return TextContent(type="text", text=json.dumps(result))


# --- Tensor Management Tools ---

@server.tool(name="tensorus_ingest_tensor")
async def tensorus_ingest_tensor(
    dataset_name: str,
    tensor_shape: Sequence[int],
    tensor_dtype: str,
    tensor_data: Any,
    metadata: Optional[dict] = None,
) -> TextContent:
    """Ingest a new tensor into a dataset."""
    payload = {
        "shape": list(tensor_shape),
        "dtype": tensor_dtype,
        "data": tensor_data,
        "metadata": metadata,
    }
    result = await _post(f"/datasets/{dataset_name}/ingest", payload)
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_get_tensor_details")
async def tensorus_get_tensor_details(dataset_name: str, record_id: str) -> TextContent:
    """Retrieve tensor data and metadata."""
    result = await _get(f"/datasets/{dataset_name}/tensors/{record_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_delete_tensor")
async def tensorus_delete_tensor(dataset_name: str, record_id: str) -> TextContent:
    """Delete a tensor from a dataset."""
    result = await _delete(f"/datasets/{dataset_name}/tensors/{record_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_update_tensor_metadata")
async def tensorus_update_tensor_metadata(
    dataset_name: str,
    record_id: str,
    new_metadata: dict,
) -> TextContent:
    """Replace metadata for a specific tensor."""
    payload = {"new_metadata": new_metadata}
    result = await _put(f"/datasets/{dataset_name}/tensors/{record_id}/metadata", payload)
    return TextContent(type="text", text=json.dumps(result))


# --- Tensor Operation Tools ---

@server.tool(name="tensorus_apply_unary_operation")
async def tensorus_apply_unary_operation(operation: str, request_payload: dict) -> TextContent:
    """Apply a unary TensorOps operation (e.g., log, reshape)."""
    result = await _post(f"/ops/{operation}", request_payload)
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_apply_binary_operation")
async def tensorus_apply_binary_operation(operation: str, request_payload: dict) -> TextContent:
    """Apply a binary TensorOps operation (e.g., add, subtract)."""
    result = await _post(f"/ops/{operation}", request_payload)
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_apply_list_operation")
async def tensorus_apply_list_operation(operation: str, request_payload: dict) -> TextContent:
    """Apply a TensorOps list operation such as concatenate or stack."""
    result = await _post(f"/ops/{operation}", request_payload)
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_apply_einsum")
async def tensorus_apply_einsum(request_payload: dict) -> TextContent:
    """Apply an einsum operation."""
    result = await _post("/ops/einsum", request_payload)
    return TextContent(type="text", text=json.dumps(result))


@server.resource("resource://datasets", name="datasets", description="List of datasets")
async def datasets_resource() -> str:
    data = await _get("/datasets")
    return json.dumps(data.get("data", []))


def main() -> None:
    global API_BASE_URL

    parser = argparse.ArgumentParser(
        description="Run the Tensorus FastMCP server exposing dataset and tensor tools"
    )
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio", help="Transport protocol"
    )
    parser.add_argument(
        "--api-url", default=API_BASE_URL, help="Base URL of the running FastAPI backend"
    )
    args = parser.parse_args()

    API_BASE_URL = args.api_url.rstrip("/")

    server.run(args.transport)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
