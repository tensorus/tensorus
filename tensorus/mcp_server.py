"""FastMCP server exposing Tensorus API endpoints as tools."""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional, Sequence

import httpx
from fastmcp import FastMCP
from fastmcp.tools import TextContent

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


@server.resource("resource://datasets", name="datasets", description="List of datasets")
async def datasets_resource() -> str:
    data = await _get("/datasets")
    return json.dumps(data.get("data", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tensorus FastMCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio", help="Transport protocol"
    )
    parser.add_argument(
        "--api-url", default=API_BASE_URL, help="Base URL of the running FastAPI backend"
    )
    args = parser.parse_args()

    global API_BASE_URL
    API_BASE_URL = args.api_url.rstrip("/")

    server.run(args.transport)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
