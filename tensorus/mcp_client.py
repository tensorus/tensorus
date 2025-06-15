"""Tensorus MCP client built on fastmcp.Client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from fastmcp.client import Client as FastMCPClient
try:
    from fastmcp.tools import TextContent
except Exception:  # pragma: no cover - minimal fallback
    @dataclass
    class TextContent:  # type: ignore
        type: str
        text: str


class TensorusMCPClient:
    """High level client for the Tensorus MCP server."""

    def __init__(self, transport: Any) -> None:
        self._client = FastMCPClient(transport)

    async def __aenter__(self) -> "TensorusMCPClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    async def _call_json(self, name: str, arguments: Optional[dict] = None) -> Any:
        result = await self._client.call_tool(name, arguments or {})
        if not result:
            return None
        content = result[0]
        if isinstance(content, TextContent):
            return json.loads(content.text)
        raise TypeError("Unexpected content type")

    # --- Dataset management ---
    async def list_datasets(self) -> Any:
        return await self._call_json("tensorus_list_datasets")

    async def create_dataset(self, dataset_name: str) -> Any:
        return await self._call_json("tensorus_create_dataset", {"dataset_name": dataset_name})

    async def delete_dataset(self, dataset_name: str) -> Any:
        return await self._call_json("tensorus_delete_dataset", {"dataset_name": dataset_name})

    # --- Tensor management ---
    async def ingest_tensor(
        self,
        dataset_name: str,
        tensor_shape: Sequence[int],
        tensor_dtype: str,
        tensor_data: Any,
        metadata: Optional[dict] = None,
    ) -> Any:
        payload = {
            "dataset_name": dataset_name,
            "tensor_shape": list(tensor_shape),
            "tensor_dtype": tensor_dtype,
            "tensor_data": tensor_data,
            "metadata": metadata,
        }
        return await self._call_json("tensorus_ingest_tensor", payload)

    async def get_tensor_details(self, dataset_name: str, record_id: str) -> Any:
        return await self._call_json(
            "tensorus_get_tensor_details",
            {"dataset_name": dataset_name, "record_id": record_id},
        )

    async def delete_tensor(self, dataset_name: str, record_id: str) -> Any:
        return await self._call_json(
            "tensorus_delete_tensor",
            {"dataset_name": dataset_name, "record_id": record_id},
        )

    async def update_tensor_metadata(
        self, dataset_name: str, record_id: str, new_metadata: dict
    ) -> Any:
        return await self._call_json(
            "tensorus_update_tensor_metadata",
            {
                "dataset_name": dataset_name,
                "record_id": record_id,
                "new_metadata": new_metadata,
            },
        )

    # --- Tensor operations ---
    async def apply_unary_operation(self, operation: str, payload: dict) -> Any:
        return await self._call_json("tensorus_apply_unary_operation", {"operation": operation, **payload})

    async def apply_binary_operation(self, operation: str, payload: dict) -> Any:
        return await self._call_json("tensorus_apply_binary_operation", {"operation": operation, **payload})

    async def apply_list_operation(self, operation: str, payload: dict) -> Any:
        return await self._call_json("tensorus_apply_list_operation", {"operation": operation, **payload})

    async def apply_einsum(self, payload: dict) -> Any:
        return await self._call_json("tensorus_apply_einsum", payload)

    # --- Misc ---
    async def save_tensor(
        self,
        dataset_name: str,
        tensor_shape: Sequence[int],
        tensor_dtype: str,
        tensor_data: Any,
        metadata: Optional[dict] = None,
    ) -> Any:
        payload = {
            "dataset_name": dataset_name,
            "tensor_shape": list(tensor_shape),
            "tensor_dtype": tensor_dtype,
            "tensor_data": tensor_data,
            "metadata": metadata,
        }
        return await self._call_json("save_tensor", payload)

    async def get_tensor(self, dataset_name: str, record_id: str) -> Any:
        return await self._call_json("get_tensor", {"dataset_name": dataset_name, "record_id": record_id})

    async def execute_nql_query(self, query: str) -> Any:
        return await self._call_json("execute_nql_query", {"query": query})
