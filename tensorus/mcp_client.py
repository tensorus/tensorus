from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Sequence, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError
from fastmcp.client import Client as FastMCPClient
from fastmcp.exceptions import FastMCPError
from fastmcp.client.transports import StreamableHttpTransport # Changed import path and class name

# Import the TextContent type that the server is actually using
from mcp.types import TextContent

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger("tensorus.mcp.client")
logger.setLevel(logging.INFO)

class MCPResponseError(Exception):
    """Generic exception for MCP response errors"""
    pass

class TensorusMCPClient:
    """High-level client for the Tensorus MCP server with typed, prompt, sync, and streaming support."""

    def __init__(self, transport: Any) -> None:
        self._client = FastMCPClient(transport)

    @staticmethod
    def from_http(url: str) -> TensorusMCPClient:
        """Factory using Streamable HTTP transport"""
        transport = StreamableHttpTransport(url=url) # Updated class name usage
        return TensorusMCPClient(transport)

    async def __aenter__(self) -> TensorusMCPClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any
    ) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def _call_json(
        self,
        name: str,
        args: Optional[dict] = None,
        response_model: Type[T] | None = None
    ) -> Union[dict, T, None]:
        """
        Internal helper to call a tool and parse JSON or Pydantic model.
        """
        try:
            result = await self._client.call_tool(name, args or {})
        except FastMCPError as e:
            logger.error(f"Tool call failed: {name} args={args} error={e}")
            raise MCPResponseError(str(e))

        if not result:
            return None
        content = result[0]
        if not isinstance(content, TextContent):
            raise TypeError(f"Unexpected content type: {type(content)}")

        data = json.loads(content.text)
        if response_model:
            try:
                # Handle cases where the actual data for the model is nested
                if isinstance(data, dict) and 'data' in data and response_model == DatasetListResponse :
                    # Specific handling for DatasetListResponse if data is nested under 'data'
                    # and the model itself expects a flat structure like {"datasets": []}
                    # This might indicate an API inconsistency or a client model mismatch.
                    # For now, let's assume the API returns {..., "data": actual_list_for_datasets_field}
                    # and DatasetListResponse expects {"datasets": actual_list_for_datasets_field}
                    # So, we need to re-wrap it:
                    return response_model.parse_obj({"datasets": data['data']})
                elif response_model == IngestTensorResponse and isinstance(data, dict) and data.get('success') is True and isinstance(data.get('data'), dict) and 'record_id' in data['data']:
                    # Specific handling for IngestTensorResponse if data is nested and needs id mapping
                    return IngestTensorResponse(id=data['data']['record_id'], status="ingested") # Assuming "ingested" is the status on success
                elif response_model == TensorDetailsResponse and isinstance(data, dict) and 'record_id' in data:
                    # Map API's 'record_id' to model's 'id' field
                    data_for_model = data.copy()
                    data_for_model['id'] = data_for_model.pop('record_id')
                    return TensorDetailsResponse.parse_obj(data_for_model)

                # For other models (like CreateDatasetResponse, DeleteDatasetResponse, etc.),
                # parse the data directly. This assumes 'data' dictionary as a whole matches the response_model.
                return response_model.parse_obj(data)
            except ValidationError as ve:
                logger.error(f"Response validation failed for {name}: {ve}. Data: {data}")
                raise
        return data

    async def call_tool_stream(
        self,
        name: str,
        args: Optional[dict] = None
    ) -> AsyncIterator[str]:
        """
        Stream partial TextContent.text chunks from a long-running tool.
        """
        try:
            async for msg in self._client.call_tool_stream(name, args or {}):
                yield msg.text
        except FastMCPError as e:
            logger.error(f"Streaming call failed: {name} error={e}")
            raise MCPResponseError(str(e))

    # --- Prompt support ---
    async def call_prompt(
        self,
        prompt_name: str,
        args: Optional[dict] = None
    ) -> List[TextContent]:
        """
        Invoke a prompt template defined on the MCP server.
        """
        try:
            return await self._client.call_prompt(prompt_name, args or {})
        except FastMCPError as e:
            logger.error(f"Prompt call failed: {prompt_name} args={args} error={e}")
            raise MCPResponseError(str(e))

    # --- Dataset management ---
    async def list_datasets(self) -> list[str]:
        return await self._call_json(
            "tensorus_list_datasets",
            response_model=DatasetListResponse
        )

    def list_datasets_sync(self) -> list[str]:
        """Sync wrapper for list_datasets"""
        return asyncio.run(self.list_datasets())

    async def create_dataset(self, dataset_name: str) -> CreateDatasetResponse:
        return await self._call_json(
            "tensorus_create_dataset",
            {"dataset_name": dataset_name},
            response_model=CreateDatasetResponse
        )

    def create_dataset_sync(self, dataset_name: str) -> CreateDatasetResponse:
        return asyncio.run(self.create_dataset(dataset_name))

    async def delete_dataset(self, dataset_name: str) -> DeleteDatasetResponse:
        return await self._call_json(
            "tensorus_delete_dataset",
            {"dataset_name": dataset_name},
            response_model=DeleteDatasetResponse
        )

    def delete_dataset_sync(self, dataset_name: str) -> DeleteDatasetResponse:
        return asyncio.run(self.delete_dataset(dataset_name))

    # --- Tensor management ---
    async def ingest_tensor(
        self,
        dataset_name: str,
        tensor_shape: Sequence[int],
        tensor_dtype: str,
        tensor_data: Any,
        metadata: Optional[dict] = None,
    ) -> IngestTensorResponse:
        payload = {
            "dataset_name": dataset_name,
            "tensor_shape": list(tensor_shape),
            "tensor_dtype": tensor_dtype,
            "tensor_data": tensor_data,
            "metadata": metadata,
        }
        return await self._call_json(
            "tensorus_ingest_tensor",
            payload,
            response_model=IngestTensorResponse
        )

    async def get_tensor_details(self, dataset_name: str, record_id: str) -> TensorDetailsResponse:
        return await self._call_json(
            "tensorus_get_tensor_details",
            {"dataset_name": dataset_name, "record_id": record_id},
            response_model=TensorDetailsResponse
        )

    async def delete_tensor(self, dataset_name: str, record_id: str) -> DeleteTensorResponse:
        return await self._call_json(
            "tensorus_delete_tensor",
            {"dataset_name": dataset_name, "record_id": record_id},
            response_model=DeleteTensorResponse
        )

    async def update_tensor_metadata(
        self,
        dataset_name: str,
        record_id: str,
        new_metadata: dict
    ) -> UpdateMetadataResponse:
        return await self._call_json(
            "tensorus_update_tensor_metadata",
            {"dataset_name": dataset_name, "record_id": record_id, "new_metadata": new_metadata},
            response_model=UpdateMetadataResponse
        )

    # --- Tensor operations ---
    async def apply_unary_operation(self, operation: str, payload: dict) -> OperationResponse:
        return await self._call_json(
            "tensorus_apply_unary_operation",
            {"operation": operation, **payload},
            response_model=OperationResponse
        )

    async def apply_binary_operation(self, operation: str, payload: dict) -> OperationResponse:
        return await self._call_json(
            "tensorus_apply_binary_operation",
            {"operation": operation, **payload},
            response_model=OperationResponse
        )

    async def apply_list_operation(self, operation: str, payload: dict) -> OperationResponse:
        return await self._call_json(
            "tensorus_apply_list_operation",
            {"operation": operation, **payload},
            response_model=OperationResponse
        )

    async def apply_einsum(self, payload: dict) -> OperationResponse:
        return await self._call_json(
            "tensorus_apply_einsum",
            payload,
            response_model=OperationResponse
        )

    # --- Miscellaneous ---
    async def execute_nql_query(self, query: str) -> NQLQueryResponse:
        return await self._call_json(
            "execute_nql_query",
            {"query": query},
            response_model=NQLQueryResponse
        )

# --- Pydantic response models ---
class DatasetListResponse(BaseModel):
    datasets: list[str]

class CreateDatasetResponse(BaseModel):
    success: bool
    message: Optional[str]

class DeleteDatasetResponse(BaseModel):
    success: bool

class IngestTensorResponse(BaseModel):
    id: str
    status: str

class TensorDetailsResponse(BaseModel):
    id: str
    shape: list[int]
    dtype: str
    data: Any
    metadata: Optional[dict]

class DeleteTensorResponse(BaseModel):
    success: bool

class UpdateMetadataResponse(BaseModel):
    success: bool

class OperationResponse(BaseModel):
    result: Any

class NQLQueryResponse(BaseModel):
    results: Any

# End of TensorusMCPClient
