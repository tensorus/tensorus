from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Sequence, Type, TypeVar, Union

from pydantic import BaseModel
from pydantic_core import ValidationError
from fastmcp.client import Client as FastMCPClient
from fastmcp.exceptions import FastMCPError
from fastmcp.client.transports import StreamableHttpTransport

# Import the TextContent type that the server is actually using
from mcp.types import TextContent

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger("tensorus.mcp.client")
logger.setLevel(logging.INFO)

# Default public MCP server hosted on HuggingFace Spaces
DEFAULT_MCP_URL = "https://tensorus-mcp.hf.space/mcp/"

class MCPResponseError(Exception):
    """Generic exception for MCP response errors"""
    pass

class TensorusMCPClient:
    """High-level client for the Tensorus MCP server with typed, prompt, sync, and streaming support.

    API Key Handling:
    The Tensorus MCP server (`mcp_server.py`) can be configured with a global API key
    (via the `--mcp-api-key` command-line argument) that it uses for requests
    to its backend (e.g., the Tensorus Core API). This is a server-side setup.

    Additionally, many tools exposed by the MCP server are defined to accept an
    `api_key` argument. If these tools require an API key for the backend, and
    the MCP server is not using a global key, this key must be provided in the
    tool call.

    Note: Most methods in the current version of `TensorusMCPClient` do not
    explicitly include an `api_key` parameter to be passed to the server-side tools.
    If you encounter authorization errors or tools fail due to missing API keys,
    you may need to:
    1. Ensure the deployed MCP server is configured with the appropriate global
       API key for its backend.
    2. Modify the specific `TensorusMCPClient` methods you are using to accept
       an `api_key` argument and include it in the payload passed to `_call_json`.
       For example, for a tool 'example_tool', the call would be
       `self._call_json("example_tool", {"arg1": "value1", "api_key": "your_key"}, ...)`.
    Always check the API key requirements of the specific MCP server deployment
    and its backend Tensorus Core API.
    """

    # Retain class attribute for backward compatibility
    DEFAULT_MCP_URL = DEFAULT_MCP_URL

    def __init__(self, transport: Any) -> None:
        self._client = FastMCPClient(transport)

    @staticmethod
    def from_http(url: str = DEFAULT_MCP_URL) -> TensorusMCPClient:
        """Factory using Streamable HTTP transport.

        Args:
            url: Base URL of the MCP server. Defaults to the public
                HuggingFace deployment.
        """
        final_url = url.rstrip("/") + "/"
        transport = StreamableHttpTransport(url=final_url)
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
        try:
            # Ensure args is not None for the call_tool itself for consistency, though FastMCPClient might handle None.
            result = await self._client.call_tool(name, args or {})
        except FastMCPError as e:
            error_str = str(e).lower()
            log_message = f"Tool call failed for tool '{name}' with args {args or {{}}}: {e}"
            if "unauthorized" in error_str or "forbidden" in error_str or "authentication" in error_str:
                log_message += " (Hint: This might be related to missing or incorrect API keys. Check server and tool requirements.)"
            logger.error(log_message)
            raise MCPResponseError(f"Tool call failed for tool '{name}' with args {args or {{}}}: {e}") from e

        if not result:
            logger.warning(f"Tool call for '{name}' with args {args or {{}}} returned no result.")
            return None

        content = result[0]
        if not isinstance(content, TextContent):
            err_msg = f"Unexpected content type: {type(content)} for tool '{name}' with args {args or {{}}}. Expected TextContent. Received: {str(content)[:200]}"
            logger.error(err_msg)
            raise MCPResponseError(err_msg)

        raw_response_text = content.text

        try:
            data = json.loads(raw_response_text)
        except json.JSONDecodeError as e:
            response_snippet = raw_response_text[:200] + ('...' if len(raw_response_text) > 200 else '')
            logger.error(f"Failed to decode JSON response for tool '{name}' with args {args or {{}}}. Raw response snippet: '{response_snippet}'. Error: {e}")
            raise MCPResponseError(f"Failed to decode JSON response for tool '{name}' with args {args or {{}}}. Raw response snippet: '{response_snippet}'.") from e

        if response_model:
            primary_validation_error = None

            # 1. Attempt generic parsing first. This is the preferred path.
            try:
                parsed_model = response_model.model_validate(data)
                return parsed_model
            except ValidationError as ve_generic:
                primary_validation_error = ve_generic # Save the error from the primary attempt.

            # 2. If generic parsing failed (primary_validation_error is set), try specific handlers as fallbacks.
            # These handlers are for known quirks or historical API response structures.
            if primary_validation_error:
                original_data_for_specific_parse = data # Keep original data reference
                data_for_specific_parse = None # This will hold the data structure for a specific model if a handler matches

                try:
                    if response_model == DatasetListResponse and isinstance(original_data_for_specific_parse, dict) and 'data' in original_data_for_specific_parse and isinstance(original_data_for_specific_parse['data'], list):
                        data_for_specific_parse = {"datasets": original_data_for_specific_parse['data']}
                    elif response_model == IngestTensorResponse and isinstance(original_data_for_specific_parse, dict) and original_data_for_specific_parse.get('success') is True and isinstance(original_data_for_specific_parse.get('data'), dict) and 'record_id' in original_data_for_specific_parse['data']:
                        # This handler constructs the model directly, not using parse_obj on a potentially incomplete dict.
                        # Ensure record_id is string as per model.
                        return IngestTensorResponse(id=str(original_data_for_specific_parse['data']['record_id']), status="ingested")
                    elif response_model == TensorDetailsResponse and isinstance(original_data_for_specific_parse, dict) and 'record_id' in original_data_for_specific_parse:
                        # This handler modifies the data structure to fit TensorDetailsResponse.
                        # It assumes other required fields (shape, dtype, data) are already in original_data_for_specific_parse.
                        data_for_specific_parse = original_data_for_specific_parse.copy()
                        data_for_specific_parse['id'] = str(data_for_specific_parse.pop('record_id'))

                    # If a specific handler prepared data_for_specific_parse, attempt to parse it.
                    if data_for_specific_parse is not None:
                        parsed_model = response_model.model_validate(data_for_specific_parse)
                        logger.info(f"Successfully parsed response for tool '{name}' using a specific fallback handler after generic parse failed.")
                        return parsed_model
                    else:
                        # No specific handler matched, so the primary_validation_error from the generic attempt is the one to report.
                        # Re-raise it to be caught by the final error handling block below.
                        raise primary_validation_error

                except ValidationError as ve_specific:
                    # A specific handler was chosen, and its parse_obj attempt failed.
                    # This ve_specific is now the most relevant error.
                    primary_validation_error = ve_specific
                # Note: Not catching other exceptions like KeyError here to keep it focused on ValidationError.
                # If KeyErrors occur, they might lead to ValidationErrors if Pydantic models expect those fields.

            # 3. If primary_validation_error is still set, it means all attempts (generic and applicable specific fallbacks) failed.
            if primary_validation_error:
                response_snippet = raw_response_text[:500] + ('...' if len(raw_response_text) > 500 else '')
                # Log the detailed error including the type of validation error and affected fields if available from Pydantic's error.
                error_log_message = (
                    f"Response validation failed for tool '{name}' with args {args or {{}}}. "
                    f"Error: {primary_validation_error}. "
                    f"Raw response snippet: '{response_snippet}'"
                )
                logger.error(error_log_message)

                # The error raised to the caller should be informative. str(primary_validation_error) is often user-friendly.
                raise MCPResponseError(
                    f"Failed to parse response for tool '{name}' with args {args or {{}}}. "
                    f"Validation Error: {str(primary_validation_error)}. "
                    f"Raw response snippet: '{response_snippet}'"
                ) from primary_validation_error

        # If response_model was None, return the raw JSON data.
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
        response_data = await self._call_json(
            "tensorus_list_datasets",
            response_model=DatasetListResponse
        )
        if response_data is None: # Check for None first
             logger.warning("list_datasets received no data from _call_json.")
             return []
        if isinstance(response_data, DatasetListResponse): # Then check for type
            return response_data.datasets
        # If not None and not DatasetListResponse, it's an unexpected type
        logger.error(f"list_datasets received unexpected data type: {type(response_data)}")
        return []

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

    # --- Tensor descriptor CRUD ---
    async def create_tensor_descriptor(
        self, descriptor_data: dict
    ) -> TensorDescriptorResponse:
        return await self._call_json(
            "create_tensor_descriptor",
            {"descriptor_data": descriptor_data},
            response_model=TensorDescriptorResponse,
        )

    async def list_tensor_descriptors(
        self,
        owner: Optional[str] = None,
        data_type: Optional[str] = None,
        tags_contain: Optional[str] = None,
        lineage_version: Optional[str] = None,
        lineage_source_type: Optional[str] = None,
        comp_algorithm: Optional[str] = None,
        comp_gpu_model: Optional[str] = None,
        quality_confidence_gt: Optional[float] = None,
        quality_noise_lt: Optional[float] = None,
        rel_collection: Optional[str] = None,
        rel_has_related_tensor_id: Optional[str] = None,
        usage_last_accessed_before: Optional[str] = None,
        usage_used_by_app: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        min_dimensions: Optional[int] = None,
    ) -> list[dict]:
        params = {}
        if owner is not None:
            params["owner"] = owner
        if data_type is not None:
            params["data_type"] = data_type
        if tags_contain is not None:
            params["tags_contain"] = tags_contain
        if lineage_version is not None:
            params["lineage.version"] = lineage_version
        if lineage_source_type is not None:
            params["lineage.source.type"] = lineage_source_type
        if comp_algorithm is not None:
            params["computational.algorithm"] = comp_algorithm
        if comp_gpu_model is not None:
            params["computational.hardware_info.gpu_model"] = comp_gpu_model
        if quality_confidence_gt is not None:
            params["quality.confidence_score_gt"] = quality_confidence_gt
        if quality_noise_lt is not None:
            params["quality.noise_level_lt"] = quality_noise_lt
        if rel_collection is not None:
            params["relational.collection"] = rel_collection
        if rel_has_related_tensor_id is not None:
            params["relational.has_related_tensor_id"] = rel_has_related_tensor_id
        if usage_last_accessed_before is not None:
            params["usage.last_accessed_before"] = usage_last_accessed_before
        if usage_used_by_app is not None:
            params["usage.used_by_app"] = usage_used_by_app
        if name is not None:
            params["name"] = name
        if description is not None:
            params["description"] = description
        if min_dimensions is not None:
            params["min_dimensions"] = min_dimensions

        return await self._call_json(
            "list_tensor_descriptors", params
        )

    async def get_tensor_descriptor(self, tensor_id: str) -> TensorDescriptorResponse:
        return await self._call_json(
            "get_tensor_descriptor",
            {"tensor_id": tensor_id},
            response_model=TensorDescriptorResponse,
        )

    async def update_tensor_descriptor(
        self, tensor_id: str, updates: dict
    ) -> TensorDescriptorResponse:
        return await self._call_json(
            "update_tensor_descriptor",
            {"tensor_id": tensor_id, "updates": updates},
            response_model=TensorDescriptorResponse,
        )

    async def delete_tensor_descriptor(self, tensor_id: str) -> MessageResponse:
        return await self._call_json(
            "delete_tensor_descriptor",
            {"tensor_id": tensor_id},
            response_model=MessageResponse,
        )

    # --- Semantic Metadata ---
    async def create_semantic_metadata_for_tensor(
        self, tensor_id: str, metadata_in: dict
    ) -> SemanticMetadataResponse:
        return await self._call_json(
            "create_semantic_metadata_for_tensor",
            {"tensor_id": tensor_id, "metadata_in": metadata_in},
            response_model=SemanticMetadataResponse,
        )

    async def get_all_semantic_metadata_for_tensor(
        self, tensor_id: str
    ) -> list[SemanticMetadataResponse]:
        raw_data_list = await self._call_json(
            "get_all_semantic_metadata_for_tensor", # Tool name
            {"tensor_id": tensor_id} # Arguments
            # No response_model, so _call_json returns raw dict/list
        )
        if isinstance(raw_data_list, list):
            parsed_list = []
            for item in raw_data_list:
                try:
                    if isinstance(item, dict):
                       parsed_list.append(SemanticMetadataResponse.model_validate(item))
                    else:
                       logger.warning(f"Item in list for get_all_semantic_metadata_for_tensor is not a dict: {type(item)}")
                       # Optionally skip, or add placeholder, or error out
                except ValidationError as e:
                    logger.error(f"Failed to parse item in get_all_semantic_metadata_for_tensor: {item}. Error: {e}")
                    # Optionally skip, or add placeholder, or error out
            return parsed_list
        elif raw_data_list is None:
            logger.warning(f"get_all_semantic_metadata_for_tensor for {tensor_id} received no data.")
            return []
        else:
            logger.error(f"get_all_semantic_metadata_for_tensor for {tensor_id} received unexpected data type: {type(raw_data_list)}. Expected list.")
            return [] # Or raise error

    async def update_named_semantic_metadata_for_tensor(
        self, tensor_id: str, current_name: str, updates: dict
    ) -> SemanticMetadataResponse:
        return await self._call_json(
            "update_named_semantic_metadata_for_tensor",
            {
                "tensor_id": tensor_id,
                "current_name": current_name,
                "updates": updates,
            },
            response_model=SemanticMetadataResponse,
        )

    async def delete_named_semantic_metadata_for_tensor(
        self, tensor_id: str, name: str
    ) -> MessageResponse:
        return await self._call_json(
            "delete_named_semantic_metadata_for_tensor",
            {"tensor_id": tensor_id, "name": name},
            response_model=MessageResponse,
        )

    # --- Extended Metadata ---
    async def upsert_lineage_metadata(self, tensor_id: str, metadata_in: dict) -> MetadataResponse:
        return await self._call_json(
            "upsert_lineage_metadata",
            {"tensor_id": tensor_id, "metadata_in": metadata_in},
            response_model=MetadataResponse,
        )

    async def get_lineage_metadata(self, tensor_id: str) -> MetadataResponse:
        return await self._call_json(
            "get_lineage_metadata",
            {"tensor_id": tensor_id},
            response_model=MetadataResponse,
        )

    async def patch_lineage_metadata(self, tensor_id: str, updates: dict) -> MetadataResponse:
        return await self._call_json(
            "patch_lineage_metadata",
            {"tensor_id": tensor_id, "updates": updates},
            response_model=MetadataResponse,
        )

    async def delete_lineage_metadata(self, tensor_id: str) -> MessageResponse:
        return await self._call_json(
            "delete_lineage_metadata",
            {"tensor_id": tensor_id},
            response_model=MessageResponse,
        )

    async def upsert_computational_metadata(self, tensor_id: str, metadata_in: dict) -> MetadataResponse:
        return await self._call_json(
            "upsert_computational_metadata",
            {"tensor_id": tensor_id, "metadata_in": metadata_in},
            response_model=MetadataResponse,
        )

    async def get_computational_metadata(self, tensor_id: str) -> MetadataResponse:
        return await self._call_json(
            "get_computational_metadata",
            {"tensor_id": tensor_id},
            response_model=MetadataResponse,
        )

    async def patch_computational_metadata(self, tensor_id: str, updates: dict) -> MetadataResponse:
        return await self._call_json(
            "patch_computational_metadata",
            {"tensor_id": tensor_id, "updates": updates},
            response_model=MetadataResponse,
        )

    async def delete_computational_metadata(self, tensor_id: str) -> MessageResponse:
        return await self._call_json(
            "delete_computational_metadata",
            {"tensor_id": tensor_id},
            response_model=MessageResponse,
        )

    async def upsert_quality_metadata(self, tensor_id: str, metadata_in: dict) -> MetadataResponse:
        return await self._call_json(
            "upsert_quality_metadata",
            {"tensor_id": tensor_id, "metadata_in": metadata_in},
            response_model=MetadataResponse,
        )

    async def get_quality_metadata(self, tensor_id: str) -> MetadataResponse:
        return await self._call_json(
            "get_quality_metadata",
            {"tensor_id": tensor_id},
            response_model=MetadataResponse,
        )

    async def patch_quality_metadata(self, tensor_id: str, updates: dict) -> MetadataResponse:
        return await self._call_json(
            "patch_quality_metadata",
            {"tensor_id": tensor_id, "updates": updates},
            response_model=MetadataResponse,
        )

    async def delete_quality_metadata(self, tensor_id: str) -> MessageResponse:
        return await self._call_json(
            "delete_quality_metadata",
            {"tensor_id": tensor_id},
            response_model=MessageResponse,
        )

    async def upsert_relational_metadata(self, tensor_id: str, metadata_in: dict) -> MetadataResponse:
        return await self._call_json(
            "upsert_relational_metadata",
            {"tensor_id": tensor_id, "metadata_in": metadata_in},
            response_model=MetadataResponse,
        )

    async def get_relational_metadata(self, tensor_id: str) -> MetadataResponse:
        return await self._call_json(
            "get_relational_metadata",
            {"tensor_id": tensor_id},
            response_model=MetadataResponse,
        )

    async def patch_relational_metadata(self, tensor_id: str, updates: dict) -> MetadataResponse:
        return await self._call_json(
            "patch_relational_metadata",
            {"tensor_id": tensor_id, "updates": updates},
            response_model=MetadataResponse,
        )

    async def delete_relational_metadata(self, tensor_id: str) -> MessageResponse:
        return await self._call_json(
            "delete_relational_metadata",
            {"tensor_id": tensor_id},
            response_model=MessageResponse,
        )

    async def upsert_usage_metadata(self, tensor_id: str, metadata_in: dict) -> MetadataResponse:
        return await self._call_json(
            "upsert_usage_metadata",
            {"tensor_id": tensor_id, "metadata_in": metadata_in},
            response_model=MetadataResponse,
        )

    async def get_usage_metadata(self, tensor_id: str) -> MetadataResponse:
        return await self._call_json(
            "get_usage_metadata",
            {"tensor_id": tensor_id},
            response_model=MetadataResponse,
        )

    async def patch_usage_metadata(self, tensor_id: str, updates: dict) -> MetadataResponse:
        return await self._call_json(
            "patch_usage_metadata",
            {"tensor_id": tensor_id, "updates": updates},
            response_model=MetadataResponse,
        )

    async def delete_usage_metadata(self, tensor_id: str) -> MessageResponse:
        return await self._call_json(
            "delete_usage_metadata",
            {"tensor_id": tensor_id},
            response_model=MessageResponse,
        )

    # --- Search and Aggregation ---
    async def search_tensors(
        self, text_query: str, fields_to_search: Optional[str] = None
    ) -> list[dict]:
        params = {"text_query": text_query}
        if fields_to_search is not None:
            params["fields_to_search"] = fields_to_search
        return await self._call_json("search_tensors", params)

    async def aggregate_tensors(
        self, group_by_field: str, agg_function: str, agg_field: Optional[str] = None
    ) -> list[dict]:
        params = {
            "group_by_field": group_by_field,
            "agg_function": agg_function,
        }
        if agg_field is not None:
            params["agg_field"] = agg_field
        return await self._call_json("aggregate_tensors", params)

    # --- Versioning and Lineage ---
    async def create_tensor_version(self, tensor_id: str, version_request: dict) -> dict:
        return await self._call_json(
            "create_tensor_version",
            {"tensor_id": tensor_id, "version_request": version_request},
        )

    async def list_tensor_versions(self, tensor_id: str) -> list[dict]:
        return await self._call_json(
            "list_tensor_versions",
            {"tensor_id": tensor_id},
        )

    async def create_lineage_relationship(self, relationship_request: dict) -> dict:
        return await self._call_json(
            "create_lineage_relationship",
            {"relationship_request": relationship_request},
        )

    async def get_parent_tensors(self, tensor_id: str) -> list[dict]:
        return await self._call_json(
            "get_parent_tensors",
            {"tensor_id": tensor_id},
        )

    async def get_child_tensors(self, tensor_id: str) -> list[dict]:
        return await self._call_json(
            "get_child_tensors",
            {"tensor_id": tensor_id},
        )

    # --- Import/Export and Management ---
    async def export_tensor_metadata(self, tensor_ids_str: Optional[str] = None) -> list[dict]:
        params = {}
        if tensor_ids_str is not None:
            params["tensor_ids"] = tensor_ids_str
        return await self._call_json("export_tensor_metadata", params)

    async def import_tensor_metadata(
        self, import_data_payload: dict, conflict_strategy: Optional[str] = "skip"
    ) -> dict:
        return await self._call_json(
            "import_tensor_metadata",
            {
                "import_data_payload": import_data_payload,
                "conflict_strategy": conflict_strategy,
            },
        )

    async def management_health_check(self) -> dict:
        return await self._call_json("management_health_check")

    async def management_get_metrics(self) -> dict:
        return await self._call_json("management_get_metrics")

    # --- Analytics ---
    async def analytics_get_co_occurring_tags(
        self, min_co_occurrence: Optional[int] = 2, limit: Optional[int] = 10
    ) -> list[dict]:
        params = {}
        if min_co_occurrence is not None:
            params["min_co_occurrence"] = min_co_occurrence
        if limit is not None:
            params["limit"] = limit
        return await self._call_json("analytics_get_co_occurring_tags", params)

    async def analytics_get_stale_tensors(
        self, threshold_days: Optional[int] = 90, limit: Optional[int] = 100
    ) -> list[dict]:
        params = {}
        if threshold_days is not None:
            params["threshold_days"] = threshold_days
        if limit is not None:
            params["limit"] = limit
        return await self._call_json("analytics_get_stale_tensors", params)

    async def analytics_get_complex_tensors(
        self,
        min_parent_count: Optional[int] = None,
        min_transformation_steps: Optional[int] = None,
        limit: Optional[int] = 100,
    ) -> list[dict]:
        params = {}
        if min_parent_count is not None:
            params["min_parent_count"] = min_parent_count
        if min_transformation_steps is not None:
            params["min_transformation_steps"] = min_transformation_steps
        if limit is not None:
            params["limit"] = limit
        return await self._call_json("analytics_get_complex_tensors", params)

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


class TensorDescriptorResponse(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None


class SemanticMetadataResponse(BaseModel):
    id: str
    tensor_descriptor_id: str
    name: Optional[str] = None
    value: Any = None
    type: Optional[str] = None


class MetadataResponse(BaseModel):
    tensor_descriptor_id: str
    data: Any


class MessageResponse(BaseModel):
    message: str

# End of TensorusMCPClient
