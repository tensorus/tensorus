"""FastMCP server exposing Tensorus API endpoints as tools.

This module registers a set of MCP tools that proxy to the Tensorus FastAPI
backend.  Tools mirror the ones documented in the README under "Available
Tools" and return results as :class:`TextContent` objects.
"""

if __package__ in (None, ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tensorus"

import argparse
import json
from typing import Any, Optional, Sequence, Dict

from pydantic import Field

from tensorus.config import settings

import httpx
from httpx import HTTPStatusError

try:
    from fastmcp import FastMCP
    from fastmcp.prompts.prompt import PromptMessage, Message, TextContent
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - support older fastmcp versions or missing deps
    from dataclasses import dataclass

    @dataclass
    class TextContent:  # minimal fallback for tests
        type: str
        text: str

    PromptMessage = Message = Any  # type: ignore
    FastMCP = None
    MCP_AVAILABLE = False

API_BASE_URL = "https://tensorus-core.hf.space"
GLOBAL_API_KEY: Optional[str] = None
DEMO_MODE: bool = False
DEMO_RESPONSES: Dict[str, Any] = {
    "/datasets": {
        "data": [
            {"name": "sample_dataset_1", "id": "demo_ds_001"},
            {"name": "sample_dataset_2", "id": "demo_ds_002"},
        ]
    },
    "/health": {
        "status": "healthy",
        "demo_mode": True,
        "message": "Backend is mocked in demo mode.",
    },
    "generic_tensor": {
        "id": "tensor_demo_001",
        "shape": [64, 64, 3],
        "dtype": "float32",
        "demo_content": True,
    },
    "default_tool_demo": {
        "message": "This tool is running in demo mode. No actual API call was made.",
        "data": "sample demo data",
    },
}

if MCP_AVAILABLE:
    server = FastMCP(name="Tensorus FastMCP")
else:
    # Create a dummy server object with no-op decorators when MCP is not available
    class DummyServer:
        def tool(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        def prompt(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
            
        def resource(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    server = DummyServer()


def _wrap_backend_response(action: str, result: Any) -> TextContent:
    """Return a TextContent message with guidance for API key or network errors."""
    # Handle both dictionary and list responses
    if not isinstance(result, dict):
        # For non-dict responses (like lists from analytics endpoints), return as-is
        return TextContent(type="text", text=json.dumps(result))
    
    status = result.get("status")
    if isinstance(result.get("detail"), dict) and "status" in result["detail"]:
        status = result["detail"]["status"]

    if result.get("error") == "Network error":
        msg = "Backend service is unreachable."
        return TextContent(type="text", text=f"{msg} Response: {json.dumps(result)}")

    if result.get("error") == "API key required" or status == 401:
        msg = f"This demo server does not support {action} without an API key."
        return TextContent(type="text", text=f"{msg} Response: {json.dumps(result)}")

    if result.get("error") == "Access forbidden" or status == 403:
        msg = f"Access forbidden when performing {action}. Provide a valid API key."
        return TextContent(type="text", text=f"{msg} Response: {json.dumps(result)}")

    return TextContent(type="text", text=json.dumps(result))


async def _post(
    path: str,
    payload: dict,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> dict:
    if DEMO_MODE:
        if path == "/datasets/create":
            return {
                "status": "success",
                "demo_mode": True,
                "message": "Dataset creation mocked.",
                "dataset_info": payload,
            }
        # Generic POST demo response
        return {
            "message": f"Demo mode: No specific mock for POST {path}",
            "payload_received": payload,
            "data": DEMO_RESPONSES["default_tool_demo"],
        }
    try:
        headers = {}
        actual_api_key = api_key if api_key is not None else GLOBAL_API_KEY
        if actual_api_key:
            headers[settings.API_KEY_HEADER_NAME] = actual_api_key
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}{path}", json=payload, params=params, headers=headers
            )
            response.raise_for_status()  # Will raise HTTPStatusError for 4xx/5xx
            return response.json()

    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return {"error": "API key required"}
        elif exc.response.status_code == 403:
            return {"error": "Access forbidden"}
        else:
            # Handle other HTTP errors
            return {
                "error": f"HTTP error: {exc.response.status_code} {exc.response.reason_phrase}",
                "message": str(exc),
            }
    except httpx.HTTPError as exc:  # Catches other network errors
        return {"error": "Network error", "message": str(exc)}


async def _get(
    path: str, params: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None
) -> dict:
    if DEMO_MODE:
        if path == "/datasets":
            return DEMO_RESPONSES["/datasets"]
        elif path == "/health":
            return DEMO_RESPONSES["/health"]
        elif (
            path.startswith("/datasets/") and "/tensors/" in path
        ):  # For get_tensor_details
            return {
                **DEMO_RESPONSES["generic_tensor"],
                "id": path.split("/")[-1],
                "path_called": path,
            }
        elif path == "/tensor_descriptors/":  # For list_tensor_descriptors
            return [
                {"id": "tensor_desc_001", "name": "demo_tensor_1", "data_type": "float32"},
                {"id": "tensor_desc_002", "name": "demo_tensor_2", "data_type": "int64"},
            ]
        elif path.startswith("/tensor_descriptors/"):  # For get_tensor_descriptor
            return {
                **DEMO_RESPONSES["generic_tensor"],
                "id": path.split("/")[-1],
                "type": "descriptor",
                "path_called": path,
            }
        elif path.startswith("/tensors/") and path.endswith("/versions"):  # For list_tensor_versions
            # Return a list of demo versions
            return [
                {"version_id": "v_demo_001", "version_tag": "v1.0", "created_at": "2024-01-01T00:00:00Z"},
                {"version_id": "v_demo_002", "version_tag": "v1.1", "created_at": "2024-01-02T00:00:00Z"},
            ]
        elif path.startswith("/tensors/") and "/lineage/parents" in path:  # For get_parent_tensors
            # Return a list of demo parent tensors
            return [
                {"tensor_id": "parent_001", "relationship_type": "derived_from"},
                {"tensor_id": "parent_002", "relationship_type": "transformed_from"},
            ]
        elif path.startswith("/tensors/") and "/lineage/children" in path:  # For get_child_tensors
            # Return a list of demo child tensors
            return [
                {"tensor_id": "child_001", "relationship_type": "derived_to"},
                {"tensor_id": "child_002", "relationship_type": "transformed_to"},
            ]
        # Fallback for unhandled demo paths in GET
        return {
            "message": f"Demo mode: No specific mock for GET {path}",
            "data": DEMO_RESPONSES["default_tool_demo"],
        }
    try:
        headers = {}
        actual_api_key = api_key if api_key is not None else GLOBAL_API_KEY
        if actual_api_key:
            headers[settings.API_KEY_HEADER_NAME] = actual_api_key
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}{path}", params=params, headers=headers
            )
            response.raise_for_status()  # Will raise HTTPStatusError for 4xx/5xx
            return response.json()

    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return {"error": "API key required"}
        elif exc.response.status_code == 403:
            return {"error": "Access forbidden"}
        else:
            # Handle other HTTP errors
            return {
                "error": f"HTTP error: {exc.response.status_code} {exc.response.reason_phrase}",
                "message": str(exc),
            }
    except httpx.HTTPError as exc:  # Catches other network errors
        return {"error": "Network error", "message": str(exc)}


async def _put(
    path: str,
    payload: dict,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> dict:
    if DEMO_MODE:
        # Generic PUT demo response
        return {
            "message": f"Demo mode: No specific mock for PUT {path}",
            "payload_received": payload,
            "data": DEMO_RESPONSES["default_tool_demo"],
        }
    try:
        headers = {}
        actual_api_key = api_key if api_key is not None else GLOBAL_API_KEY
        if actual_api_key:
            headers[settings.API_KEY_HEADER_NAME] = actual_api_key
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{API_BASE_URL}{path}", json=payload, params=params, headers=headers
            )
            response.raise_for_status()  # Will raise HTTPStatusError for 4xx/5xx
            return response.json()

    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return {"error": "API key required"}
        elif exc.response.status_code == 403:
            return {"error": "Access forbidden"}
        else:
            # Handle other HTTP errors
            return {
                "error": f"HTTP error: {exc.response.status_code} {exc.response.reason_phrase}",
                "message": str(exc),
            }
    except httpx.HTTPError as exc:  # Catches other network errors
        return {"error": "Network error", "message": str(exc)}


async def _delete(path: str, api_key: Optional[str] = None) -> dict:
    if DEMO_MODE:
        # Generic DELETE demo response
        return {
            "message": f"Demo mode: No specific mock for DELETE {path}",
            "data": DEMO_RESPONSES["default_tool_demo"],
        }
    try:
        headers = {}
        actual_api_key = api_key if api_key is not None else GLOBAL_API_KEY
        if actual_api_key:
            headers[settings.API_KEY_HEADER_NAME] = actual_api_key
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{API_BASE_URL}{path}", headers=headers)
            response.raise_for_status()  # Will raise HTTPStatusError for 4xx/5xx
            return response.json()

    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return {"error": "API key required"}
        elif exc.response.status_code == 403:
            return {"error": "Access forbidden"}
        else:
            # Handle other HTTP errors
            return {
                "error": f"HTTP error: {exc.response.status_code} {exc.response.reason_phrase}",
                "message": str(exc),
            }
    except httpx.HTTPError as exc:  # Catches other network errors
        return {"error": "Network error", "message": str(exc)}


async def _patch(
    path: str,
    payload: dict,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> dict:
    if DEMO_MODE:
        # Generic PATCH demo response
        return {
            "message": f"Demo mode: No specific mock for PATCH {path}",
            "payload_received": payload,
            "data": DEMO_RESPONSES["default_tool_demo"],
        }
    try:
        headers = {}
        actual_api_key = api_key if api_key is not None else GLOBAL_API_KEY
        if actual_api_key:
            headers[settings.API_KEY_HEADER_NAME] = actual_api_key
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{API_BASE_URL}{path}", json=payload, params=params, headers=headers
            )
            response.raise_for_status()  # Will raise HTTPStatusError for 4xx/5xx
            return response.json()

    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return {"error": "API key required"}
        elif exc.response.status_code == 403:
            return {"error": "Access forbidden"}
        else:
            # Handle other HTTP errors
            return {
                "error": f"HTTP error: {exc.response.status_code} {exc.response.reason_phrase}",
                "message": str(exc),
            }
    except httpx.HTTPError as exc:  # Catches other network errors
        return {"error": "Network error", "message": str(exc)}


async def fetch_metadata(record_id: str) -> dict:
    """Helper to fetch metadata for dynamic prompts."""
    return await _get(f"/tensors/{record_id}/metadata")


@server.prompt()
async def ask_about_topic(topic: str) -> str:
    """Generate a user message asking for an explanation of a topic."""
    return f"Can you explain the concept of '{topic}'?"


@server.prompt()
async def summarize_text(
    text: str = Field(description="Text to summarize"), max_length: int = 100
) -> str:
    return f"Summarize the following in {max_length} words:\n\n{text}"


@server.prompt(
    name="data_analysis_request",
    description="Builds a prompt to analyze a dataset",
    tags={"analysis", "data"},
)
async def data_analysis_prompt(data_uri: str) -> str:
    return f"Analyze the data at {data_uri} and report key insights."


@server.prompt()
async def dynamic_prompt(record_id: str) -> str:
    data = await fetch_metadata(record_id)
    return f"Here\u2019s the metadata: {data}"


@server.tool()
async def save_tensor(
    dataset_name: str,
    tensor_shape: Sequence[int],
    tensor_dtype: str,
    tensor_data: Any,
    metadata: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> TextContent:
    """Save a tensor to a dataset.
    Requires API key if not in demo mode.
    """
    payload = {
        "shape": list(tensor_shape),
        "dtype": tensor_dtype,
        "data": tensor_data,
        "metadata": metadata,
    }
    result = await _post(f"/datasets/{dataset_name}/ingest", payload, api_key=api_key)
    return _wrap_backend_response("save_tensor", result)


@server.tool()
async def get_tensor(dataset_name: str, record_id: str) -> TextContent:
    """Retrieve a tensor by record ID."""
    result = await _get(f"/datasets/{dataset_name}/tensors/{record_id}")
    return _wrap_backend_response("get_tensor", result)


@server.tool()
async def execute_nql_query(query: str, api_key: Optional[str] = None) -> TextContent:
    """Execute a Natural Query Language query.
    Requires API key if not in demo mode.
    """
    result = await _post("/query", {"query": query}, api_key=api_key)
    return _wrap_backend_response("execute_nql_query", result)


# --- Dataset Management Tools ---


@server.tool(name="tensorus_list_datasets")
async def tensorus_list_datasets() -> TextContent:
    """List all available datasets."""
    result = await _get("/datasets")
    return _wrap_backend_response("tensorus_list_datasets", result)


@server.tool(name="tensorus_create_dataset")
async def tensorus_create_dataset(
    dataset_name: str, api_key: Optional[str] = None
) -> TextContent:
    """Create a new dataset.
    Requires API key if not in demo mode.
    """
    result = await _post("/datasets/create", {"name": dataset_name}, api_key=api_key)
    return _wrap_backend_response("tensorus_create_dataset", result)


@server.tool(name="tensorus_delete_dataset")
async def tensorus_delete_dataset(
    dataset_name: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete an existing dataset.
    Requires API key if not in demo mode.
    """
    result = await _delete(f"/datasets/{dataset_name}", api_key=api_key)
    return _wrap_backend_response("tensorus_delete_dataset", result)


# --- Tensor Management Tools ---


@server.tool(name="tensorus_ingest_tensor")
async def tensorus_ingest_tensor(
    dataset_name: str,
    tensor_shape: Sequence[int],
    tensor_dtype: str,
    tensor_data: Any,
    metadata: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> TextContent:
    """Ingest a new tensor into a dataset.
    Requires API key if not in demo mode.
    """
    payload = {
        "shape": list(tensor_shape),
        "dtype": tensor_dtype,
        "data": tensor_data,
        "metadata": metadata,
    }
    result = await _post(f"/datasets/{dataset_name}/ingest", payload, api_key=api_key)
    return _wrap_backend_response("tensorus_ingest_tensor", result)


@server.tool(name="tensorus_get_tensor_details")
async def tensorus_get_tensor_details(dataset_name: str, record_id: str) -> TextContent:
    """Retrieve tensor data and metadata."""
    result = await _get(f"/datasets/{dataset_name}/tensors/{record_id}")
    return _wrap_backend_response("tensorus_get_tensor_details", result)


@server.tool(name="tensorus_delete_tensor")
async def tensorus_delete_tensor(
    dataset_name: str, record_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete a tensor from a dataset.
    Requires API key if not in demo mode.
    """
    result = await _delete(
        f"/datasets/{dataset_name}/tensors/{record_id}", api_key=api_key
    )
    return _wrap_backend_response("tensorus_delete_tensor", result)


@server.tool(name="tensorus_update_tensor_metadata")
async def tensorus_update_tensor_metadata(
    dataset_name: str,
    record_id: str,
    new_metadata: dict,
    api_key: Optional[str] = None,
) -> TextContent:
    """Replace metadata for a specific tensor.
    Requires API key if not in demo mode.
    """
    payload = {"new_metadata": new_metadata}
    result = await _put(
        f"/datasets/{dataset_name}/tensors/{record_id}/metadata",
        payload,
        api_key=api_key,
    )
    return _wrap_backend_response("tensorus_update_tensor_metadata", result)


# --- Tensor Operation Tools ---


@server.tool(name="tensorus_apply_unary_operation")
async def tensorus_apply_unary_operation(
    operation: str, request_payload: dict, api_key: Optional[str] = None
) -> TextContent:
    """Apply a unary TensorOps operation (e.g., log, reshape).
    Requires API key if not in demo mode.
    """
    result = await _post(f"/ops/{operation}", request_payload, api_key=api_key)
    return _wrap_backend_response("tensorus_apply_unary_operation", result)


@server.tool(name="tensorus_apply_binary_operation")
async def tensorus_apply_binary_operation(
    operation: str, request_payload: dict, api_key: Optional[str] = None
) -> TextContent:
    """Apply a binary TensorOps operation (e.g., add, subtract).
    Requires API key if not in demo mode.
    """
    result = await _post(f"/ops/{operation}", request_payload, api_key=api_key)
    return _wrap_backend_response("tensorus_apply_binary_operation", result)


@server.tool(name="tensorus_apply_list_operation")
async def tensorus_apply_list_operation(
    operation: str, request_payload: dict, api_key: Optional[str] = None
) -> TextContent:
    """Apply a TensorOps list operation such as concatenate or stack.
    Requires API key if not in demo mode.
    """
    result = await _post(f"/ops/{operation}", request_payload, api_key=api_key)
    return _wrap_backend_response("tensorus_apply_list_operation", result)


@server.tool(name="tensorus_apply_einsum")
async def tensorus_apply_einsum(
    request_payload: dict, api_key: Optional[str] = None
) -> TextContent:
    """Apply an einsum operation.
    Requires API key if not in demo mode.
    """
    result = await _post("/ops/einsum", request_payload, api_key=api_key)
    return _wrap_backend_response("tensorus_apply_einsum", result)


# --- Tensor Descriptor Tools ---


@server.tool()
async def create_tensor_descriptor(
    descriptor_data: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Create a new tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post("/tensor_descriptors/", descriptor_data, api_key=api_key)
    return _wrap_backend_response("create_tensor_descriptor", result)


@server.tool()
async def list_tensor_descriptors(
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
) -> TextContent:
    """List tensor descriptors with extensive optional filters."""
    params: Dict[str, Any] = {}
    if owner is not None:
        params["owner"] = owner
    if data_type is not None:
        params["data_type"] = data_type  # API uses data_type
    if tags_contain is not None:
        params["tags_contain"] = (
            tags_contain  # API uses tags_contain (FastAPI handles comma-separated string to List)
        )
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

    result = await _get("/tensor_descriptors/", params=params)
    return _wrap_backend_response("list_tensor_descriptors", result)


@server.tool()
async def get_tensor_descriptor(tensor_id: str) -> TextContent:
    """Get a tensor descriptor by its ID."""
    result = await _get(f"/tensor_descriptors/{tensor_id}")
    return _wrap_backend_response("get_tensor_descriptor", result)


@server.tool()
async def update_tensor_descriptor(
    tensor_id: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Update a tensor descriptor by its ID.
    Requires API key if not in demo mode.
    """
    result = await _put(f"/tensor_descriptors/{tensor_id}", updates, api_key=api_key)
    return _wrap_backend_response("update_tensor_descriptor", result)


@server.tool()
async def delete_tensor_descriptor(
    tensor_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete a tensor descriptor by its ID.
    Requires API key if not in demo mode.
    """
    result = await _delete(f"/tensor_descriptors/{tensor_id}", api_key=api_key)
    return _wrap_backend_response("delete_tensor_descriptor", result)


# --- Semantic Metadata Tools ---


@server.tool()
async def create_semantic_metadata_for_tensor(
    tensor_id: str, metadata_in: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Create semantic metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensor_descriptors/{tensor_id}/semantic/", metadata_in, api_key=api_key
    )
    return _wrap_backend_response("create_semantic_metadata_for_tensor", result)


@server.tool()
async def get_all_semantic_metadata_for_tensor(tensor_id: str) -> TextContent:
    """Get all semantic metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/semantic/")
    return _wrap_backend_response("get_all_semantic_metadata_for_tensor", result)


@server.tool()
async def update_named_semantic_metadata_for_tensor(
    tensor_id: str, current_name: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Update a named piece of semantic metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _put(
        f"/tensor_descriptors/{tensor_id}/semantic/{current_name}",
        updates,
        api_key=api_key,
    )
    return _wrap_backend_response("update_named_semantic_metadata_for_tensor", result)


@server.tool()
async def delete_named_semantic_metadata_for_tensor(
    tensor_id: str, name: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete a named piece of semantic metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _delete(
        f"/tensor_descriptors/{tensor_id}/semantic/{name}", api_key=api_key
    )
    return _wrap_backend_response("delete_named_semantic_metadata_for_tensor", result)


# --- Extended Metadata Tools ---


# --- Lineage Metadata Tools ---
@server.tool()
async def upsert_lineage_metadata(
    tensor_id: str, metadata_in: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Upsert lineage metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensor_descriptors/{tensor_id}/lineage/", metadata_in, api_key=api_key
    )
    return _wrap_backend_response("upsert_lineage_metadata", result)


@server.tool()
async def get_lineage_metadata(tensor_id: str) -> TextContent:
    """Get lineage metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/lineage/")
    return _wrap_backend_response("get_lineage_metadata", result)


@server.tool()
async def patch_lineage_metadata(
    tensor_id: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Patch lineage metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _patch(
        f"/tensor_descriptors/{tensor_id}/lineage/", updates, api_key=api_key
    )
    return _wrap_backend_response("patch_lineage_metadata", result)


@server.tool()
async def delete_lineage_metadata(
    tensor_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete lineage metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _delete(f"/tensor_descriptors/{tensor_id}/lineage/", api_key=api_key)
    return _wrap_backend_response("delete_lineage_metadata", result)


# --- Computational Metadata Tools ---
@server.tool()
async def upsert_computational_metadata(
    tensor_id: str, metadata_in: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Upsert computational metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensor_descriptors/{tensor_id}/computational/", metadata_in, api_key=api_key
    )
    return _wrap_backend_response("upsert_computational_metadata", result)


@server.tool()
async def get_computational_metadata(tensor_id: str) -> TextContent:
    """Get computational metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/computational/")
    return _wrap_backend_response("get_computational_metadata", result)


@server.tool()
async def patch_computational_metadata(
    tensor_id: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Patch computational metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _patch(
        f"/tensor_descriptors/{tensor_id}/computational/", updates, api_key=api_key
    )
    return _wrap_backend_response("patch_computational_metadata", result)


@server.tool()
async def delete_computational_metadata(
    tensor_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete computational metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _delete(
        f"/tensor_descriptors/{tensor_id}/computational/", api_key=api_key
    )
    return _wrap_backend_response("delete_computational_metadata", result)


# --- Quality Metadata Tools ---
@server.tool()
async def upsert_quality_metadata(
    tensor_id: str, metadata_in: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Upsert quality metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensor_descriptors/{tensor_id}/quality/", metadata_in, api_key=api_key
    )
    return _wrap_backend_response("upsert_quality_metadata", result)


@server.tool()
async def get_quality_metadata(tensor_id: str) -> TextContent:
    """Get quality metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/quality/")
    return _wrap_backend_response("get_quality_metadata", result)


@server.tool()
async def patch_quality_metadata(
    tensor_id: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Patch quality metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _patch(
        f"/tensor_descriptors/{tensor_id}/quality/", updates, api_key=api_key
    )
    return _wrap_backend_response("patch_quality_metadata", result)


@server.tool()
async def delete_quality_metadata(
    tensor_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete quality metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _delete(f"/tensor_descriptors/{tensor_id}/quality/", api_key=api_key)
    return _wrap_backend_response("delete_quality_metadata", result)


# --- Relational Metadata Tools ---
@server.tool()
async def upsert_relational_metadata(
    tensor_id: str, metadata_in: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Upsert relational metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensor_descriptors/{tensor_id}/relational/", metadata_in, api_key=api_key
    )
    return _wrap_backend_response("upsert_relational_metadata", result)


@server.tool()
async def get_relational_metadata(tensor_id: str) -> TextContent:
    """Get relational metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/relational/")
    return _wrap_backend_response("get_relational_metadata", result)


@server.tool()
async def patch_relational_metadata(
    tensor_id: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Patch relational metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _patch(
        f"/tensor_descriptors/{tensor_id}/relational/", updates, api_key=api_key
    )
    return _wrap_backend_response("patch_relational_metadata", result)


@server.tool()
async def delete_relational_metadata(
    tensor_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete relational metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _delete(
        f"/tensor_descriptors/{tensor_id}/relational/", api_key=api_key
    )
    return _wrap_backend_response("delete_relational_metadata", result)


# --- Usage Metadata Tools ---
@server.tool()
async def upsert_usage_metadata(
    tensor_id: str, metadata_in: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Upsert usage metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensor_descriptors/{tensor_id}/usage/", metadata_in, api_key=api_key
    )
    return _wrap_backend_response("upsert_usage_metadata", result)


@server.tool()
async def get_usage_metadata(tensor_id: str) -> TextContent:
    """Get usage metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/usage/")
    return _wrap_backend_response("get_usage_metadata", result)


@server.tool()
async def patch_usage_metadata(
    tensor_id: str, updates: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Patch usage metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _patch(
        f"/tensor_descriptors/{tensor_id}/usage/", updates, api_key=api_key
    )
    return _wrap_backend_response("patch_usage_metadata", result)


@server.tool()
async def delete_usage_metadata(
    tensor_id: str, api_key: Optional[str] = None
) -> TextContent:
    """Delete usage metadata for a given tensor descriptor.
    Requires API key if not in demo mode.
    """
    result = await _delete(f"/tensor_descriptors/{tensor_id}/usage/", api_key=api_key)
    return _wrap_backend_response("delete_usage_metadata", result)


# --- Search and Aggregation Tools ---


@server.tool()
async def search_tensors(
    text_query: str, fields_to_search: Optional[str] = None  # Comma-separated string
) -> TextContent:
    """Search for tensors based on a text query, optionally specifying fields to search."""
    params = {"text_query": text_query}
    if fields_to_search:
        params["fields_to_search"] = fields_to_search
    result = await _get("/search/tensors/", params=params)
    return _wrap_backend_response("search_tensors", result)


@server.tool()
async def aggregate_tensors(
    group_by_field: str, agg_function: str, agg_field: Optional[str] = None
) -> TextContent:
    """Aggregate tensor metadata based on a grouping field and aggregation function."""
    params = {
        "group_by_field": group_by_field,
        "agg_function": agg_function,
    }
    if agg_field:
        params["agg_field"] = agg_field
    result = await _get("/aggregate/tensors/", params=params)
    return _wrap_backend_response("aggregate_tensors", result)


# --- Versioning and Lineage Tools ---


@server.tool()
async def create_tensor_version(
    tensor_id: str, version_request: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Create a new version for a given tensor.
    Requires API key if not in demo mode.
    """
    result = await _post(
        f"/tensors/{tensor_id}/versions", version_request, api_key=api_key
    )
    return _wrap_backend_response("create_tensor_version", result)


@server.tool()
async def list_tensor_versions(tensor_id: str) -> TextContent:
    """List all versions for a given tensor."""
    result = await _get(f"/tensors/{tensor_id}/versions")
    return _wrap_backend_response("list_tensor_versions", result)


@server.tool()
async def create_lineage_relationship(
    relationship_request: Dict, api_key: Optional[str] = None
) -> TextContent:
    """Create a lineage relationship between tensors.
    Requires API key if not in demo mode.
    """
    # relationship_request should contain source_tensor_id, target_tensor_id, relationship_type, etc.
    result = await _post(
        "/lineage/relationships/", relationship_request, api_key=api_key
    )
    return _wrap_backend_response("create_lineage_relationship", result)


@server.tool()
async def get_parent_tensors(tensor_id: str) -> TextContent:
    """Get the parent tensors for a given tensor in the lineage."""
    result = await _get(f"/tensors/{tensor_id}/lineage/parents")
    return _wrap_backend_response("get_parent_tensors", result)


@server.tool()
async def get_child_tensors(tensor_id: str) -> TextContent:
    """Get the child tensors for a given tensor in the lineage."""
    result = await _get(f"/tensors/{tensor_id}/lineage/children")
    return _wrap_backend_response("get_child_tensors", result)


@server.tool()
async def mcp_server_status() -> TextContent:
    """Check the MCP server's status and current operational mode."""
    status_info = {"status": "MCP server is running"}
    if DEMO_MODE:
        status_info["mode"] = "demo"
        status_info["message"] = "Server is operating in demo mode with mock data."
    else:
        status_info["mode"] = "live"
        status_info["message"] = (
            "Server is operating in live mode, connecting to the backend API."
        )
        if GLOBAL_API_KEY:
            status_info["api_key_status"] = "Global API key is configured."
        else:
            status_info["api_key_status"] = (
                "No global API key configured. Tools requiring authentication will need 'api_key' parameter."
            )
    return TextContent(type="text", text=json.dumps(status_info))


@server.tool()
async def connection_test() -> TextContent:
    """Simple connectivity test returning a static ok status."""
    return TextContent(type="text", text=json.dumps({"status": "ok"}))


@server.tool()
async def backend_ping() -> TextContent:
    """Ping the backend `/health` endpoint and forward the response."""
    if DEMO_MODE:
        return TextContent(
            type="text", text=json.dumps(DEMO_RESPONSES.get("/health", {"status": "healthy", "demo_mode": True}))
        )
    result = await _get("/health")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def backend_connectivity_test() -> TextContent:
    """Test connectivity to the backend API.
    In demo mode, this will return a mock success response.
    Requires API key for the '/health' endpoint if not in demo mode and if the endpoint is protected.
    """
    if DEMO_MODE:
        # Use the pre-defined mock response for /health
        response_data = DEMO_RESPONSES.get(
            "/health",
            {
                "status": "healthy",
                "demo_mode": True,
                "message": "Backend is mocked in demo mode.",
            },
        )
        return TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": "Backend connectivity test skipped in demo mode.",
                    "backend_response": response_data,
                }
            ),
        )
    else:
        # Attempt to call the actual /health endpoint
        # The _get function will handle API key logic and error formatting
        health_result = await _get("/health")

        # Check if the health_result indicates an error itself
        if "error" in health_result:
            return TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "Backend connectivity test failed or API key issue.",
                        "details": health_result,
                    }
                ),
            )
        else:
            return TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "Backend connectivity test successful.",
                        "backend_response": health_result,
                    }
                ),
            )


@server.resource("resource://datasets", name="datasets", description="List of datasets")
async def datasets_resource() -> str:
    # Assuming datasets_resource doesn't need params, or adjust if it does.
    data = await _get("/datasets")
    return json.dumps(data.get("data", []))


# --- Import/Export Tools ---


@server.tool()
async def export_tensor_metadata(tensor_ids_str: Optional[str] = None) -> TextContent:
    """Export tensor metadata for specified tensor IDs or all tensors if IDs are not provided."""
    params = {}
    if tensor_ids_str:
        params["tensor_ids"] = tensor_ids_str
    result = await _get("/tensors/export", params=params)
    return _wrap_backend_response("export_tensor_metadata", result)


@server.tool()
async def import_tensor_metadata(
    import_data_payload: Dict,
    conflict_strategy: Optional[str] = "skip",
    api_key: Optional[str] = None,
) -> TextContent:
    """Import tensor metadata with a specified conflict strategy (skip or overwrite).
    Requires API key if not in demo mode.
    """
    params = {"conflict_strategy": conflict_strategy}
    result = await _post(
        "/tensors/import", payload=import_data_payload, params=params, api_key=api_key
    )
    return _wrap_backend_response("import_tensor_metadata", result)


# --- Management Tools ---


@server.tool()
async def management_health_check() -> TextContent:
    """Perform a health check on the Tensorus service."""
    result = await _get("/health")
    return _wrap_backend_response("management_health_check", result)


@server.tool()
async def management_get_metrics() -> TextContent:
    """Retrieve operational metrics from the Tensorus service."""
    result = await _get("/metrics")
    return _wrap_backend_response("management_get_metrics", result)


# --- Analytics Tools ---


@server.tool()
async def analytics_get_co_occurring_tags(
    min_co_occurrence: Optional[int] = 2, limit: Optional[int] = 10
) -> TextContent:
    """Get co-occurring tags based on minimum co-occurrence and limit."""
    params = {}
    if min_co_occurrence is not None:
        params["min_co_occurrence"] = min_co_occurrence
    if limit is not None:
        params["limit"] = limit
    result = await _get("/analytics/co_occurring_tags", params=params)
    return _wrap_backend_response("analytics_get_co_occurring_tags", result)


@server.tool()
async def analytics_get_stale_tensors(
    threshold_days: Optional[int] = 90, limit: Optional[int] = 100
) -> TextContent:
    """Get stale tensors based on a threshold of days and limit."""
    params = {}
    if threshold_days is not None:
        params["threshold_days"] = threshold_days
    if limit is not None:
        params["limit"] = limit
    result = await _get("/analytics/stale_tensors", params=params)
    return _wrap_backend_response("analytics_get_stale_tensors", result)


@server.tool()
async def analytics_get_complex_tensors(
    min_parent_count: Optional[int] = None,
    min_transformation_steps: Optional[int] = None,
    limit: Optional[int] = 100,
) -> TextContent:
    """Get complex tensors based on minimum parent count, transformation steps, and limit."""
    params = {}
    if min_parent_count is not None:
        params["min_parent_count"] = min_parent_count
    if min_transformation_steps is not None:
        params["min_transformation_steps"] = min_transformation_steps
    if limit is not None:
        params["limit"] = limit
    result = await _get("/analytics/complex_tensors", params=params)
    return _wrap_backend_response("analytics_get_complex_tensors", result)


def main() -> None:
    global API_BASE_URL, GLOBAL_API_KEY

    parser = argparse.ArgumentParser(
        description="Run the Tensorus FastMCP server exposing dataset and tensor tools"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help=(
            "Transport protocol. SSE is deprecated; use streamable-http for web deployments"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--path", default="/mcp", help="Base path for Streamable HTTP")
    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help="Base URL of the running FastAPI backend",
    )
    parser.add_argument(
        "--mcp-api-key",
        default=None,
        help="Global API key for the MCP server to use for backend requests",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",  # Sets to True if flag is present
        help="Enable demo mode to use mock data instead of real API calls.",
    )
    args = parser.parse_args()

    API_BASE_URL = args.api_url.rstrip("/")
    GLOBAL_API_KEY = args.mcp_api_key

    global DEMO_MODE  # Ensure you're assigning to the global
    DEMO_MODE = args.demo_mode
    if DEMO_MODE:
        print("Tensorus FastMCP Server is running in DEMO MODE.")

    if args.transport == "streamable-http":
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            path=args.path,
            log_level="info",
        )
    else:
        server.run(args.transport)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
