"""FastMCP server exposing Tensorus API endpoints as tools.

This module registers a set of MCP tools that proxy to the Tensorus FastAPI
backend.  Tools mirror the ones documented in the README under "Available
Tools" and return results as :class:`TextContent` objects.
"""

import argparse
import json
from typing import Any, Optional, Sequence, Dict

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

API_BASE_URL = "https://tensorus-core.hf.space"

server = FastMCP(name="Tensorus FastMCP")


async def _post(path: str, payload: dict, params: Optional[Dict[str, Any]] = None) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_BASE_URL}{path}", json=payload, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:  # pragma: no cover - network failures
        return {"error": str(exc)}


async def _get(path: str, params: Optional[Dict[str, Any]] = None) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}{path}", params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:  # pragma: no cover - network failures
        return {"error": str(exc)}


async def _put(path: str, payload: dict, params: Optional[Dict[str, Any]] = None) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(f"{API_BASE_URL}{path}", json=payload, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:  # pragma: no cover - network failures
        return {"error": str(exc)}


async def _delete(path: str) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{API_BASE_URL}{path}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:  # pragma: no cover - network failures
        return {"error": str(exc)}


async def _patch(path: str, payload: dict, params: Optional[Dict[str, Any]] = None) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(f"{API_BASE_URL}{path}", json=payload, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:  # pragma: no cover - network failures
        return {"error": str(exc)}


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


# --- Tensor Descriptor Tools ---

@server.tool()
async def create_tensor_descriptor(descriptor_data: Dict) -> TextContent:
    """Create a new tensor descriptor."""
    result = await _post("/tensor_descriptors/", descriptor_data)
    return TextContent(type="text", text=json.dumps(result))


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
    min_dimensions: Optional[int] = None
) -> TextContent:
    """List tensor descriptors with extensive optional filters."""
    params: Dict[str, Any] = {}
    if owner is not None:
        params["owner"] = owner
    if data_type is not None:
        params["data_type"] = data_type # API uses data_type
    if tags_contain is not None:
        params["tags_contain"] = tags_contain # API uses tags_contain (FastAPI handles comma-separated string to List)
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
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def get_tensor_descriptor(tensor_id: str) -> TextContent:
    """Get a tensor descriptor by its ID."""
    result = await _get(f"/tensor_descriptors/{tensor_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def update_tensor_descriptor(tensor_id: str, updates: Dict) -> TextContent:
    """Update a tensor descriptor by its ID."""
    result = await _put(f"/tensor_descriptors/{tensor_id}", updates)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def delete_tensor_descriptor(tensor_id: str) -> TextContent:
    """Delete a tensor descriptor by its ID."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}")
    return TextContent(type="text", text=json.dumps(result))


# --- Semantic Metadata Tools ---

@server.tool()
async def create_semantic_metadata_for_tensor(tensor_id: str, metadata_in: Dict) -> TextContent:
    """Create semantic metadata for a given tensor descriptor."""
    result = await _post(f"/tensor_descriptors/{tensor_id}/semantic/", metadata_in)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def get_all_semantic_metadata_for_tensor(tensor_id: str) -> TextContent:
    """Get all semantic metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/semantic/")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def update_named_semantic_metadata_for_tensor(tensor_id: str, current_name: str, updates: Dict) -> TextContent:
    """Update a named piece of semantic metadata for a given tensor descriptor."""
    result = await _put(f"/tensor_descriptors/{tensor_id}/semantic/{current_name}", updates)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def delete_named_semantic_metadata_for_tensor(tensor_id: str, name: str) -> TextContent:
    """Delete a named piece of semantic metadata for a given tensor descriptor."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}/semantic/{name}")
    return TextContent(type="text", text=json.dumps(result))


# --- Extended Metadata Tools ---

# --- Lineage Metadata Tools ---
@server.tool()
async def upsert_lineage_metadata(tensor_id: str, metadata_in: Dict) -> TextContent:
    """Upsert lineage metadata for a given tensor descriptor."""
    result = await _post(f"/tensor_descriptors/{tensor_id}/lineage/", metadata_in)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def get_lineage_metadata(tensor_id: str) -> TextContent:
    """Get lineage metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/lineage/")
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def patch_lineage_metadata(tensor_id: str, updates: Dict) -> TextContent:
    """Patch lineage metadata for a given tensor descriptor."""
    result = await _patch(f"/tensor_descriptors/{tensor_id}/lineage/", updates)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def delete_lineage_metadata(tensor_id: str) -> TextContent:
    """Delete lineage metadata for a given tensor descriptor."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}/lineage/")
    return TextContent(type="text", text=json.dumps(result))

# --- Computational Metadata Tools ---
@server.tool()
async def upsert_computational_metadata(tensor_id: str, metadata_in: Dict) -> TextContent:
    """Upsert computational metadata for a given tensor descriptor."""
    result = await _post(f"/tensor_descriptors/{tensor_id}/computational/", metadata_in)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def get_computational_metadata(tensor_id: str) -> TextContent:
    """Get computational metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/computational/")
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def patch_computational_metadata(tensor_id: str, updates: Dict) -> TextContent:
    """Patch computational metadata for a given tensor descriptor."""
    result = await _patch(f"/tensor_descriptors/{tensor_id}/computational/", updates)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def delete_computational_metadata(tensor_id: str) -> TextContent:
    """Delete computational metadata for a given tensor descriptor."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}/computational/")
    return TextContent(type="text", text=json.dumps(result))

# --- Quality Metadata Tools ---
@server.tool()
async def upsert_quality_metadata(tensor_id: str, metadata_in: Dict) -> TextContent:
    """Upsert quality metadata for a given tensor descriptor."""
    result = await _post(f"/tensor_descriptors/{tensor_id}/quality/", metadata_in)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def get_quality_metadata(tensor_id: str) -> TextContent:
    """Get quality metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/quality/")
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def patch_quality_metadata(tensor_id: str, updates: Dict) -> TextContent:
    """Patch quality metadata for a given tensor descriptor."""
    result = await _patch(f"/tensor_descriptors/{tensor_id}/quality/", updates)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def delete_quality_metadata(tensor_id: str) -> TextContent:
    """Delete quality metadata for a given tensor descriptor."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}/quality/")
    return TextContent(type="text", text=json.dumps(result))

# --- Relational Metadata Tools ---
@server.tool()
async def upsert_relational_metadata(tensor_id: str, metadata_in: Dict) -> TextContent:
    """Upsert relational metadata for a given tensor descriptor."""
    result = await _post(f"/tensor_descriptors/{tensor_id}/relational/", metadata_in)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def get_relational_metadata(tensor_id: str) -> TextContent:
    """Get relational metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/relational/")
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def patch_relational_metadata(tensor_id: str, updates: Dict) -> TextContent:
    """Patch relational metadata for a given tensor descriptor."""
    result = await _patch(f"/tensor_descriptors/{tensor_id}/relational/", updates)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def delete_relational_metadata(tensor_id: str) -> TextContent:
    """Delete relational metadata for a given tensor descriptor."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}/relational/")
    return TextContent(type="text", text=json.dumps(result))

# --- Usage Metadata Tools ---
@server.tool()
async def upsert_usage_metadata(tensor_id: str, metadata_in: Dict) -> TextContent:
    """Upsert usage metadata for a given tensor descriptor."""
    result = await _post(f"/tensor_descriptors/{tensor_id}/usage/", metadata_in)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def get_usage_metadata(tensor_id: str) -> TextContent:
    """Get usage metadata for a given tensor descriptor."""
    result = await _get(f"/tensor_descriptors/{tensor_id}/usage/")
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def patch_usage_metadata(tensor_id: str, updates: Dict) -> TextContent:
    """Patch usage metadata for a given tensor descriptor."""
    result = await _patch(f"/tensor_descriptors/{tensor_id}/usage/", updates)
    return TextContent(type="text", text=json.dumps(result))

@server.tool()
async def delete_usage_metadata(tensor_id: str) -> TextContent:
    """Delete usage metadata for a given tensor descriptor."""
    result = await _delete(f"/tensor_descriptors/{tensor_id}/usage/")
    return TextContent(type="text", text=json.dumps(result))


# --- Search and Aggregation Tools ---

@server.tool()
async def search_tensors(
    text_query: str,
    fields_to_search: Optional[str] = None  # Comma-separated string
) -> TextContent:
    """Search for tensors based on a text query, optionally specifying fields to search."""
    params = {"text_query": text_query}
    if fields_to_search:
        params["fields_to_search"] = fields_to_search
    result = await _get("/search/tensors/", params=params)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def aggregate_tensors(
    group_by_field: str,
    agg_function: str,
    agg_field: Optional[str] = None
) -> TextContent:
    """Aggregate tensor metadata based on a grouping field and aggregation function."""
    params = {
        "group_by_field": group_by_field,
        "agg_function": agg_function,
    }
    if agg_field:
        params["agg_field"] = agg_field
    result = await _get("/aggregate/tensors/", params=params)
    return TextContent(type="text", text=json.dumps(result))


# --- Versioning and Lineage Tools ---

@server.tool()
async def create_tensor_version(tensor_id: str, version_request: Dict) -> TextContent:
    """Create a new version for a given tensor."""
    result = await _post(f"/tensors/{tensor_id}/versions", version_request)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def list_tensor_versions(tensor_id: str) -> TextContent:
    """List all versions for a given tensor."""
    result = await _get(f"/tensors/{tensor_id}/versions")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def create_lineage_relationship(relationship_request: Dict) -> TextContent:
    """Create a lineage relationship between tensors."""
    # relationship_request should contain source_tensor_id, target_tensor_id, relationship_type, etc.
    result = await _post("/lineage/relationships/", relationship_request)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def get_parent_tensors(tensor_id: str) -> TextContent:
    """Get the parent tensors for a given tensor in the lineage."""
    result = await _get(f"/tensors/{tensor_id}/lineage/parents")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def get_child_tensors(tensor_id: str) -> TextContent:
    """Get the child tensors for a given tensor in the lineage."""
    result = await _get(f"/tensors/{tensor_id}/lineage/children")
    return TextContent(type="text", text=json.dumps(result))


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
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def import_tensor_metadata(
    import_data_payload: Dict,
    conflict_strategy: Optional[str] = "skip"
) -> TextContent:
    """Import tensor metadata with a specified conflict strategy (skip or overwrite)."""
    params = {"conflict_strategy": conflict_strategy}
    result = await _post("/tensors/import", payload=import_data_payload, params=params)
    return TextContent(type="text", text=json.dumps(result))


# --- Management Tools ---

@server.tool()
async def management_health_check() -> TextContent:
    """Perform a health check on the Tensorus service."""
    result = await _get("/health")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def management_get_metrics() -> TextContent:
    """Retrieve operational metrics from the Tensorus service."""
    result = await _get("/metrics")
    return TextContent(type="text", text=json.dumps(result))


# --- Analytics Tools ---

@server.tool()
async def analytics_get_co_occurring_tags(
    min_co_occurrence: Optional[int] = 2,
    limit: Optional[int] = 10
) -> TextContent:
    """Get co-occurring tags based on minimum co-occurrence and limit."""
    params = {}
    if min_co_occurrence is not None:
        params["min_co_occurrence"] = min_co_occurrence
    if limit is not None:
        params["limit"] = limit
    result = await _get("/analytics/co_occurring_tags", params=params)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def analytics_get_stale_tensors(
    threshold_days: Optional[int] = 90,
    limit: Optional[int] = 100
) -> TextContent:
    """Get stale tensors based on a threshold of days and limit."""
    params = {}
    if threshold_days is not None:
        params["threshold_days"] = threshold_days
    if limit is not None:
        params["limit"] = limit
    result = await _get("/analytics/stale_tensors", params=params)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def analytics_get_complex_tensors(
    min_parent_count: Optional[int] = None,
    min_transformation_steps: Optional[int] = None,
    limit: Optional[int] = 100
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
    return TextContent(type="text", text=json.dumps(result))


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
