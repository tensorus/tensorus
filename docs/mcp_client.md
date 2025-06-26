# MCP Client Methods

The `TensorusMCPClient` in `tensorus.mcp_client` provides an asynchronous interface to the
Model Context Protocol server shipped with Tensorus.  It wraps the underlying
`fastmcp` client and exposes convenience methods for each API tool.

Use it like so:

```python
from tensorus.mcp_client import TensorusMCPClient

async with TensorusMCPClient.from_http() as client:
    datasets = await client.list_datasets()
    print(datasets)
```

Below is a summary of all available methods.  See the
[API Guide](api_guide.md) and [API Endpoints](../README.md#api-endpoints) in the
README for details on the REST endpoints each method calls.

## Server Introspection

- `list_tools()` – retrieve the list of tool names available from the MCP
  server.
  ```python
  tool_names = await client.list_tools()
  ```

## Dataset Management

- `list_datasets()` – list existing datasets.
  ```python
  datasets = await client.list_datasets()
  ```
- `create_dataset(name)` – create a new dataset.
  ```python
  await client.create_dataset("my_ds")
  ```
- `delete_dataset(name)` – remove a dataset.
  ```python
  await client.delete_dataset("my_ds")
  ```

## Tensor Management

- `ingest_tensor(dataset_name, tensor_shape, tensor_dtype, tensor_data, metadata=None)` –
  store a tensor and optional metadata.
  ```python
  await client.ingest_tensor("my_ds", [2, 2], "float32", [[1,2],[3,4]])
  ```
- `get_tensor_details(dataset_name, record_id)` – fetch a tensor and its metadata.
  ```python
  info = await client.get_tensor_details("my_ds", record_id)
  ```
- `delete_tensor(dataset_name, record_id)` – delete a tensor.
  ```python
  await client.delete_tensor("my_ds", record_id)
  ```
- `update_tensor_metadata(dataset_name, record_id, new_metadata)` – update metadata for a tensor.
  ```python
  await client.update_tensor_metadata("my_ds", record_id, {"tag": "updated"})
  ```

## Tensor Operations

- `apply_unary_operation(operation, payload)` – apply unary operations such as `log` or `sum`.
  ```python
  await client.apply_unary_operation("log", {"dataset_name": "my_ds", "record_id": record_id})
  ```
- `apply_binary_operation(operation, payload)` – binary operations like `add` or `matmul`.
  ```python
  await client.apply_binary_operation("add", {"a": {...}, "b": {...}})
  ```
- `apply_list_operation(operation, payload)` – operations on a list of tensors (e.g. `concatenate`).
- `apply_einsum(payload)` – execute an Einstein summation.

## Tensor Descriptor CRUD

- `create_tensor_descriptor(data)` – create a descriptor entry.
- `list_tensor_descriptors(**filters)` – list descriptors with optional filters.
- `get_tensor_descriptor(tensor_id)` – fetch descriptor details.
- `update_tensor_descriptor(tensor_id, updates)` – modify a descriptor.
- `delete_tensor_descriptor(tensor_id)` – delete a descriptor.

## Semantic Metadata

- `create_semantic_metadata_for_tensor(tensor_id, metadata_in)` – add semantic metadata.
- `get_all_semantic_metadata_for_tensor(tensor_id)` – list semantic metadata.
- `update_named_semantic_metadata_for_tensor(tensor_id, current_name, updates)` – update an entry.
- `delete_named_semantic_metadata_for_tensor(tensor_id, name)` – remove an entry.

## Extended Metadata

- `upsert_lineage_metadata(tensor_id, metadata_in)`
- `get_lineage_metadata(tensor_id)`
- `patch_lineage_metadata(tensor_id, updates)`
- `delete_lineage_metadata(tensor_id)`
- `upsert_computational_metadata(tensor_id, metadata_in)`
- `get_computational_metadata(tensor_id)`
- `patch_computational_metadata(tensor_id, updates)`
- `delete_computational_metadata(tensor_id)`
- `upsert_quality_metadata(tensor_id, metadata_in)`
- `get_quality_metadata(tensor_id)`
- `patch_quality_metadata(tensor_id, updates)`
- `delete_quality_metadata(tensor_id)`
- `upsert_relational_metadata(tensor_id, metadata_in)`
- `get_relational_metadata(tensor_id)`
- `patch_relational_metadata(tensor_id, updates)`
- `delete_relational_metadata(tensor_id)`
- `upsert_usage_metadata(tensor_id, metadata_in)`
- `get_usage_metadata(tensor_id)`
- `patch_usage_metadata(tensor_id, updates)`
- `delete_usage_metadata(tensor_id)`

## Search and Aggregation

- `search_tensors(text_query, fields_to_search=None)`
- `aggregate_tensors(group_by_field, agg_function, agg_field=None)`

## Versioning and Lineage

- `create_tensor_version(tensor_id, version_request)`
- `list_tensor_versions(tensor_id)`
- `create_lineage_relationship(relationship_request)`
- `get_parent_tensors(tensor_id)`
- `get_child_tensors(tensor_id)`

## Import/Export and Management

- `export_tensor_metadata(tensor_ids_str=None)`
- `import_tensor_metadata(import_data_payload, conflict_strategy="skip")`
- `management_health_check()`
- `management_get_metrics()`

## Analytics

- `analytics_get_co_occurring_tags(min_co_occurrence=2, limit=10)`
- `analytics_get_stale_tensors(threshold_days=90, limit=100)`
- `analytics_get_complex_tensors(min_parent_count=None, min_transformation_steps=None, limit=100)`

## Miscellaneous

- `execute_nql_query(query)` – run an NQL query.

