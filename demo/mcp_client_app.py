import asyncio
import json

import pandas as pd
import streamlit as st

# Tensorus releases prior to 0.2.0 did not export DEFAULT_MCP_URL at the
# module level. We attempt to import it, but gracefully fall back to the class
# attribute or a hard-coded default so the demo works with older versions.
try:
    from tensorus.mcp_client import TensorusMCPClient, DEFAULT_MCP_URL
except ImportError:  # pragma: no cover - fallback for old versions
    from tensorus.mcp_client import TensorusMCPClient
    DEFAULT_MCP_URL = getattr(
        TensorusMCPClient, "DEFAULT_MCP_URL", "https://tensorus-mcp.hf.space/mcp/"
    )

st.title("Tensorus MCP Client Demo")
st.markdown("Interact with a Tensorus MCP server without writing any code.")

mcp_url = st.text_input("MCP server URL", DEFAULT_MCP_URL)
api_key = st.text_input("API Key (Optional)", type="password", key="api_key_input")


def run_async(coro):
    return asyncio.run(coro)


def parse_json_field(text: str):
    """Parse a text field as JSON if not empty."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


st.header("Datasets")
if st.button("List datasets"):

    async def _list():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.list_datasets()

    result = run_async(_list())
    if result:
        st.write(pd.DataFrame(result.datasets, columns=["Datasets"]))

create_name = st.text_input("New dataset name")
if st.button("Create dataset") and create_name:

    async def _create():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.create_dataset(create_name)

    res = run_async(_create())
    if res:
        st.success(res.message or "Dataset created")

st.header("Ingest Tensor")
with st.form("ingest"):
    ingest_ds = st.text_input("Dataset", key="ingest_ds")
    tensor_shape = st.text_input("Tensor shape", value="2,2")
    tensor_dtype = st.text_input("Tensor dtype", value="float32")
    tensor_data = st.text_area("Tensor data (JSON)", value="[[0, 0], [1, 1]]")
    metadata = st.text_area("Metadata (JSON)", value="{}")
    submitted = st.form_submit_button("Ingest")

if submitted:
    try:
        shape = [int(x) for x in tensor_shape.split(",") if x.strip()]
        data = json.loads(tensor_data)
        meta = json.loads(metadata) if metadata.strip() else None

        async def _ingest():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await client.ingest_tensor(
                    dataset_name=ingest_ds,
                    tensor_shape=shape,
                    tensor_dtype=tensor_dtype,
                    tensor_data=data,
                    metadata=meta,
                )

        response = run_async(_ingest())
        st.write(response)
        st.success(f"Ingested tensor {response.id}")
    except Exception as e:
        st.error(f"Failed to ingest: {e}")

st.header("Run NQL Query")
query = st.text_input("Query", key="nql_query")
if st.button("Execute") and query:

    async def _query():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.execute_nql_query(query)

    result = run_async(_query())
    if isinstance(result.results, list):
        st.write(pd.DataFrame(result.results))
    else:
        st.json(result.results)

st.header("Tensor Descriptors")

st.subheader("Create Tensor Descriptor")
with st.form("create_td"):
    descriptor_data = st.text_area("Descriptor data (JSON)", value="{}")
    submitted_create_td = st.form_submit_button("Create")

if submitted_create_td:
    data = parse_json_field(descriptor_data) or {}

    async def _create_td():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.create_tensor_descriptor(data)

    res = run_async(_create_td())
    st.json(res.dict())

st.subheader("List Tensor Descriptors")
with st.form("list_tds"):
    owner = st.text_input("owner", key="owner")
    data_type = st.text_input("data_type")
    tags_contain = st.text_input("tags_contain")
    lineage_version = st.text_input("lineage_version")
    lineage_source_type = st.text_input("lineage_source_type")
    comp_algorithm = st.text_input("comp_algorithm")
    comp_gpu_model = st.text_input("comp_gpu_model")
    quality_confidence_gt = st.text_input("quality_confidence_gt")
    quality_noise_lt = st.text_input("quality_noise_lt")
    rel_collection = st.text_input("rel_collection")
    rel_has_related_tensor_id = st.text_input("rel_has_related_tensor_id")
    usage_last_accessed_before = st.text_input("usage_last_accessed_before")
    usage_used_by_app = st.text_input("usage_used_by_app")
    name = st.text_input("name")
    description = st.text_input("description")
    min_dimensions = st.text_input("min_dimensions")
    submitted_list_tds = st.form_submit_button("List")

if submitted_list_tds:
    params = {
        "owner": owner or None,
        "data_type": data_type or None,
        "tags_contain": tags_contain or None,
        "lineage_version": lineage_version or None,
        "lineage_source_type": lineage_source_type or None,
        "comp_algorithm": comp_algorithm or None,
        "comp_gpu_model": comp_gpu_model or None,
        "quality_confidence_gt": float(quality_confidence_gt) if quality_confidence_gt else None,
        "quality_noise_lt": float(quality_noise_lt) if quality_noise_lt else None,
        "rel_collection": rel_collection or None,
        "rel_has_related_tensor_id": rel_has_related_tensor_id or None,
        "usage_last_accessed_before": usage_last_accessed_before or None,
        "usage_used_by_app": usage_used_by_app or None,
        "name": name or None,
        "description": description or None,
        "min_dimensions": int(min_dimensions) if min_dimensions else None,
    }

    async def _list_tds():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.list_tensor_descriptors(**{k: v for k, v in params.items() if v is not None})

    res = run_async(_list_tds())
    st.json(res)

st.subheader("Get Tensor Descriptor")
with st.form("get_td"):
    g_tensor_id = st.text_input("tensor_id", key="g_tensor_id")
    submitted_get_td = st.form_submit_button("Get")

if submitted_get_td:
    async def _get_td():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.get_tensor_descriptor(g_tensor_id)

    res = run_async(_get_td())
    st.json(res.dict())

st.subheader("Update Tensor Descriptor")
with st.form("update_td"):
    u_tensor_id = st.text_input("tensor_id", key="u_tensor_id")
    updates = st.text_area("updates (JSON)", value="{}")
    submitted_update_td = st.form_submit_button("Update")

if submitted_update_td:
    updates_data = parse_json_field(updates) or {}

    async def _update_td():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.update_tensor_descriptor(u_tensor_id, updates_data)

    res = run_async(_update_td())
    st.json(res.dict())

st.subheader("Delete Tensor Descriptor")
with st.form("delete_td"):
    d_tensor_id = st.text_input("tensor_id", key="d_tensor_id")
    submitted_delete_td = st.form_submit_button("Delete")

if submitted_delete_td:
    async def _delete_td():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.delete_tensor_descriptor(d_tensor_id)

    res = run_async(_delete_td())
    st.json(res.dict())

st.header("Semantic Metadata")

st.subheader("Create Semantic Metadata")
with st.form("create_sem_meta"):
    sem_tensor_id = st.text_input("tensor_id", key="sem_tensor_id")
    sem_data = st.text_area("metadata_in (JSON)", value="{}")
    submitted_sem_create = st.form_submit_button("Create")

if submitted_sem_create:
    sem_payload = parse_json_field(sem_data) or {}

    async def _create_sem():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.create_semantic_metadata_for_tensor(sem_tensor_id, sem_payload)

    res = run_async(_create_sem())
    st.json(res.dict())

st.subheader("List Semantic Metadata")
with st.form("list_sem_meta"):
    list_sem_tensor_id = st.text_input("tensor_id", key="list_sem_tensor_id")
    submitted_list_sem = st.form_submit_button("List")

if submitted_list_sem:
    async def _list_sem():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.get_all_semantic_metadata_for_tensor(list_sem_tensor_id)

    res = run_async(_list_sem())
    st.json(res)

st.subheader("Update Named Semantic Metadata")
with st.form("update_sem_meta"):
    upd_sem_tensor_id = st.text_input("tensor_id", key="upd_sem_tensor_id")
    current_name = st.text_input("current_name")
    updates_sem = st.text_area("updates (JSON)", value="{}")
    submitted_update_sem = st.form_submit_button("Update")

if submitted_update_sem:
    updates_sem_data = parse_json_field(updates_sem) or {}

    async def _update_sem():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.update_named_semantic_metadata_for_tensor(
                upd_sem_tensor_id, current_name, updates_sem_data
            )

    res = run_async(_update_sem())
    st.json(res.dict())

st.subheader("Delete Named Semantic Metadata")
with st.form("delete_sem_meta"):
    del_sem_tensor_id = st.text_input("tensor_id", key="del_sem_tensor_id")
    del_name = st.text_input("name")
    submitted_delete_sem = st.form_submit_button("Delete")

if submitted_delete_sem:
    async def _delete_sem():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.delete_named_semantic_metadata_for_tensor(del_sem_tensor_id, del_name)

    res = run_async(_delete_sem())
    st.json(res.dict())

st.header("Extended Metadata")

def metadata_forms(prefix: str, upsert_func, get_func, patch_func, delete_func):
    st.subheader(prefix.capitalize())
    with st.form(f"{prefix}_upsert"):
        m_tensor_id = st.text_input("tensor_id", key=f"{prefix}_upsert_id")
        metadata_in = st.text_area("metadata_in (JSON)", value="{}")
        submitted_upsert = st.form_submit_button("Upsert")

    if submitted_upsert:
        payload = parse_json_field(metadata_in) or {}

        async def _upsert():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await upsert_func(client, m_tensor_id, payload)

        res = run_async(_upsert())
        st.json(res.dict())

    with st.form(f"{prefix}_get"):
        g_id = st.text_input("tensor_id", key=f"{prefix}_get_id")
        submitted_get = st.form_submit_button("Get")

    if submitted_get:
        async def _get():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await get_func(client, g_id)

        res = run_async(_get())
        st.json(res.dict())

    with st.form(f"{prefix}_patch"):
        p_id = st.text_input("tensor_id", key=f"{prefix}_patch_id")
        updates = st.text_area("updates (JSON)", value="{}")
        submitted_patch = st.form_submit_button("Patch")

    if submitted_patch:
        upd = parse_json_field(updates) or {}

        async def _patch():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await patch_func(client, p_id, upd)

        res = run_async(_patch())
        st.json(res.dict())

    with st.form(f"{prefix}_delete"):
        d_id = st.text_input("tensor_id", key=f"{prefix}_delete_id")
        submitted_del = st.form_submit_button("Delete")

    if submitted_del:
        async def _del():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await delete_func(client, d_id)

        res = run_async(_del())
        st.json(res.dict())


metadata_forms(
    "lineage",
    lambda c, i, d: c.upsert_lineage_metadata(i, d),
    lambda c, i: c.get_lineage_metadata(i),
    lambda c, i, u: c.patch_lineage_metadata(i, u),
    lambda c, i: c.delete_lineage_metadata(i),
)

metadata_forms(
    "computational",
    lambda c, i, d: c.upsert_computational_metadata(i, d),
    lambda c, i: c.get_computational_metadata(i),
    lambda c, i, u: c.patch_computational_metadata(i, u),
    lambda c, i: c.delete_computational_metadata(i),
)

metadata_forms(
    "quality",
    lambda c, i, d: c.upsert_quality_metadata(i, d),
    lambda c, i: c.get_quality_metadata(i),
    lambda c, i, u: c.patch_quality_metadata(i, u),
    lambda c, i: c.delete_quality_metadata(i),
)

metadata_forms(
    "relational",
    lambda c, i, d: c.upsert_relational_metadata(i, d),
    lambda c, i: c.get_relational_metadata(i),
    lambda c, i, u: c.patch_relational_metadata(i, u),
    lambda c, i: c.delete_relational_metadata(i),
)

metadata_forms(
    "usage",
    lambda c, i, d: c.upsert_usage_metadata(i, d),
    lambda c, i: c.get_usage_metadata(i),
    lambda c, i, u: c.patch_usage_metadata(i, u),
    lambda c, i: c.delete_usage_metadata(i),
)

st.header("Search & Aggregation")

st.subheader("Search Tensors")
with st.form("search_tensors"):
    text_query = st.text_input("text_query")
    fields_to_search = st.text_input("fields_to_search")
    submitted_search = st.form_submit_button("Search")

if submitted_search:
    async def _search():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.search_tensors(text_query, fields_to_search or None)

    res = run_async(_search())
    st.json(res)

st.subheader("Aggregate Tensors")
with st.form("aggregate_tensors"):
    group_by_field = st.text_input("group_by_field")
    agg_function = st.text_input("agg_function")
    agg_field = st.text_input("agg_field")
    submitted_agg = st.form_submit_button("Aggregate")

if submitted_agg:
    async def _agg():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.aggregate_tensors(
                group_by_field,
                agg_function,
                agg_field or None,
            )

    res = run_async(_agg())
    st.json(res)

st.header("Versioning & Lineage")

st.subheader("Create Tensor Version")
with st.form("create_version"):
    ver_tensor_id = st.text_input("tensor_id", key="ver_tensor_id")
    version_request = st.text_area("version_request (JSON)", value="{}")
    submitted_version = st.form_submit_button("Create")

if submitted_version:
    vr = parse_json_field(version_request) or {}

    async def _create_ver():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.create_tensor_version(ver_tensor_id, vr)

    res = run_async(_create_ver())
    st.json(res)

st.subheader("List Tensor Versions")
with st.form("list_versions"):
    list_ver_tensor_id = st.text_input("tensor_id", key="list_ver_tensor_id")
    submitted_list_ver = st.form_submit_button("List")

if submitted_list_ver:
    async def _list_ver():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.list_tensor_versions(list_ver_tensor_id)

    res = run_async(_list_ver())
    st.json(res)

st.subheader("Create Lineage Relationship")
with st.form("create_relationship"):
    rel_request = st.text_area("relationship_request (JSON)", value="{}")
    submitted_rel = st.form_submit_button("Create")

if submitted_rel:
    req = parse_json_field(rel_request) or {}

    async def _create_rel():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.create_lineage_relationship(req)

    res = run_async(_create_rel())
    st.json(res)

st.subheader("Get Parent Tensors")
with st.form("get_parents"):
    parent_tensor_id = st.text_input("tensor_id", key="parent_tensor_id")
    submitted_parents = st.form_submit_button("Get")

if submitted_parents:
    async def _parents():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.get_parent_tensors(parent_tensor_id)

    res = run_async(_parents())
    st.json(res)

st.subheader("Get Child Tensors")
with st.form("get_children"):
    child_tensor_id = st.text_input("tensor_id", key="child_tensor_id")
    submitted_children = st.form_submit_button("Get")

if submitted_children:
    async def _children():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.get_child_tensors(child_tensor_id)

    res = run_async(_children())
    st.json(res)

st.header("Import/Export & Management")

st.subheader("Export Tensor Metadata")
with st.form("export_metadata"):
    tensor_ids_str = st.text_input("tensor_ids (comma separated)")
    submitted_export = st.form_submit_button("Export")

if submitted_export:
    async def _export():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.export_tensor_metadata(tensor_ids_str or None)

    res = run_async(_export())
    st.json(res)

st.subheader("Import Tensor Metadata")
with st.form("import_metadata"):
    import_payload = st.text_area("import_data_payload (JSON)", value="{}")
    conflict_strategy = st.text_input("conflict_strategy", value="skip")
    submitted_import = st.form_submit_button("Import")

if submitted_import:
    payload = parse_json_field(import_payload) or {}

    async def _import():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.import_tensor_metadata(payload, conflict_strategy or None)

    res = run_async(_import())
    st.json(res)

col1, col2 = st.columns(2)
with col1:
    if st.button("Management Health Check"):
        async def _health():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await client.management_health_check()

        st.json(run_async(_health()))

with col2:
    if st.button("Management Get Metrics"):
        async def _metrics():
            async with TensorusMCPClient.from_http(
                url=mcp_url,
                auth_token=api_key if api_key else None,
                auth_header_name="X-API-KEY"
            ) as client:
                return await client.management_get_metrics()

        st.json(run_async(_metrics()))

st.header("Analytics")

st.subheader("Co-occurring Tags")
with st.form("analytics_co_tags"):
    min_co_occurrence = st.text_input("min_co_occurrence", value="2")
    limit_co = st.text_input("limit", value="10")
    submitted_co = st.form_submit_button("Get")

if submitted_co:
    async def _co():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.analytics_get_co_occurring_tags(
                int(min_co_occurrence) if min_co_occurrence else None,
                int(limit_co) if limit_co else None,
            )

    st.json(run_async(_co()))

st.subheader("Stale Tensors")
with st.form("analytics_stale"):
    threshold_days = st.text_input("threshold_days", value="90")
    limit_stale = st.text_input("limit", value="100")
    submitted_stale = st.form_submit_button("Get")

if submitted_stale:
    async def _stale():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.analytics_get_stale_tensors(
                int(threshold_days) if threshold_days else None,
                int(limit_stale) if limit_stale else None,
            )

    st.json(run_async(_stale()))

st.subheader("Complex Tensors")
with st.form("analytics_complex"):
    min_parent_count = st.text_input("min_parent_count")
    min_steps = st.text_input("min_transformation_steps")
    limit_complex = st.text_input("limit", value="100")
    submitted_complex = st.form_submit_button("Get")

if submitted_complex:
    async def _complex():
        async with TensorusMCPClient.from_http(
            url=mcp_url,
            auth_token=api_key if api_key else None,
            auth_header_name="X-API-KEY"
        ) as client:
            return await client.analytics_get_complex_tensors(
                int(min_parent_count) if min_parent_count else None,
                int(min_steps) if min_steps else None,
                int(limit_complex) if limit_complex else None,
            )

    st.json(run_async(_complex()))

if __name__ == "__main__":
    print("Attempting to list datasets automatically...")
    async def _list_for_testing():
        async with TensorusMCPClient.from_http(url=DEFAULT_MCP_URL, auth_token=None) as client: # Use DEFAULT_MCP_URL
            print("Client created, listing datasets...")
            datasets = await client.list_datasets()
            print(f"Successfully listed datasets: {datasets}")
            return datasets

    try:
        asyncio.run(_list_for_testing())
        print("Test: Successfully listed datasets without HTTPStatusError.")
    except Exception as e:
        print(f"Test: Encountered an error: {e}")
        # Re-raise the exception to ensure the test fails if an error occurs
        raise
