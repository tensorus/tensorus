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

# Legacy Tensorus releases might not provide the ``from_http`` helper.
# ``_make_client`` abstracts client creation so the rest of the demo works
# across versions. We first try ``from_http`` if available.  If not, we fall
# back to calling the constructor directly with the URL and, as a last resort,
# manually constructing the transport using ``fastmcp``.
def _make_client(url: str) -> TensorusMCPClient:
    if hasattr(TensorusMCPClient, "from_http"):
        return TensorusMCPClient.from_http(url=url)
    try:  # Older versions may accept the URL directly
        return TensorusMCPClient(url)  # type: ignore[arg-type]
    except Exception:
        try:
            from fastmcp.client import Client as FastMCPClient
            from fastmcp.client.transports import StreamableHttpTransport

            transport = StreamableHttpTransport(url=url.rstrip("/"))
            return TensorusMCPClient(FastMCPClient(transport))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - very old versions
            raise RuntimeError("Unsupported TensorusMCPClient version") from exc

st.title("Tensorus MCP Client Demo")
st.markdown("Interact with a Tensorus MCP server without writing any code.")

mcp_url = st.text_input("MCP server URL", DEFAULT_MCP_URL)


def run_async(coro):
    return asyncio.run(coro)


st.header("Datasets")
if st.button("List datasets"):

    async def _list():
        async with _make_client(mcp_url) as client:
            return await client.list_datasets()

    result = run_async(_list())
    if result:
        st.write(pd.DataFrame(result.datasets, columns=["Datasets"]))

create_name = st.text_input("New dataset name")
if st.button("Create dataset") and create_name:

    async def _create():
        async with _make_client(mcp_url) as client:
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
            async with _make_client(mcp_url) as client:
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
        async with _make_client(mcp_url) as client:
            return await client.execute_nql_query(query)

    result = run_async(_query())
    if isinstance(result.results, list):
        st.write(pd.DataFrame(result.results))
    else:
        st.json(result.results)
