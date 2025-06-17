import asyncio
import json

import pandas as pd
import streamlit as st
from tensorus.mcp_client import TensorusMCPClient, DEFAULT_MCP_URL

st.title("Tensorus MCP Client Demo")
st.markdown("Interact with a Tensorus MCP server without writing any code.")

mcp_url = st.text_input("MCP server URL", DEFAULT_MCP_URL)


def run_async(coro):
    return asyncio.run(coro)


st.header("Datasets")
if st.button("List datasets"):

    async def _list():
        async with TensorusMCPClient.from_http(url=mcp_url) as client:
            return await client.list_datasets()

    result = run_async(_list())
    if result:
        st.write(pd.DataFrame(result.datasets, columns=["Datasets"]))

create_name = st.text_input("New dataset name")
if st.button("Create dataset") and create_name:

    async def _create():
        async with TensorusMCPClient.from_http(url=mcp_url) as client:
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
            async with TensorusMCPClient.from_http(url=mcp_url) as client:
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
        async with TensorusMCPClient.from_http(url=mcp_url) as client:
            return await client.execute_nql_query(query)

    result = run_async(_query())
    if isinstance(result.results, list):
        st.write(pd.DataFrame(result.results))
    else:
        st.json(result.results)
