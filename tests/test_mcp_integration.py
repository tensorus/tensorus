import pytest
import pytest_asyncio
import subprocess
import time
import httpx
import asyncio
import sys
import uuid
import os # Added import

from tensorus.mcp_client import (
    TensorusMCPClient,
    DatasetListResponse,
    CreateDatasetResponse,
    DeleteDatasetResponse,
    IngestTensorResponse,
    TensorDetailsResponse,
    DeleteTensorResponse
)

TEST_API_KEY = "test_integration_key_123abc" # Added test API key

@pytest_asyncio.fixture(scope="session")
async def mcp_servers():
    api_process = None
    mcp_server_process = None
    api_url = "http://localhost:8000"
    mcp_url = "http://localhost:7860/mcp"

    api_logs = []
    mcp_logs = []

    try:
        print("\nStarting FastAPI backend...")
        api_env = os.environ.copy()
        api_env["TENSORUS_VALID_API_KEYS"] = TEST_API_KEY
        api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "tensorus.api:app", "--port", "8000", "--log-level", "warning"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=api_env
        )

        print("Starting MCP server...")
        mcp_server_process = subprocess.Popen(
            [
                sys.executable,
                "tensorus/mcp_server.py",
                "--port", "7860",
                "--api-url", api_url,
                "--transport", "streamable-http",
                "--mcp-api-key", TEST_API_KEY
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        max_wait_total = 120
        poll_interval = 2
        start_time = time.time()
        mcp_ready = False

        print(f"Waiting for MCP server at {mcp_url}...")
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < max_wait_total:
                try:
                    try:
                        # Client used for readiness check does not need API key for list_datasets if it's public
                        # However, if list_datasets becomes protected, this check might need an api_key.
                        # For now, assuming list_datasets called by the fixture for readiness check
                        # doesn't strictly require the key from the client perspective for this simple check.
                        # The key purpose here is to ensure the *servers* are configured with keys.
                        test_client = TensorusMCPClient.from_http(url=mcp_url, api_key=TEST_API_KEY)
                        async with test_client:
                            await test_client.list_datasets() # list_datasets itself doesn't take api_key param
                        print("MCP server responded to list_datasets. Ready.")
                        mcp_ready = True
                        break
                    except Exception as e:
                        print(f"MCP server client check failed: {e}. Retrying...")
                        await asyncio.sleep(poll_interval)

                except httpx.ConnectError:
                    print(f"MCP Server not responding yet at {mcp_url}. Retrying...")
                    await asyncio.sleep(poll_interval)
                except Exception as e:
                    print(f"An error occurred while checking MCP server: {e}. Retrying...")
                    await asyncio.sleep(poll_interval)

        if not mcp_ready:
            if api_process:
                api_process.terminate()
                stdout, stderr = api_process.communicate(timeout=5)
                api_logs.append("API STDOUT:\n" + stdout)
                api_logs.append("API STDERR:\n" + stderr)
            if mcp_server_process:
                mcp_server_process.terminate()
                stdout, stderr = mcp_server_process.communicate(timeout=5)
                mcp_logs.append("MCP STDOUT:\n" + stdout)
                mcp_logs.append("MCP STDERR:\n" + stderr)

            log_output = "\n".join(api_logs) + "\n" + "\n".join(mcp_logs)
            raise RuntimeError(f"MCP server did not start or become ready in time. Logs:\n{log_output}")

        yield api_url, mcp_url

    finally:
        print("\nTearing down servers...")
        processes_terminated = True
        if mcp_server_process:
            print("Terminating MCP server...")
            mcp_server_process.terminate()
            try:
                stdout, stderr = mcp_server_process.communicate(timeout=10)
                mcp_logs.append("MCP STDOUT at shutdown:\n" + stdout)
                mcp_logs.append("MCP STDERR at shutdown:\n" + stderr)
            except subprocess.TimeoutExpired:
                print("MCP server communicate timeout, killing.")
                mcp_server_process.kill()
                processes_terminated = False
            if mcp_server_process.poll() is None:
                print("MCP server did not terminate gracefully, killing.")
                mcp_server_process.kill()
                processes_terminated = False

        if api_process:
            print("Terminating FastAPI backend...")
            api_process.terminate()
            try:
                stdout, stderr = api_process.communicate(timeout=10)
                api_logs.append("API STDOUT at shutdown:\n" + stdout)
                api_logs.append("API STDERR at shutdown:\n" + stderr)
            except subprocess.TimeoutExpired:
                print("API server communicate timeout, killing.")
                api_process.kill()
                processes_terminated = False
            if api_process.poll() is None:
                print("API server did not terminate gracefully, killing.")
                api_process.kill()
                processes_terminated = False

        print("\nCaptured API logs:\n" + "\n".join(api_logs))
        print("\nCaptured MCP logs:\n" + "\n".join(mcp_logs))

        if not processes_terminated:
            print("Warning: One or more server processes had to be killed forcefully.")
        print("Servers teardown finished.")


@pytest.mark.asyncio
async def test_list_datasets_integration(mcp_servers):
    _, mcp_url = mcp_servers

    async with TensorusMCPClient.from_http(url=mcp_url, api_key=TEST_API_KEY) as client:
        datasets_response = await client.list_datasets()
        assert isinstance(datasets_response, DatasetListResponse), f"Expected DatasetListResponse, got {type(datasets_response)}"
        assert 'count_ds' in datasets_response.datasets
        print(f"test_list_datasets_integration: Successfully asserted dataset list: {datasets_response.datasets}")

@pytest.mark.asyncio
async def test_create_and_delete_dataset_integration(mcp_servers):
    _, mcp_url = mcp_servers
    dataset_name = f"test_dataset_integration_{uuid.uuid4()}"

    async with TensorusMCPClient.from_http(url=mcp_url, api_key=TEST_API_KEY) as client:
        # Create dataset
        create_response = await client.create_dataset(dataset_name)
        assert isinstance(create_response, CreateDatasetResponse)
        assert create_response.success is True
        print(f"Dataset '{dataset_name}' creation reported success: {create_response.success}")

        # Verify dataset is listed
        list_response_after_create = await client.list_datasets()
        assert isinstance(list_response_after_create, DatasetListResponse)
        assert dataset_name in list_response_after_create.datasets
        print(f"Dataset '{dataset_name}' found in list: {list_response_after_create.datasets}")

        # Delete dataset
        delete_response = await client.delete_dataset(dataset_name)
        assert isinstance(delete_response, DeleteDatasetResponse)
        assert delete_response.success is True
        print(f"Dataset '{dataset_name}' deletion reported success: {delete_response.success}")

        # Verify dataset is no longer listed
        list_response_after_delete = await client.list_datasets()
        assert isinstance(list_response_after_delete, DatasetListResponse)
        assert dataset_name not in list_response_after_delete.datasets
        print(f"Dataset '{dataset_name}' not found in list after deletion: {list_response_after_delete.datasets}")

@pytest.mark.asyncio
async def test_ingest_and_get_tensor_integration(mcp_servers):
    _, mcp_url = mcp_servers
    dataset_name = f"test_dataset_ingest_{uuid.uuid4()}"

    tensor_shape = [2, 2]
    tensor_dtype = "float32"
    tensor_data = [[1.0, 2.0], [3.0, 4.0]]
    metadata = {"source": "integration_test", "version": 1.0}

    async with TensorusMCPClient.from_http(url=mcp_url, api_key=TEST_API_KEY) as client:
        # Create dataset
        create_ds_response = await client.create_dataset(dataset_name)
        assert create_ds_response.success is True
        print(f"Dataset '{dataset_name}' created for tensor ingestion.")

        # Ingest tensor
        ingest_response = await client.ingest_tensor(
            dataset_name=dataset_name,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            tensor_data=tensor_data,
            metadata=metadata
        )
        assert isinstance(ingest_response, IngestTensorResponse)
        assert ingest_response.id is not None
        assert ingest_response.status == "ingested"
        record_id = ingest_response.id
        print(f"Tensor ingested with record_id: {record_id}, status: {ingest_response.status}")

        # Get tensor details
        details_response = await client.get_tensor_details(dataset_name, record_id)
        assert isinstance(details_response, TensorDetailsResponse)
        assert details_response.id == record_id
        assert details_response.shape == tensor_shape
        assert details_response.dtype == tensor_dtype
        assert details_response.data == tensor_data
        # Check if all original metadata items are present in the retrieved metadata
        for key, value in metadata.items():
            assert key in details_response.metadata
            assert details_response.metadata[key] == value
        print(f"Tensor details retrieved and assertions passed for record_id: {record_id}")

        # Clean up tensor
        delete_tensor_response = await client.delete_tensor(dataset_name, record_id)
        assert isinstance(delete_tensor_response, DeleteTensorResponse)
        assert delete_tensor_response.success is True
        print(f"Tensor record_id: {record_id} deleted successfully.")

        # Clean up dataset
        delete_ds_response = await client.delete_dataset(dataset_name)
        assert isinstance(delete_ds_response, DeleteDatasetResponse)
        assert delete_ds_response.success is True
        print(f"Dataset '{dataset_name}' deleted successfully after tensor operations.")
