import pytest
pytest.importorskip("torch")
pytest.importorskip("httpx")
pytest.skip("API tests require FastAPI test client", allow_module_level=True)
import torch
from fastapi.testclient import TestClient

# Assuming your FastAPI app instance is named 'app' in 'api.py'
# and tensor_storage_instance is the global TensorStorage.
from tensorus.api import app, tensor_storage_instance

client = TestClient(app)

# Helper to manage datasets created during tests
# A more robust solution might involve a temporary storage backend for TensorStorage
# or more specific fixtures for dataset/tensor creation and cleanup.
TEST_DATASETS = set()

def _cleanup_test_datasets():
    # print(f"Cleaning up test datasets: {list(TEST_DATASETS)}")
    for ds_name in list(TEST_DATASETS):
        try:
            if tensor_storage_instance.dataset_exists(ds_name):
                # print(f"Attempting to delete dataset: {ds_name}")
                # This requires delete_dataset to be robust.
                # If delete_dataset is an API call, this helper becomes more complex.
                # For now, assuming direct access to tensor_storage_instance for cleanup.
                tensor_storage_instance.delete_dataset(ds_name)
                # print(f"Successfully deleted dataset: {ds_name}")
        except Exception as e:
            print(f"Error cleaning up dataset {ds_name}: {e}")
        finally:
            TEST_DATASETS.discard(ds_name)
    # print("Cleanup complete.")


@pytest.fixture(autouse=True)
def auto_cleanup_datasets(request):
    """Automatically clean up datasets after each test."""
    # No setup needed before test
    yield
    # Teardown after test
    _cleanup_test_datasets()


def _ingest_tensor_for_test(client: TestClient, dataset_name: str, record_id_hint: str, shape: list, dtype: str, data: list, metadata: dict = None) -> str:
    """Helper function to ingest a tensor and return its record_id."""
    if not tensor_storage_instance.dataset_exists(dataset_name):
        client.post("/datasets/create", json={"name": dataset_name})
        TEST_DATASETS.add(dataset_name) # Track for cleanup

    # Use record_id_hint to make it easier to predict/check record_id if needed,
    # though the API currently might generate its own.
    # For now, the Python API's ingest endpoint generates the record_id.
    payload = {
        "shape": shape,
        "dtype": dtype,
        "data": data,
        "metadata": metadata or {"source": "test", "record_hint": record_id_hint}
    }
    response = client.post(f"/datasets/{dataset_name}/ingest", json=payload)
    assert response.status_code == 201
    record_id = response.json()["data"]["record_id"]
    return record_id

# --- TensorStorage Management Endpoint Tests ---

def test_get_tensor_by_id_api():
    dataset_name = "test_get_ds"
    tensor_data = [[1.0, 2.0], [3.0, 4.0]]
    record_id = _ingest_tensor_for_test(client, dataset_name, "t1", [2,2], "float32", tensor_data)

    # Get existing tensor
    response = client.get(f"/datasets/{dataset_name}/tensors/{record_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["record_id"] == record_id
    assert data["shape"] == [2,2]
    assert data["data"] == tensor_data

    # Get non-existent tensor
    response = client.get(f"/datasets/{dataset_name}/tensors/nonexistent_id")
    assert response.status_code == 404

    # Get tensor from non-existent dataset
    response = client.get(f"/datasets/nonexistent_ds/tensors/{record_id}")
    assert response.status_code == 404


def test_delete_dataset_api():
    dataset_name = "test_delete_ds"
    client.post("/datasets/create", json={"name": dataset_name})
    TEST_DATASETS.add(dataset_name) # Ensure it's tracked

    # Delete existing dataset
    response = client.delete(f"/datasets/{dataset_name}")
    assert response.status_code == 200
    assert response.json()["message"] == f"Dataset '{dataset_name}' deleted successfully."
    TEST_DATASETS.discard(dataset_name) # No longer needs cleanup by fixture

    # Try deleting again
    response = client.delete(f"/datasets/{dataset_name}")
    assert response.status_code == 404

    # Try deleting non-existent dataset
    response = client.delete("/datasets/nonexistent_ds_never_created")
    assert response.status_code == 404


def test_delete_tensor_api():
    dataset_name = "test_delete_tensor_ds"
    record_id = _ingest_tensor_for_test(client, dataset_name, "t_del", [2], "int32", [10, 20])

    # Delete existing tensor
    response = client.delete(f"/datasets/{dataset_name}/tensors/{record_id}")
    assert response.status_code == 200
    assert response.json()["message"] == f"Tensor record '{record_id}' deleted successfully."

    # Try deleting again
    response = client.delete(f"/datasets/{dataset_name}/tensors/{record_id}")
    assert response.status_code == 404
    
    # Try deleting tensor from non-existent dataset
    response = client.delete(f"/datasets/nonexistent_ds/tensors/{record_id}")
    assert response.status_code == 404
    
    # Try deleting non-existent tensor from existing dataset
    response = client.delete(f"/datasets/{dataset_name}/tensors/non_id")
    assert response.status_code == 404


def test_update_tensor_metadata_api():
    dataset_name = "test_update_meta_ds"
    initial_metadata = {"source": "initial", "old_field": "keep_me"}
    record_id = _ingest_tensor_for_test(client, dataset_name, "t_meta", [1], "bool", [True], metadata=initial_metadata)

    new_metadata = {"source": "updated", "version": 2}
    response = client.put(f"/datasets/{dataset_name}/tensors/{record_id}/metadata", json={"new_metadata": new_metadata})
    assert response.status_code == 200
    assert response.json()["message"] == "Tensor metadata updated successfully."

    # Fetch and verify
    response = client.get(f"/datasets/{dataset_name}/tensors/{record_id}")
    assert response.status_code == 200
    # The new_metadata should completely replace the old one, plus any system-added keys like original record_id.
    # Check if new_metadata items are present.
    retrieved_metadata = response.json()["metadata"]
    for k, v in new_metadata.items():
        assert retrieved_metadata[k] == v
    assert "record_id" in retrieved_metadata  # System should preserve/add this
    assert "old_field" not in retrieved_metadata

    # Update non-existent tensor
    response = client.put(f"/datasets/{dataset_name}/tensors/non_id/metadata", json={"new_metadata": new_metadata})
    assert response.status_code == 404


# --- TensorOps Endpoints Tests ---
# Default output dataset for ops results, will be cleaned up
OPS_RESULT_DS = "tensor_ops_results"
TEST_DATASETS.add(OPS_RESULT_DS)


def test_ops_log():
    ds_in = "ops_log_in_ds"
    tensor_a_data = [[1.0, 10.0], [100.0, 1000.0]]
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "log_a", [2,2], "float32", tensor_a_data)

    request_payload = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "output_dataset_name": OPS_RESULT_DS 
    }
    response = client.post("/ops/log", json=request_payload)
    assert response.status_code == 200
    ops_data = response.json()
    assert ops_data["success"]
    assert ops_data["output_dataset_name"] == OPS_RESULT_DS
    out_record_id = ops_data["output_record_id"]
    
    # Fetch and verify result
    res_response = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{out_record_id}")
    assert res_response.status_code == 200
    result_tensor = res_response.json()
    
    expected_log_data = torch.log(torch.tensor(tensor_a_data)).tolist()
    assert result_tensor["data"] == expected_log_data
    assert result_tensor["metadata"]["operation"] == "log"


def test_ops_reshape():
    ds_in = "ops_reshape_in_ds"
    tensor_a_data = [1, 2, 3, 4, 5, 6]
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "reshape_a", [6], "int32", tensor_a_data)

    # Valid reshape
    request_payload = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"new_shape": [2, 3]},
        "output_dataset_name": OPS_RESULT_DS
    }
    response = client.post("/ops/reshape", json=request_payload)
    assert response.status_code == 200
    ops_data = response.json()
    out_record_id = ops_data["output_record_id"]
    
    res_response = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{out_record_id}")
    assert res_response.status_code == 200
    assert res_response.json()["shape"] == [2, 3]
    assert res_response.json()["data"] == [[1,2,3],[4,5,6]]

    # Invalid reshape (wrong number of elements)
    request_payload_invalid = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"new_shape": [2, 2]}, # 4 elements, input has 6
         "output_dataset_name": OPS_RESULT_DS
    }
    response_invalid = client.post("/ops/reshape", json=request_payload_invalid)
    assert response_invalid.status_code == 400 # TensorOps should raise error


def test_ops_sum():
    ds_in = "ops_sum_in_ds"
    tensor_a_data = [[1, 2, 3], [4, 5, 6]] # Sum = 21
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "sum_a", [2,3], "int32", tensor_a_data)

    # Sum all elements
    request_payload_all = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"dim": None, "keepdim": False}, # Explicitly None for dim
        "output_dataset_name": OPS_RESULT_DS
    }
    response_all = client.post("/ops/sum", json=request_payload_all)
    assert response_all.status_code == 200
    ops_data_all = response_all.json()
    res_all = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data_all['output_record_id']}").json()
    assert res_all["data"] == 21 # Sum of all elements
    assert res_all["shape"] == [] # Scalar result

    # Sum along dim 0
    request_payload_dim0 = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"dim": 0, "keepdim": False},
        "output_dataset_name": OPS_RESULT_DS
    }
    response_dim0 = client.post("/ops/sum", json=request_payload_dim0)
    assert response_dim0.status_code == 200
    ops_data_dim0 = response_dim0.json()
    res_dim0 = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data_dim0['output_record_id']}").json()
    assert res_dim0["data"] == [5, 7, 9] # [1+4, 2+5, 3+6]
    assert res_dim0["shape"] == [3]


def test_ops_add():
    ds_in = "ops_add_in_ds"
    tensor_a_data = [[1.0, 2.0], [3.0, 4.0]]
    tensor_b_data = [[0.5, 0.5], [0.5, 0.5]]
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "add_a", [2,2], "float32", tensor_a_data)
    tensor_b_id = _ingest_tensor_for_test(client, ds_in, "add_b", [2,2], "float32", tensor_b_data)
    scalar_val = 10.0

    # Tensor + Scalar
    req_scalar = {
        "input1": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "input2": {"scalar_value": scalar_val},
        "output_dataset_name": OPS_RESULT_DS
    }
    res_scalar = client.post("/ops/add", json=req_scalar)
    assert res_scalar.status_code == 200
    data_scalar = res_scalar.json()
    res_tensor_scalar = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{data_scalar['output_record_id']}").json()
    expected_scalar_add = (torch.tensor(tensor_a_data) + scalar_val).tolist()
    assert res_tensor_scalar["data"] == expected_scalar_add

    # Tensor + Tensor
    req_tensor = {
        "input1": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "input2": {"tensor_ref": {"dataset_name": ds_in, "record_id": tensor_b_id}},
        "output_dataset_name": OPS_RESULT_DS
    }
    res_tensor = client.post("/ops/add", json=req_tensor)
    assert res_tensor.status_code == 200
    data_tensor = res_tensor.json()
    res_tensor_tensor = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{data_tensor['output_record_id']}").json()
    expected_tensor_add = (torch.tensor(tensor_a_data) + torch.tensor(tensor_b_data)).tolist()
    assert res_tensor_tensor["data"] == expected_tensor_add


def test_ops_matmul():
    ds_in = "ops_matmul_in_ds"
    # A: 2x3
    tensor_a_data = [[1, 2, 3], [4, 5, 6]]
    # B: 3x2
    tensor_b_data = [[7, 8], [9, 10], [11, 12]]
    # C: 2x2 (incompatible with A for A@C)
    tensor_c_data = [[1,0],[0,1]]

    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "matmul_a", [2,3], "int32", tensor_a_data)
    tensor_b_id = _ingest_tensor_for_test(client, ds_in, "matmul_b", [3,2], "int32", tensor_b_data)
    tensor_c_id = _ingest_tensor_for_test(client, ds_in, "matmul_c", [2,2], "int32", tensor_c_data)

    # Valid Matmul A@B
    request_payload_valid = {
        "input1": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "input2": {"tensor_ref": {"dataset_name": ds_in, "record_id": tensor_b_id}},
        "output_dataset_name": OPS_RESULT_DS
    }
    response_valid = client.post("/ops/matmul", json=request_payload_valid)
    assert response_valid.status_code == 200
    ops_data_valid = response_valid.json()
    res_valid = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data_valid['output_record_id']}").json()
    
    expected_matmul_data = torch.matmul(torch.tensor(tensor_a_data), torch.tensor(tensor_b_data)).tolist()
    assert res_valid["data"] == expected_matmul_data # Should be [[58, 64], [139, 154]]
    assert res_valid["shape"] == [2,2]

    # Invalid Matmul A@C (shape mismatch)
    request_payload_invalid = {
        "input1": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "input2": {"tensor_ref": {"dataset_name": ds_in, "record_id": tensor_c_id}},
        "output_dataset_name": OPS_RESULT_DS
    }
    response_invalid = client.post("/ops/matmul", json=request_payload_invalid)
    assert response_invalid.status_code == 400 # PyTorch matmul raises RuntimeError for shape mismatch

    # Invalid - matmul with scalar
    request_payload_scalar = {
        "input1": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "input2": {"scalar_value": 5},
         "output_dataset_name": OPS_RESULT_DS
    }
    response_scalar = client.post("/ops/matmul", json=request_payload_scalar)
    assert response_scalar.status_code == 400 
    assert "Input2 for matmul must be a tensor" in response_scalar.json()["detail"]


def test_ops_concatenate():
    ds_in = "ops_concat_in_ds"
    tensor_a_data = [[1, 2]]
    tensor_b_data = [[3, 4]]
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "concat_a", [1,2], "int32", tensor_a_data)
    tensor_b_id = _ingest_tensor_for_test(client, ds_in, "concat_b", [1,2], "int32", tensor_b_data)

    # Concatenate along dim 0
    request_payload_dim0 = {
        "input_tensors": [
            {"dataset_name": ds_in, "record_id": tensor_a_id},
            {"dataset_name": ds_in, "record_id": tensor_b_id}
        ],
        "params": {"dim": 0},
        "output_dataset_name": OPS_RESULT_DS
    }
    response_dim0 = client.post("/ops/concatenate", json=request_payload_dim0)
    assert response_dim0.status_code == 200
    ops_data_dim0 = response_dim0.json()
    res_dim0 = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data_dim0['output_record_id']}").json()
    assert res_dim0["data"] == [[1,2],[3,4]]
    assert res_dim0["shape"] == [2,2]

    # Concatenate along dim 1
    request_payload_dim1 = {
        "input_tensors": [
            {"dataset_name": ds_in, "record_id": tensor_a_id},
            {"dataset_name": ds_in, "record_id": tensor_b_id}
        ],
        "params": {"dim": 1},
        "output_dataset_name": OPS_RESULT_DS
    }
    response_dim1 = client.post("/ops/concatenate", json=request_payload_dim1)
    assert response_dim1.status_code == 200
    ops_data_dim1 = response_dim1.json()
    res_dim1 = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data_dim1['output_record_id']}").json()
    assert res_dim1["data"] == [[1,2,3,4]]
    assert res_dim1["shape"] == [1,4]

    # Error: Mismatched shapes for concatenation (other than on concat dim)
    tensor_c_data = [[5,6,7]] # shape [1,3]
    tensor_c_id = _ingest_tensor_for_test(client, ds_in, "concat_c", [1,3], "int32", tensor_c_data)
    request_payload_invalid = {
        "input_tensors": [
            {"dataset_name": ds_in, "record_id": tensor_a_id}, # shape [1,2]
            {"dataset_name": ds_in, "record_id": tensor_c_id}  # shape [1,3]
        ],
        "params": {"dim": 0}, # Should fail because dim 1 sizes are different (2 vs 3)
        "output_dataset_name": OPS_RESULT_DS
    }
    response_invalid = client.post("/ops/concatenate", json=request_payload_invalid)
    assert response_invalid.status_code == 400


def test_ops_transpose():
    ds_in = "ops_transpose_in_ds"
    tensor_a_data = [[1,2,3],[4,5,6]] # 2x3
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "transpose_a", [2,3], "int32", tensor_a_data)

    request_payload = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"dim0": 0, "dim1": 1},
        "output_dataset_name": OPS_RESULT_DS
    }
    response = client.post("/ops/transpose", json=request_payload)
    assert response.status_code == 200
    ops_data = response.json()
    res = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data['output_record_id']}").json()
    
    expected_data = torch.tensor(tensor_a_data).transpose(0,1).tolist()
    assert res["data"] == expected_data
    assert res["shape"] == [3,2]

def test_ops_permute():
    ds_in = "ops_permute_in_ds"
    tensor_a_data = [[[1,2],[3,4]],[[5,6],[7,8]]] # 2x2x2
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "permute_a", [2,2,2], "int32", tensor_a_data)

    request_payload = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"dims": [2,0,1]}, # Permute to 2x2x2 -> 2x2x2 (but reordered)
        "output_dataset_name": OPS_RESULT_DS
    }
    response = client.post("/ops/permute", json=request_payload)
    assert response.status_code == 200
    ops_data = response.json()
    res = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data['output_record_id']}").json()
    
    expected_data = torch.tensor(tensor_a_data).permute(2,0,1).tolist()
    assert res["data"] == expected_data
    assert res["shape"] == [2,2,2]


def test_ops_mean():
    ds_in = "ops_mean_in_ds"
    tensor_a_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] 
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "mean_a", [2,3], "float32", tensor_a_data)

    # Mean of all elements
    request_payload_all = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"dim": None, "keepdim": False},
        "output_dataset_name": OPS_RESULT_DS
    }
    response_all = client.post("/ops/mean", json=request_payload_all)
    assert response_all.status_code == 200
    ops_data_all = response_all.json()
    res_all = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{ops_data_all['output_record_id']}").json()
    assert res_all["data"] == pytest.approx(3.5) # (1+2+3+4+5+6)/6 = 21/6 = 3.5
    assert res_all["shape"] == []


def test_ops_min_max():
    ds_in = "ops_minmax_in_ds"
    tensor_a_data = [[1, 5], [0, 9], [-2, 3]]
    tensor_a_id = _ingest_tensor_for_test(client, ds_in, "minmax_a", [3,2], "int32", tensor_a_data)

    # Min all elements
    req_min_all = {"input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id}, "output_dataset_name": OPS_RESULT_DS}
    res_min_all = client.post("/ops/min", json=req_min_all).json()
    val_min_all = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_min_all['output_record_id']}").json()
    assert val_min_all["data"] == -2

    # Max with dim and keepdim
    req_max_dim = {
        "input_tensor": {"dataset_name": ds_in, "record_id": tensor_a_id},
        "params": {"dim": 0, "keepdim": True},
        "output_dataset_name": OPS_RESULT_DS
    }
    res_max_dim = client.post("/ops/max", json=req_max_dim).json()
    val_max_dim = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_max_dim['output_record_id']}").json()
    assert val_max_dim["data"] == [[1, 9]] # Max along dim 0, kept dim
    assert val_max_dim["shape"] == [1,2]
    assert "(values tensor stored)" in res_max_dim["message"]


def test_ops_subtract_multiply_divide_power():
    ds_in = "ops_submuldivpow_in_ds"
    t_a_data = [[10, 20]]
    t_b_data = [[2, 5]]
    t_a_id = _ingest_tensor_for_test(client, ds_in, "sub_a", [1,2], "int32", t_a_data)
    t_b_id = _ingest_tensor_for_test(client, ds_in, "sub_b", [1,2], "int32", t_b_data)
    scalar = 2

    # Subtract tensor
    res_sub = client.post("/ops/subtract", json={
        "input1": {"dataset_name": ds_in, "record_id": t_a_id},
        "input2": {"tensor_ref": {"dataset_name": ds_in, "record_id": t_b_id}},
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    val_sub = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_sub['output_record_id']}").json()
    assert val_sub["data"] == [[8, 15]]

    # Multiply scalar
    res_mul = client.post("/ops/multiply", json={
        "input1": {"dataset_name": ds_in, "record_id": t_a_id},
        "input2": {"scalar_value": scalar},
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    val_mul = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_mul['output_record_id']}").json()
    assert val_mul["data"] == [[20, 40]]
    
    # Divide tensor
    res_div = client.post("/ops/divide", json={
        "input1": {"dataset_name": ds_in, "record_id": t_a_id},
        "input2": {"tensor_ref": {"dataset_name": ds_in, "record_id": t_b_id}}, # 10/2, 20/5
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    val_div = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_div['output_record_id']}").json()
    # Note: integer division might occur if inputs are int. For float, use float32.
    # Here, default torch behavior for int/int is floor division.
    assert val_div["data"] == [[5.0, 4.0]] # Assuming TensorOps promotes to float for divide

    # Power scalar
    res_pow = client.post("/ops/power", json={
        "base_tensor": {"dataset_name": ds_in, "record_id": t_b_id}, # t_b_data = [[2,5]]
        "exponent": {"scalar_value": scalar}, # scalar = 2
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    val_pow = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_pow['output_record_id']}").json()
    assert val_pow["data"] == [[4, 25]] # 2^2, 5^2


def test_ops_dot():
    ds_in = "ops_dot_in_ds"
    t_a_data = [1, 2, 3]
    t_b_data = [4, 5, 6]
    t_a_id = _ingest_tensor_for_test(client, ds_in, "dot_a", [3], "int32", t_a_data)
    t_b_id = _ingest_tensor_for_test(client, ds_in, "dot_b", [3], "int32", t_b_data)

    res_dot = client.post("/ops/dot", json={
        "input1": {"dataset_name": ds_in, "record_id": t_a_id},
        "input2": {"tensor_ref": {"dataset_name": ds_in, "record_id": t_b_id}},
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    assert res_dot["success"]
    val_dot = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_dot['output_record_id']}").json()
    assert val_dot["data"] == (1*4 + 2*5 + 3*6) # 4 + 10 + 18 = 32
    assert val_dot["shape"] == []


def test_ops_stack():
    ds_in = "ops_stack_in_ds"
    t_a_data = [1,2] # shape [2]
    t_b_data = [3,4] # shape [2]
    t_a_id = _ingest_tensor_for_test(client, ds_in, "stack_a", [2], "int32", t_a_data)
    t_b_id = _ingest_tensor_for_test(client, ds_in, "stack_b", [2], "int32", t_b_data)

    # Stack along new dim 0
    res_stack = client.post("/ops/stack", json={
        "input_tensors": [
            {"dataset_name": ds_in, "record_id": t_a_id},
            {"dataset_name": ds_in, "record_id": t_b_id}
        ],
        "params": {"dim": 0},
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    assert res_stack["success"]
    val_stack = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_stack['output_record_id']}").json()
    assert val_stack["data"] == [[1,2],[3,4]]
    assert val_stack["shape"] == [2,2]


def test_ops_einsum():
    ds_in = "ops_einsum_in_ds"
    # Matrix multiplication: A (2x3) @ B (3x2) -> C (2x2)
    # Equation: "ij,jk->ik"
    t_a_data = [[1,2,3],[4,5,6]]
    t_b_data = [[1,0],[0,1],[1,1]]
    t_a_id = _ingest_tensor_for_test(client, ds_in, "einsum_a", [2,3], "int32", t_a_data)
    t_b_id = _ingest_tensor_for_test(client, ds_in, "einsum_b", [3,2], "int32", t_b_data)

    res_einsum = client.post("/ops/einsum", json={
        "input_tensors": [
            {"dataset_name": ds_in, "record_id": t_a_id},
            {"dataset_name": ds_in, "record_id": t_b_id}
        ],
        "params": {"equation": "ij,jk->ik"},
        "output_dataset_name": OPS_RESULT_DS
    }).json()
    assert res_einsum["success"]
    val_einsum = client.get(f"/datasets/{OPS_RESULT_DS}/tensors/{res_einsum['output_record_id']}").json()
    
    expected_data = torch.einsum("ij,jk->ik", torch.tensor(t_a_data), torch.tensor(t_b_data)).tolist()
    assert val_einsum["data"] == expected_data
    assert val_einsum["shape"] == [2,2]

    # Test invalid equation (e.g., too few inputs)
    res_einsum_invalid = client.post("/ops/einsum", json={
        "input_tensors": [{"dataset_name": ds_in, "record_id": t_a_id}], # Only one tensor
        "params": {"equation": "ij,jk->ik"}, # Equation expects two
        "output_dataset_name": OPS_RESULT_DS
    })
    assert res_einsum_invalid.status_code == 400
