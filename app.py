# app.py
"""
Streamlit frontend application for the Tensorus platform.
Interacts with the FastAPI backend (api.py).
"""

import streamlit as st
import json
import time
import requests # Needed for ui_utils functions if integrated
import logging # Needed for ui_utils functions if integrated
import torch # Needed for integrated tensor utils
from typing import List, Dict, Any, Optional, Union, Tuple # Needed for integrated tensor utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Tensorus Platform",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configure Logging (Optional but good practice) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Integrated Tensor Utilities ---

def _validate_tensor_data(data: List[Any], shape: List[int]):
    """
    Validates if the nested list structure of 'data' matches the 'shape'.
    Raises ValueError on mismatch. (Optional validation)
    """
    if not shape:
        if not isinstance(data, (int, float)): raise ValueError("Scalar tensor data must be a single number.")
        return True
    if not isinstance(data, list): raise ValueError(f"Data for shape {shape} must be a list.")
    expected_len = shape[0]
    if len(data) != expected_len: raise ValueError(f"Dimension 0 mismatch: Expected {expected_len}, got {len(data)} for shape {shape}.")
    if len(shape) > 1:
        for item in data: _validate_tensor_data(item, shape[1:])
    elif len(shape) == 1:
        if not all(isinstance(x, (int, float)) for x in data): raise ValueError("Innermost list elements must be numbers.")
    return True

def list_to_tensor(shape: List[int], dtype_str: str, data: Union[List[Any], int, float]) -> torch.Tensor:
    """
    Converts a Python list (potentially nested) or scalar into a PyTorch tensor
    with the specified shape and dtype.
    """
    try:
        dtype_map = {
            'float32': torch.float32, 'float': torch.float,
            'float64': torch.float64, 'double': torch.double,
            'int32': torch.int32, 'int': torch.int,
            'int64': torch.int64, 'long': torch.long,
            'bool': torch.bool
        }
        torch_dtype = dtype_map.get(dtype_str.lower())
        if torch_dtype is None: raise ValueError(f"Unsupported dtype string: {dtype_str}")

        tensor = torch.tensor(data, dtype=torch_dtype)

        if list(tensor.shape) != shape:
            logger.debug(f"Initial tensor shape {list(tensor.shape)} differs from target {shape}. Attempting reshape.")
            try:
                tensor = tensor.reshape(shape)
            except RuntimeError as reshape_err:
                raise ValueError(f"Created tensor shape {list(tensor.shape)} != requested {shape} and reshape failed: {reshape_err}") from reshape_err

        return tensor
    except (TypeError, ValueError) as e:
        logger.error(f"Error converting list to tensor: {e}. Shape: {shape}, Dtype: {dtype_str}")
        raise ValueError(f"Failed tensor conversion: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error during list_to_tensor: {e}")
        raise ValueError(f"Unexpected tensor conversion error: {e}") from e

def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[int], str, List[Any]]:
    """
    Converts a PyTorch tensor back into its shape, dtype string, and nested list representation.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    shape = list(tensor.shape)
    dtype_str = str(tensor.dtype).split('.')[-1]
    data = tensor.tolist()
    return shape, dtype_str, data

# --- Integrated UI Utilities (from former ui_utils.py) ---

# Define the base URL of your FastAPI backend
API_BASE_URL = "http://127.0.0.1:8000" # Make sure this matches where api.py runs

def get_api_status():
    """Checks if the backend API is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return True, response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error: {e}")
        return False, {"error": str(e)}

def get_agent_status():
    """Fetches status for all agents from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/agents/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error fetching agent status: {e}")
        return None

def start_agent(agent_id: str):
    """Sends a request to start an agent."""
    try:
        response = requests.post(f"{API_BASE_URL}/agents/{agent_id}/start", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error starting agent {agent_id}: {e}")
        return {"success": False, "message": str(e)}

def stop_agent(agent_id: str):
    """Sends a request to stop an agent."""
    try:
        response = requests.post(f"{API_BASE_URL}/agents/{agent_id}/stop", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error stopping agent {agent_id}: {e}")
        return {"success": False, "message": str(e)}

def configure_agent(agent_id: str, config: dict):
    """Sends a request to configure an agent."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/agents/{agent_id}/configure",
            json={"config": config},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error configuring agent {agent_id}: {e}")
        return {"success": False, "message": str(e)}

def post_nql_query(query: str):
    """Sends an NQL query to the backend."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/query",
            json={"query": query},
            timeout=15 # Allow more time for potentially complex queries
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error posting NQL query: {e}")
        return {"query": query, "response_text": "Error connecting to backend.", "error": str(e)}

def get_datasets():
    """Fetches the list of available datasets."""
    try:
        response = requests.get(f"{API_BASE_URL}/explorer/datasets", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("datasets", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error fetching datasets: {e}")
        return [] # Return empty list on error

def get_dataset_preview(dataset_name: str, limit: int = 5):
    """Fetches preview data for a specific dataset."""
    try:
        response = requests.get(f"{API_BASE_URL}/explorer/dataset/{dataset_name}/preview?limit={limit}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error fetching preview for {dataset_name}: {e}")
        return None

def operate_explorer(dataset: str, operation: str, index: int, params: dict):
    """Sends an operation request to the data explorer."""
    payload = {
        "dataset": dataset,
        "operation": operation,
        "tensor_index": index,
        "params": params
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/explorer/operate",
            json=payload,
            timeout=15 # Allow time for computation
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error performing operation '{operation}' on {dataset}: {e}")
        return {"success": False, "metadata": {"error": str(e)}, "result_data": None}


# --- Initialize Session State ---
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = []
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'dataset_preview' not in st.session_state:
    st.session_state.dataset_preview = None
if 'explorer_result' not in st.session_state:
    st.session_state.explorer_result = None
if 'nql_response' not in st.session_state:
    st.session_state.nql_response = None


# --- Sidebar ---
with st.sidebar:
    st.title("Tensorus Control")
    st.markdown("---")

    # API Status Check
    st.subheader("API Status")
    api_ok, api_info = get_api_status() # Use integrated function
    if api_ok:
        st.success(f"Connected to API v{api_info.get('version', 'N/A')}")
    else:
        st.error(f"API Connection Failed: {api_info.get('error', 'Unknown error')}")
        st.warning("Ensure the backend (`uvicorn api:app ...`) is running.")
        st.stop()

    st.markdown("---")
    # Navigation
    app_mode = st.radio(
        "Select Feature",
        ("Dashboard", "Agent Control", "NQL Chat", "Data Explorer")
    )
    st.markdown("---")


# --- Main Page Content ---

if app_mode == "Dashboard":
    st.title("📊 Operations Dashboard")
    st.warning("Live WebSocket dashboard view is best accessed directly via the backend's `/dashboard` HTML page or a dedicated JS frontend. This is a simplified view.")
    st.markdown(f"Access the basic live dashboard here.", unsafe_allow_html=True) # Link to backend dashboard
    st.info("This Streamlit view doesn't currently support live WebSocket updates.")


elif app_mode == "Agent Control":
    st.title("🤖 Agent Control Panel")

    if st.button("Refresh Agent Status"):
        st.session_state.agent_status = get_agent_status() # Use integrated function

    if st.session_state.agent_status:
        agents = st.session_state.agent_status
        agent_ids = list(agents.keys())

        if not agent_ids:
            st.warning("No agents reported by the backend.")
        else:
            selected_agent_id = st.selectbox("Select Agent", agent_ids)

            if selected_agent_id:
                agent_info = agents[selected_agent_id]
                st.subheader(f"Agent: {agent_info.get('name', selected_agent_id)}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Status", "Running" if agent_info.get('running') else "Stopped")
                    st.write("**Configuration:**")
                    st.json(agent_info.get('config', {}))
                with col2:
                    st.write("**Recent Logs:**")
                    st.code('\n'.join(agent_info.get('logs', [])), language='log')

                st.write("**Actions:**")
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("Start Agent", key=f"start_{selected_agent_id}", disabled=agent_info.get('running')):
                        result = start_agent(selected_agent_id) # Use integrated function
                        st.toast(result.get("message", "Request sent."))
                        st.session_state.agent_status = get_agent_status() # Refresh status
                        st.rerun()
                with btn_col2:
                    if st.button("Stop Agent", key=f"stop_{selected_agent_id}", disabled=not agent_info.get('running')):
                        result = stop_agent(selected_agent_id) # Use integrated function
                        st.toast(result.get("message", "Request sent."))
                        st.session_state.agent_status = get_agent_status() # Refresh status
                        st.rerun()

    else:
        st.info("Click 'Refresh Agent Status' to load agent information.")


elif app_mode == "NQL Chat":
    st.title("💬 Natural Language Query (NQL)")
    st.info("Ask questions about the data stored in Tensorus (e.g., 'show me tensors from rl_experiences', 'count records in sample_data').")

    user_query = st.text_input("Enter your query:", key="nql_query_input")

    if st.button("Submit Query", key="nql_submit"):
        if user_query:
            with st.spinner("Processing query..."):
                st.session_state.nql_response = post_nql_query(user_query) # Use integrated function
        else:
            st.warning("Please enter a query.")

    if st.session_state.nql_response:
        resp = st.session_state.nql_response
        st.markdown("---")
        st.write(f"**Query:** {resp.get('query')}")
        if resp.get("error"):
            st.error(f"Error: {resp.get('error')}")
        else:
            st.success(f"**Response:** {resp.get('response_text')}")
            if resp.get("results"):
                st.write("**Results Preview:**")
                st.json(resp.get("results"))
        st.session_state.nql_response = None


elif app_mode == "Data Explorer":
    st.title("🔍 Data Explorer")

    if not st.session_state.datasets or st.button("Refresh Datasets"):
        st.session_state.datasets = get_datasets() # Use integrated function

    if not st.session_state.datasets:
        st.warning("No datasets found or failed to fetch from backend.")
    else:
        st.session_state.selected_dataset = st.selectbox(
            "Select Dataset",
            st.session_state.datasets,
            index=st.session_state.datasets.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in st.session_state.datasets else 0
        )

        if st.session_state.selected_dataset:
            if st.button("Show Preview", key="preview_btn"):
                with st.spinner(f"Fetching preview for {st.session_state.selected_dataset}..."):
                    st.session_state.dataset_preview = get_dataset_preview(st.session_state.selected_dataset) # Use integrated function

            if st.session_state.dataset_preview:
                st.subheader(f"Preview: {st.session_state.dataset_preview.get('dataset')}")
                st.write(f"Total Records: {st.session_state.dataset_preview.get('record_count')}")
                st.dataframe(st.session_state.dataset_preview.get('preview', []))
                st.markdown("---")

            st.subheader("Perform Operation")
            record_count = st.session_state.dataset_preview.get('record_count', 1) if st.session_state.dataset_preview else 1
            tensor_index = st.number_input("Select Tensor Index", min_value=0, max_value=max(0, record_count - 1), value=0, step=1)

            operations = ["info", "head", "slice", "sum", "mean", "reshape", "transpose"]
            selected_op = st.selectbox("Select Operation", operations)

            params = {}
            # Dynamic parameter inputs
            if selected_op == "head":
                params['count'] = st.number_input("Count", min_value=1, value=5, step=1)
            elif selected_op == "slice":
                params['dim'] = st.number_input("Dimension (dim)", value=0, step=1)
                params['start'] = st.number_input("Start Index", value=0, step=1)
                params['end'] = st.number_input("End Index (optional)", value=None, step=1, format="%d")
                params['step'] = st.number_input("Step (optional)", value=None, step=1, format="%d")
            elif selected_op in ["sum", "mean"]:
                dim_input = st.text_input("Dimension(s) (optional, e.g., 0 or 0,1)")
                if dim_input:
                    try: params['dim'] = [int(x.strip()) for x in dim_input.split(',')] if ',' in dim_input else int(dim_input)
                    except ValueError: st.warning("Invalid dimension format.")
                params['keepdim'] = st.checkbox("Keep Dimensions (keepdim)", value=False)
            elif selected_op == "reshape":
                shape_input = st.text_input("Target Shape (comma-separated, e.g., 2,3,5)")
                if shape_input:
                    try: params['shape'] = [int(x.strip()) for x in shape_input.split(',')]
                    except ValueError: st.warning("Invalid shape format.")
            elif selected_op == "transpose":
                params['dim0'] = st.number_input("Dimension 0", value=0, step=1)
                params['dim1'] = st.number_input("Dimension 1", value=1, step=1)

            if st.button("Run Operation", key="run_op_btn"):
                valid_request = True
                if selected_op == "reshape" and not params.get('shape'):
                    st.error("Target Shape is required for reshape.")
                    valid_request = False

                if valid_request:
                    with st.spinner(f"Running {selected_op} on {st.session_state.selected_dataset}[{tensor_index}]..."):
                        st.session_state.explorer_result = operate_explorer( # Use integrated function
                            st.session_state.selected_dataset,
                            selected_op,
                            tensor_index,
                            params
                        )

            if st.session_state.explorer_result:
                st.markdown("---")
                st.subheader("Operation Result")
                st.write("**Metadata:**")
                st.json(st.session_state.explorer_result.get("metadata", {}))
                st.write("**Result Data:**")
                st.json(st.session_state.explorer_result.get("result_data", "No data returned."))

