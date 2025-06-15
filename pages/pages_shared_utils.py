# pages/pages_shared_utils.py
"""
Shared utility functions for Streamlit pages.
Copied/adapted from app.py to avoid complex import issues.
"""
import streamlit as st
import requests
import logging
import os # Added import
from typing import Optional, List, Dict, Any # Added for new functions

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000") # Changed API_BASE_URL

def load_css():
    """Loads the main CSS styles. Assumes app.py's CSS content."""
    st.markdown("""
<style>
    /* --- Shared Base Styles for Tensorus Platform (Nexus Theme) --- */

    /* General Page Styles */
    body {
        font-family: 'Arial', 'Helvetica Neue', 'Helvetica', sans-serif;
        line-height: 1.6;
    }
    .stApp { /* Main Streamlit app container */
        background-color: #0a0f2c; /* Primary Background: Dark blue/purple */
        color: #e0e0e0; /* Default Text Color: Light grey */
    }

    /* Headings & Titles */
    h1, .stTitle { /* Main page titles */
        color: #d0d0ff !important; /* Primary Heading Color: Light purple/blue */
        font-weight: bold !important;
    }
    h2, .stSubheader { /* Section headers */
        color: #c0c0ef !important; /* Secondary Heading Color */
        font-weight: bold !important;
        border-bottom: 1px solid #3a3f5c; /* Accent Border for separation */
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 { /* General h3, often used in st.markdown */
        color: #b0b0df !important; /* Tertiary Heading Color */
        font-weight: bold !important;
    }
    .stMarkdown p, .stText, .stListItem { /* General text elements */
        color: #c0c0dd; /* Softer light text */
        font-size: 1rem;
    }
    .stCaption, caption { /* Streamlit captions and HTML captions */
        font-size: 0.85rem !important;
        color: #a0a0c0 !important; /* Muted color for captions */
    }

    /* Custom Top Navigation Bar */
    .topnav-container {
        background-color: #1a1f3c; /* Nav Background: Slightly darker than page */
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #3a3f5c; /* Accent Border */
        display: flex;
        justify-content: flex-start;
        align-items: center;
        position: sticky; top: 0; z-index: 1000; /* Ensure it's on top */
        width: 100%;
        box-sizing: border-box;
    }
    .topnav-container .logo {
        font-size: 1.5em;
        font-weight: bold;
        color: #d0d0ff; /* Primary Heading Color for logo */
        margin-right: 2rem;
    }
    .topnav-container nav a {
        color: #c0c0ff; /* Lighter Text for Nav Links */
        padding: 0.75rem 1rem;
        text-decoration: none;
        font-weight: 500;
        margin-right: 0.5rem;
        border-radius: 4px;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    .topnav-container nav a:hover {
        background-color: #2a2f4c; /* Nav Link Hover Background */
        color: #ffffff; /* Nav Link Hover Text */
    }
    .topnav-container nav a.active {
        background-color: #3a6fbf; /* Active Nav Link Background (Accent Blue) */
        color: #ffffff; /* Active Nav Link Text */
        font-weight: bold;
    }

    /* Common Card Style (base for metric cards, agent cards, etc.) */
    .common-card {
        background-color: #18223f; /* Card Background: Darker than nav, but lighter than page */
        border: 1px solid #2a3f5c; /* Card Border: Accent Border color */
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem; /* Space below cards */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        color: #e0e0e0; /* Default text color within cards */
    }
    .common-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    .common-card h3 { /* Titles within cards */
        color: #b0b0df !important; /* Card Title Color: Slightly lighter than main headings */
        font-size: 1.2em !important; /* Slightly larger for card titles */
        margin-top: 0 !important; /* Remove default top margin for h3 in card */
        margin-bottom: 0.75rem !important;
        font-weight: bold !important;
        border-bottom: none !important; /* Override general h2 border for card h3 */
    }
    .common-card p { /* Paragraphs within cards */
        font-size: 0.95em !important; /* Slightly smaller for card content */
        color: #c0c0dd !important;
        margin-bottom: 0.5rem !important;
    }
    .common-card .icon { /* For icons within cards, like dashboard metric cards */
        font-size: 2.5em;
        margin-bottom: 0.75rem;
        color: #7070ff; /* Icon Color: Muted accent */
    }


    /* Status Indicators (can be used with <span> or <p> or custom divs) */
    .status-indicator {
        padding: 0.4rem 0.8rem !important; /* Slightly more padding */
        border-radius: 15px !important; /* Pill shape */
        font-weight: bold !important;
        font-size: 0.85em !important;
        display: inline-block !important;
        text-align: center !important;
    }
    .status-success, .status-running { color: #ffffff !important; background-color: #4CAF50 !important; } /* Green */
    .status-error   { color: #ffffff !important; background-color: #F44336 !important; } /* Red */
    .status-warning { color: #000000 !important; background-color: #FFC107 !important; } /* Amber */
    .status-info    { color: #ffffff !important; background-color: #2196F3 !important; } /* Blue */
    .status-stopped { color: #e0e0e0 !important; background-color: #525252 !important; } /* Darker Grey for stopped */
    .status-unknown { color: #333333 !important; background-color: #BDBDBD !important; } /* Lighter grey for unknown */


    /* Standardized Streamlit Input Styling */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea, 
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border: 1px solid #3a3f5c !important; /* Accent Border */
        background-color: #1a1f3c !important; /* Nav Background color for inputs */
        color: #e0e0e0 !important; /* Default Text Color */
        border-radius: 5px !important;
    }
    .stMultiSelect > div > div > div { /* Multiselect options container */
         border: 1px solid #3a3f5c !important;
         background-color: #1a1f3c !important;
    }
    .stMultiSelect span[data-baseweb="tag"] { /* Selected items in multiselect */
        background-color: #3a6fbf !important; /* Active Nav Link Background */
    }


    /* Standardized Streamlit Button Styling */
    .stButton > button {
        border: 1px solid #3a6fbf !important; /* Accent Blue for border */
        background-color: #3a6fbf !important; /* Accent Blue for background */
        color: white !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #4a7fdc !important; /* Lighter Accent Blue for hover */
        border-color: #4a7fdc !important;
    }
    .stButton > button:disabled {
        background-color: #2a2f4c !important;
        border-color: #2a2f4c !important;
        color: #777777 !important;
    }
    /* For secondary buttons, Streamlit uses a 'kind' attribute in HTML we can't directly target via pure CSS.
       Instead, use st.button(..., type="secondary") and rely on Streamlit's handling,
       or use st.markdown for fully custom buttons if default secondary is not enough.
       The below attempts to style based on common Streamlit secondary button appearance.
       Note: This specific selector for secondary buttons might be unstable if Streamlit changes its internal class names.
    */
    .stButton button.st-emotion-cache-LPTKCI { /* Example selector for a secondary button, MAY BE UNSTABLE */
        background-color: #2a2f4c !important; 
        border: 1px solid #2a2f4c !important;
        color: #c0c0ff !important;
    }
     .stButton button.st-emotion-cache-LPTKCI:hover {
        background-color: #3a3f5c !important;
        border-color: #3a3f5c !important;
        color: #ffffff !important;
    }
    .stButton button.st-emotion-cache-LPTKCI:disabled { /* Disabled secondary button */
        background-color: #1e2a47 !important;
        border-color: #1e2a47 !important;
        color: #555555 !important;
    }


    /* Dataframe styling */
    .stDataFrame { /* Main container for dataframes */
        border: 1px solid #2a3f5c !important; /* Accent Border */
        border-radius: 5px !important;
        background-color: #1a1f3c !important; /* Nav Background for dataframe background */
    }
    .stDataFrame th { /* Headers */
        background-color: #2a2f4c !important; /* Nav Link Hover Background for headers */
        color: #d0d0ff !important; /* Primary Heading Color for header text */
        font-weight: bold;
    }
    .stDataFrame td { /* Cells */
        color: #c0c0dd !important; /* Softer light text for cell data */
        border-bottom-color: #2a3f5c !important; /* Accent border for cell lines */
        border-top-color: #2a3f5c !important;
    }
    
</style>
""", unsafe_allow_html=True)

def get_api_status() -> tuple[bool, dict]:
    """
    Checks if the backend API is reachable and returns its status.

    Uses the `API_BASE_URL` constant defined in this module.

    Returns:
        tuple[bool, dict]: A tuple where:
            - The first element is a boolean: True if the API is reachable and returns a 2xx status, False otherwise.
            - The second element is a dictionary: 
                - If successful, contains API information (e.g., from `response.json()`).
                - If unsuccessful, contains an 'error' key with a descriptive message.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=3) # Increased timeout slightly
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error in get_api_status (shared_utils): {e}")
        return False, {"error": f"API connection failed: {str(e)}"}
    except Exception as e:
        logger.exception(f"Unexpected error in get_api_status (shared_utils): {e}")
        return False, {"error": f"An unexpected error occurred: {str(e)}"}

def get_agent_status() -> Optional[dict]:
    """
    Fetches the status for all registered agents from the backend.

    Uses the `API_BASE_URL` constant defined in this module.
    On successful API call, returns a dictionary where keys are agent IDs
    and values are dictionaries containing status and configuration for each agent.
    Returns None if the API call fails or an exception occurs.

    Returns:
        Optional[dict]: Agent statuses dictionary or None.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/agents/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching agent status (shared_utils): {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_agent_status (shared_utils): {e}")
        return None

def start_agent(agent_id: str) -> dict:
    """
    Sends a request to the backend to start a specific agent.

    Uses the `API_BASE_URL` constant defined in this module.
    Constructs a POST request to the `/agents/{agent_id}/start` endpoint.

    Args:
        agent_id (str): The unique identifier of the agent to start.

    Returns:
        dict: A dictionary containing the API response. Typically includes a 'success' boolean
              and a 'message' string. In case of connection or unexpected errors,
              it also returns a dict with 'success': False and an error 'message'.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/agents/{agent_id}/start", timeout=7) # Increased timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error starting agent {agent_id} (shared_utils): {e}")
        return {"success": False, "message": f"Failed to start agent {agent_id}: {str(e)}"}
    except Exception as e:
        logger.exception(f"Unexpected error in start_agent (shared_utils) for {agent_id}: {e}")
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

def stop_agent(agent_id: str) -> dict:
    """
    Sends a request to the backend to stop a specific agent.

    Uses the `API_BASE_URL` constant defined in this module.
    Constructs a POST request to the `/agents/{agent_id}/stop` endpoint.

    Args:
        agent_id (str): The unique identifier of the agent to stop.

    Returns:
        dict: A dictionary containing the API response. Typically includes a 'success' boolean
              and a 'message' string. In case of connection or unexpected errors,
              it also returns a dict with 'success': False and an error 'message'.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/agents/{agent_id}/stop", timeout=7) # Increased timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error stopping agent {agent_id} (shared_utils): {e}")
        return {"success": False, "message": f"Failed to stop agent {agent_id}: {str(e)}"}
    except Exception as e:
        logger.exception(f"Unexpected error in stop_agent (shared_utils) for {agent_id}: {e}")
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

def get_datasets() -> list[str]:
    """
    Fetches the list of available dataset names from the backend.

    Uses the `API_BASE_URL` constant defined in this module.
    Targets the `/explorer/datasets` endpoint.

    Returns:
        list[str]: A list of dataset names. Returns an empty list if the API call
                   fails, if the 'datasets' key is missing in the response,
                   or if an exception occurs.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/explorer/datasets", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("datasets", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching datasets (shared_utils): {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error in get_datasets (shared_utils): {e}")
        return []

def get_dataset_preview(dataset_name: str, limit: int = 10) -> Optional[dict]:
    """
    Fetches preview data for a specific dataset from the backend.

    Uses the `API_BASE_URL` constant defined in this module.
    Targets the `/explorer/dataset/{dataset_name}/preview` endpoint with a `limit` parameter.

    Args:
        dataset_name (str): The name of the dataset to preview.
        limit (int): The maximum number of records to fetch for the preview. Defaults to 10.

    Returns:
        Optional[dict]: A dictionary containing dataset information (e.g., 'dataset', 
                        'record_count', 'preview' list of records) if successful. 
                        Each record in the 'preview' list is a dictionary typically 
                        containing 'id', 'shape', 'dtype', 'metadata', and 'data' (raw list).
                        Returns None if the API call fails or an exception occurs.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/explorer/dataset/{dataset_name}/preview?limit={limit}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching dataset preview for {dataset_name} (shared_utils): {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_dataset_preview (shared_utils) for {dataset_name}: {e}")
        return None




def get_tensor_metadata(dataset_name: str, tensor_id: str) -> Optional[dict]:
    """Fetch metadata for a specific tensor via the Explorer API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/explorer/dataset/{dataset_name}/tensor/{tensor_id}/metadata",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("metadata", data)
    except requests.exceptions.RequestException as e:
        logger.error(
            f"API error fetching tensor metadata for {dataset_name}/{tensor_id} (shared_utils): {e}"
        )
        return None
    except Exception as e:
        logger.exception(
            f"Unexpected error in get_tensor_metadata (shared_utils) for {dataset_name}/{tensor_id}: {e}"

        )

        return None

def list_all_agents() -> list[dict[str, str]]:
    """
    Returns a list of agent details based on the current statuses fetched by `get_agent_status`.

    Each agent detail is a dictionary with 'id', 'name', and 'status' keys.
    If an agent's name is not explicitly provided in the status data, a default name
    is generated by capitalizing the agent_id and replacing underscores with spaces.

    This function is a convenience wrapper around `get_agent_status()` if a list
    format of agent information is preferred.

    Returns:
        list[dict[str, str]]: A list of dictionaries, where each dictionary represents an agent
                              and contains its 'id', 'name', and 'status'.
                              Returns an empty list if agent statuses cannot be fetched.
    """
    agents_status_data = get_agent_status()
    if agents_status_data:
        return [
            {
                "id": agent_id,
                "name": agent_data.get("name", agent_id.replace("_", " ").title()), # Default formatted name
                "status": agent_data.get("status", "unknown")
            }
            for agent_id, agent_data in agents_status_data.items()
        ]
    return []

def post_nql_query(query: str) -> dict:
    """
    Sends an NQL query to the backend for processing.

    Uses the `API_BASE_URL` constant defined in this module.
    Constructs a POST request to the `/chat/query` endpoint with the user's query.

    Args:
        query (str): The Natural Query Language query string provided by the user.

    Returns:
        dict: A dictionary containing the API response. 
              On success, this typically includes:
                - 'query': The original query string.
                - 'response_text': A textual summary of the NQL agent's action or findings.
                - 'results': A list of records (tensors with metadata) if the query involved data retrieval.
                             Each record is a dictionary, potentially including 'id', 'shape', 
                             'dtype', 'metadata', and 'data'.
                - 'count': Number of results found (if applicable).
              On failure (e.g., connection error, API error, unexpected server error):
                - 'query': The original query.
                - 'response_text': An error message.
                - 'error': A more detailed error string.
                - 'results': None or an empty list.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/query", # Ensure API_BASE_URL is defined in this file
            json={"query": query},
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error posting NQL query from pages_shared_utils: {e}")
        # Let the caller handle UI error display
        return {"query": query, "response_text": "Error connecting to backend or processing query.", "error": str(e), "results": None}
    except Exception as e:
        logger.exception(f"Unexpected error in post_nql_query (pages_shared_utils): {e}")
        return {"query": query, "response_text": "An unexpected error occurred.", "error": str(e), "results": None}

# --- Functions to be added from app.py ---

def configure_agent(agent_id: str, config: dict) -> dict:
    """
    Sends a request to the backend to configure a specific agent.

    Uses the `API_BASE_URL` constant defined in this module.
    Constructs a POST request to the `/agents/{agent_id}/configure` endpoint.

    Args:
        agent_id (str): The unique identifier of the agent to configure.
        config (dict): The configuration dictionary for the agent.

    Returns:
        dict: A dictionary containing the API response. Typically includes 'success' boolean
              and a 'message' string. In case of connection or unexpected errors,
              it also returns a dict with 'success': False and an error 'message'.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/agents/{agent_id}/configure",
            json={"config": config},
            timeout=7 # Increased timeout similar to start/stop
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error configuring agent {agent_id} (shared_utils): {e}")
        return {"success": False, "message": f"Failed to configure agent {agent_id}: {str(e)}"}
    except Exception as e:
        logger.exception(f"Unexpected error in configure_agent (shared_utils) for {agent_id}: {e}")
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

def operate_explorer(dataset: str, operation: str, index: int, params: dict) -> dict:
    """
    Sends an operation request to the data explorer for a specific tensor.

    Uses the `API_BASE_URL` constant defined in this module.
    Constructs a POST request to the `/explorer/operate` endpoint.

    Args:
        dataset (str): The name of the dataset containing the tensor.
        operation (str): The operation to perform (e.g., 'view', 'transform').
        index (int): The index of the tensor within the dataset.
        params (dict): Additional parameters required for the operation.

    Returns:
        dict: A dictionary containing the API response. Typically includes:
              - 'success': A boolean indicating if the operation was accepted.
              - 'metadata': A dictionary with details about the operation or resulting tensor.
              - 'result_data': The data of the resulting tensor (if applicable), or None.
              In case of connection or server-side errors, it returns a dict with
              'success': False, 'metadata': {'error': error_message}, and 'result_data': None.
    """
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
            timeout=15 # Standard timeout for potentially long operations
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error in operate_explorer for {dataset} (shared_utils): {e}")
        return {"success": False, "metadata": {"error": str(e)}, "result_data": None}
    except Exception as e:
        logger.exception(f"Unexpected error in operate_explorer (shared_utils) for {dataset}: {e}")
        return {"success": False, "metadata": {"error": str(e)}, "result_data": None}
