# ui_utils.py (Modifications for Step 3)
"""Utility functions for the Tensorus Streamlit UI, now using API calls."""

import requests
import streamlit as st
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Configuration ---
TENSORUS_API_URL = "http://127.0.0.1:7860" # Ensure FastAPI runs here

# --- API Interaction Functions ---

def get_api_status() -> bool:
    """Checks if the Tensorus API is reachable."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        logger.error(f"Error checking API status: {e}")
        return False

def list_datasets() -> Optional[List[str]]:
    """Fetches the list of dataset names from the API."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/datasets")
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("data", [])
        else:
            st.error(f"API Error listing datasets: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error listing datasets: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error listing datasets: {e}")
        logger.exception("Unexpected error in list_datasets")
        return None

def fetch_dataset_data(dataset_name: str, offset: int = 0, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    """Fetches a page of records from a dataset via API."""
    try:
        params = {"offset": offset, "limit": limit}
        response = requests.get(f"{TENSORUS_API_URL}/datasets/{dataset_name}/records", params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data.get("data", [])
        else:
            st.error(f"API Error fetching '{dataset_name}': {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error fetching '{dataset_name}': {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching '{dataset_name}': {e}")
        logger.exception(f"Unexpected error in fetch_dataset_data for {dataset_name}")
        return None

def execute_nql_query(query: str) -> Optional[Dict[str, Any]]:
    """Sends an NQL query to the API."""
    try:
        payload = {"query": query}
        response = requests.post(f"{TENSORUS_API_URL}/query", json=payload)
        # Handle specific NQL errors (400) vs other errors
        if response.status_code == 400:
             error_detail = response.json().get("detail", "Unknown NQL processing error")
             return {"success": False, "message": error_detail, "results": None, "count": None}
        response.raise_for_status() # Raise for 5xx etc.
        return response.json() # Return the full NQLResponse structure
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error executing NQL query: {e}")
        return {"success": False, "message": f"Connection Error: {e}", "results": None, "count": None}
    except Exception as e:
        st.error(f"Unexpected error executing NQL query: {e}")
        logger.exception("Unexpected error in execute_nql_query")
        return {"success": False, "message": f"Unexpected Error: {e}", "results": None, "count": None}

# --- NEW/UPDATED Agent and Metrics Functions ---

def list_all_agents() -> Optional[List[Dict[str, Any]]]:
    """Fetches the list of all registered agents from the API."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/agents")
        response.raise_for_status()
        # The response is directly the list of AgentInfo objects
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error listing agents: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error listing agents: {e}")
        logger.exception("Unexpected error in list_all_agents")
        return None

def get_agent_status(agent_id: str) -> Optional[Dict[str, Any]]:
    """Fetches status for a specific agent from the API."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/agents/{agent_id}/status")
        if response.status_code == 404:
            st.error(f"Agent '{agent_id}' not found via API.")
            return None
        response.raise_for_status()
        # Returns AgentStatus model dict
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error getting status for agent '{agent_id}': {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error getting status for agent '{agent_id}': {e}")
        logger.exception(f"Unexpected error in get_agent_status for {agent_id}")
        return None

def get_agent_logs(agent_id: str, lines: int = 20) -> Optional[List[str]]:
    """Fetches recent logs for a specific agent from the API."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/agents/{agent_id}/logs", params={"lines": lines})
        if response.status_code == 404:
            st.error(f"Agent '{agent_id}' not found via API for logs.")
            return None
        response.raise_for_status()
        data = response.json()
        # Returns AgentLogResponse model dict
        return data.get("logs", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error getting logs for agent '{agent_id}': {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error getting logs for agent '{agent_id}': {e}")
        logger.exception(f"Unexpected error in get_agent_logs for {agent_id}")
        return None

def start_agent(agent_id: str) -> bool:
    """Sends a start signal to an agent via the API."""
    try:
        response = requests.post(f"{TENSORUS_API_URL}/agents/{agent_id}/start")
        if response.status_code == 404:
            st.error(f"Agent '{agent_id}' not found via API.")
            return False
        # 202 Accepted is success, other 2xx might be okay too (e.g. already running if handled gracefully)
        # 4xx errors indicate failure
        if 200 <= response.status_code < 300:
            api_response = response.json()
            if api_response.get("success"):
                 st.success(f"API: {api_response.get('message', 'Start signal sent.')}")
                 return True
            else:
                 # API indicated logical failure (e.g., already running)
                 st.warning(f"API: {api_response.get('message', 'Agent might already be running.')}")
                 return False
        else:
             # Handle other potential errors reported by API
             error_detail = "Unknown error"
             try: error_detail = response.json().get("detail", error_detail)
             except: pass
             st.error(f"API Error starting agent '{agent_id}': {error_detail} (Status: {response.status_code})")
             return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error starting agent '{agent_id}': {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error starting agent '{agent_id}': {e}")
        logger.exception(f"Unexpected error in start_agent for {agent_id}")
        return False

def stop_agent(agent_id: str) -> bool:
    """Sends a stop signal to an agent via the API."""
    try:
        response = requests.post(f"{TENSORUS_API_URL}/agents/{agent_id}/stop")
        if response.status_code == 404:
            st.error(f"Agent '{agent_id}' not found via API.")
            return False
        if 200 <= response.status_code < 300:
            api_response = response.json()
            if api_response.get("success"):
                 st.success(f"API: {api_response.get('message', 'Stop signal sent.')}")
                 return True
            else:
                 st.warning(f"API: {api_response.get('message', 'Agent might already be stopped.')}")
                 return False
        else:
             error_detail = "Unknown error"
             try: error_detail = response.json().get("detail", error_detail)
             except: pass
             st.error(f"API Error stopping agent '{agent_id}': {error_detail} (Status: {response.status_code})")
             return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error stopping agent '{agent_id}': {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error stopping agent '{agent_id}': {e}")
        logger.exception(f"Unexpected error in stop_agent for {agent_id}")
        return False

def get_dashboard_metrics() -> Optional[Dict[str, Any]]:
    """Fetches dashboard metrics from the API."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/metrics/dashboard")
        response.raise_for_status()
        # Returns DashboardMetrics model dict
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error fetching dashboard metrics: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching dashboard metrics: {e}")
        logger.exception("Unexpected error in get_dashboard_metrics")
        return None

def get_agent_config(agent_id: str) -> Optional[Dict[str, Any]]:
    """Fetch an agent's configuration from the API."""
    try:
        response = requests.get(f"{TENSORUS_API_URL}/agents/{agent_id}/config")
        if response.status_code == 404:
            st.error(f"Agent '{agent_id}' not found via API for configuration.")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error fetching config for agent '{agent_id}': {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching config for agent '{agent_id}': {e}")
        logger.exception(f"Unexpected error in get_agent_config for {agent_id}")
        return None

def update_agent_config(agent_id: str, config: Dict[str, Any]) -> bool:
    """Send updated configuration for an agent to the API."""
    try:
        response = requests.post(
            f"{TENSORUS_API_URL}/agents/{agent_id}/configure",
            json={"config": config},
        )
        if response.status_code == 404:
            st.error(f"Agent '{agent_id}' not found via API for configuration.")
            return False
        if 200 <= response.status_code < 300:
            api_response = response.json()
            if api_response.get("success"):
                st.success(api_response.get("message", "Configuration updated."))
                return True
            else:
                st.error(api_response.get("message", "Failed to update configuration."))
                return False
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            st.error(
                f"API Error updating config for '{agent_id}': {error_detail} (Status: {response.status_code})"
            )
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error updating config for agent '{agent_id}': {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error updating config for agent '{agent_id}': {e}")
        logger.exception(f"Unexpected error in update_agent_config for {agent_id}")
        return False
