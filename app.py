# app.py
"""
Streamlit frontend application for the Tensorus platform.
New UI structure with top navigation and Nexus Dashboard.
"""

import streamlit as st
import json
import time
import requests # Needed for ui_utils functions if integrated
import logging # Needed for ui_utils functions if integrated
import torch # Needed for integrated tensor utils
from typing import List, Dict, Any, Optional, Union, Tuple # Needed for integrated tensor utils
from pages.pages_shared_utils import get_api_status, get_agent_status, get_datasets # Updated imports

# Work around a Streamlit bug where inspecting `torch.classes` during module
# watching can raise a `RuntimeError`. Removing the module from `sys.modules`
# prevents Streamlit's watcher from trying to access it.
import sys
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# --- Page Configuration ---
st.set_page_config(
    page_title="Tensorus Platform",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse sidebar as nav is now at top
)

# --- Configure Logging ---
logger = logging.getLogger(__name__)

# --- Integrated Tensor Utilities (Preserved) ---

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

# --- Helper functions for dashboard (can be expanded) ---
def get_total_tensors_placeholder():
    # For now, as this endpoint is hypothetical for this task
    return "N/A"

@st.cache_data(ttl=300)
def get_active_datasets_placeholder():
    datasets = get_datasets()
    if datasets: # get_datasets returns [] on error or if no datasets
        return str(len(datasets))
    return "Error"

@st.cache_data(ttl=60)
def get_agents_online_placeholder():
    agent_data = get_agent_status()
    if agent_data:
        try:
            # Assuming agent_data is a dict like {'agent_id': {'status': 'running', ...}}
            # or {'agent_id': {'running': True, ...}}
            online_agents = sum(1 for agent in agent_data.values()
                                if agent.get('running') is True or str(agent.get('status', '')).lower() == 'running')
            total_agents = len(agent_data)
            return f"{online_agents}/{total_agents} Online"
        except Exception as e:
            logger.error(f"Error processing agent data for dashboard: {e}")
            return "Error"
    return "N/A" # If agent_data is None

# --- CSS Styles ---
# Renaming app.py's specific CSS loader to avoid confusion with the shared one.
def load_app_specific_css():
    # This function now only loads styles specific to the Nexus Dashboard content in app.py
    # General styles (body, .stApp, nav, common-card, etc.) are in pages_shared_utils.load_shared_css()
    st.markdown("""
<style>
    /* Nexus Dashboard Specific Styles */
    .dashboard-title { /* Custom title for the dashboard */
        color: #e0e0ff; /* Light purple/blue, matching shared h1 */
        text-align: center;
        margin-top: 1rem; /* Standardized margin */
        margin-bottom: 2rem;
        font-size: 2.8em; /* Slightly larger for main dashboard title */
        font-weight: bold;
    }

    /* Metric Cards Container for Dashboard */
    .metric-card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around; 
        gap: 20px; 
        padding: 0 1rem; 
    }
    
    /* Individual Metric Card - inherits from .common-card (defined in shared_utils) */
    /* This specific .metric-card class is for dashboard cards if they need further specialization */
    .metric-card { 
        /* Inherits background, border, padding, shadow, transition from .common-card */
        flex: 1 1 220px; /* Adjusted flex basis */
        min-width: 200px; 
        max-width: 300px;
        text-align: center;
    }
    /* .metric-card:hover is inherited from .common-card:hover */
    
    .metric-card .icon { /* Specific styling for icons within dashboard metric cards */
        font-size: 2.8em; /* Slightly larger icon for dashboard */
        margin-bottom: 0.5rem; /* Tighter spacing */
        /* color is inherited from .common-card .icon or can be overridden here */
    }
    .metric-card h3 { /* Metric card titles */
        /* color, font-size, margin-bottom, font-weight inherited from .common-card h3 */
        /* No specific overrides here unless needed for dashboard metric cards */
    }
    .metric-card p.metric-value { /* Specific class for the main value display */
        font-size: 2em; /* Larger font for the metric value */
        font-weight: bold;
        color: #ffffff; /* White color for emphasis */
        margin-top: 0.25rem; /* Adjust as needed */
        margin-bottom: 0;
    }

    /* Specific status icon colors for API status card in dashboard */
    .metric-card.api-status-connected .icon { color: #50C878; } /* Emerald Green */
    .metric-card.api-status-disconnected .icon { color: #FF6961; } /* Pastel Red */

    /* Activity Feed Styles for Dashboard */
    .activity-feed-container { /* Container for the activity feed section */
        margin-top: 2.5rem;
        padding: 0 1.5rem; 
    }
    /* .activity-feed-container h2 is covered by shared h2 styles */
    
    .activity-item { /* Individual item in the feed */
        background-color: #1e2a47; /* Slightly lighter than common-card, for variety */
        padding: 0.85rem 1.25rem; /* Adjusted padding */
        border-radius: 6px;
        margin-bottom: 0.6rem; /* Slightly more space */
        font-size: 0.95em;
        border-left: 4px solid #3a6fbf; /* Accent Blue border */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .activity-item .timestamp { /* Timestamp within an activity item */
        color: #8080af; /* Muted purple for timestamp */
        font-weight: bold;
        font-size: 0.85em; /* Smaller timestamp */
        margin-right: 0.75em; /* More space after timestamp */
    }
    .activity-item strong { /* Agent name or key part of activity */
        color: #b0b0df; /* Tertiary heading color for emphasis */
    }
</style>
""", unsafe_allow_html=True)

# --- Page Functions ---

def nexus_dashboard_content():
    # Uses .dashboard-title for its main heading
    st.markdown('<h1 class="dashboard-title">Tensorus Nexus</h1>', unsafe_allow_html=True)

    # System Health & Key Metrics
    # Uses .metric-card-container for the overall layout
    st.markdown('<div class="metric-card-container">', unsafe_allow_html=True)

    # Card 1: Total Tensors
    total_tensors_val = get_total_tensors_placeholder()
    st.markdown(f"""
    <div class="common-card metric-card">
        <div class="icon">⚙️</div>
        <h3>Total Tensors</h3>
        <p class="metric-value">{total_tensors_val}</p>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Active Datasets
    active_datasets_val = get_active_datasets_placeholder()
    st.markdown(f"""
    <div class="common-card metric-card">
        <div class="icon">📚</div>
        <h3>Active Datasets</h3>
        <p class="metric-value">{active_datasets_val}</p>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Agents Online
    agents_online_val = get_agents_online_placeholder()
    st.markdown(f"""
    <div class="common-card metric-card">
        <div class="icon">🤖</div>
        <h3>Agents Online</h3>
        <p class="metric-value">{agents_online_val}</p>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: API Status
    @st.cache_data(ttl=30)
    def cached_get_api_status():
        return get_api_status()

    api_ok, _ = cached_get_api_status()
    api_status_text_val = "Connected" if api_ok else "Disconnected"
    # Add specific class for API status icon coloring based on shared status styles
    api_status_icon_class = "api-status-connected" if api_ok else "api-status-disconnected"
    api_icon_char = "✔️" if api_ok else "❌"
    st.markdown(f"""
    <div class="common-card metric-card {api_status_icon_class}">
        <div class="icon">{api_icon_char}</div>
        <h3>API Status</h3>
        <p class="metric-value">{api_status_text_val}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close metric-card-container

    # Agent Activity Feed
    # Uses .activity-feed-container and h2 (which is styled by shared CSS)
    st.markdown('<div class="activity-feed-container">', unsafe_allow_html=True)
    st.markdown('<h2>Recent Agent Activity</h2>', unsafe_allow_html=True)

    # Placeholder activity items
    activity_items = [
        {"timestamp": "2023-10-27 10:05:15", "agent": "IngestionAgent", "action": "added 'img_new.png' to 'raw_images'"},
        {"timestamp": "2023-10-27 10:02:30", "agent": "RLAgent", "action": "completed training cycle, reward: 75.2"},
        {"timestamp": "2023-10-27 09:55:48", "agent": "MonitoringAgent", "action": "detected high CPU usage on node 'compute-01'"},
        {"timestamp": "2023-10-27 09:45:10", "agent": "IngestionAgent", "action": "processed batch of 100 sensor readings"},
    ]

    for item in activity_items:
        st.markdown(f"""
        <div class="activity-item">
            <span class="timestamp">[{item['timestamp']}]</span>
            <strong>{item['agent']}:</strong> {item['action']}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close activity-feed-container


# --- Main Application ---
# Import the shared CSS loader
try:
    from pages.pages_shared_utils import load_css as load_shared_css
except ImportError:
    st.error("Failed to import shared CSS loader. Page styling will be incomplete.")
    def load_shared_css(): pass # Dummy function

def main():
    load_shared_css() # Load shared styles first
    load_app_specific_css() # Then load app-specific styles (for dashboard)

    # Initialize session state for current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Nexus Dashboard"

    # --- Top Navigation Bar ---
    nav_items = {
        "Nexus Dashboard": "Nexus Dashboard",
        "Agents": "Agents",
        "Explorer": "Explorer",
        "Query Hub": "Query Hub",
        "API Docs": "API Docs"
    }

    nav_html_parts = [f'<div class="topnav-container"><span class="logo">🧊 Tensorus</span><nav>']
    for page_id, page_name in nav_items.items():
        active_class = "active" if st.session_state.current_page == page_id else ""
        # Use st.button an invisible character as key to make Streamlit rerun and update session state
        # This is a common workaround for making nav links update state
        # We'll use a more robust JavaScript approach if this is problematic, but st.query_params is better
        
        # Using query_params for navigation state is more robust
        # Check if query_params for page is set, if so, it overrides session_state
        query_params = st.query_params.to_dict()
        if "page" in query_params and query_params["page"] in nav_items:
            st.session_state.current_page = query_params["page"]
            # Clear the query param after use to avoid it sticking on manual refresh
            # However, for deeplinking, we might want to keep it.
            # For now, let's allow it to persist. To clear: st.query_params.clear()
            
        # Construct the link with query_params
        nav_html_parts.append(
            f'<a href="?page={page_id}" class="{active_class}" target="_self">{page_name}</a>'
        )

    nav_html_parts.append('</nav></div>')
    st.markdown("".join(nav_html_parts), unsafe_allow_html=True)
    
    # Handle page selection clicks (alternative to query_params if that proves problematic)
    # This part is tricky with pure st.markdown links.
    # The query_params approach is generally preferred for web-like navigation.

    # --- Content Area ---
    # The main app.py now acts as a router to other pages or displays dashboard content directly.
    if st.session_state.current_page == "Nexus Dashboard":
        nexus_dashboard_content()
    elif st.session_state.current_page == "Agents":
        st.switch_page("pages/control_panel_v2.py")
    elif st.session_state.current_page == "Explorer":
        st.switch_page("pages/data_explorer_v2.py")
    elif st.session_state.current_page == "Query Hub":
        st.switch_page("pages/nql_chatbot_v2.py")
    elif st.session_state.current_page == "API Docs":
        st.switch_page("pages/api_playground_v2.py")
    else:
        # Default to Nexus Dashboard if current_page is unrecognized
        st.session_state.current_page = "Nexus Dashboard"
        nexus_dashboard_content()
        # It's good practice to trigger a rerun if state was corrected
        st.rerun()


if __name__ == "__main__":
    # --- Initialize Old Session State Keys (to avoid errors if they are still used by preserved code) ---
    # This should be phased out as those sections are rebuilt.
    if 'agent_status' not in st.session_state: st.session_state.agent_status = None
    if 'datasets' not in st.session_state: st.session_state.datasets = []
    if 'selected_dataset' not in st.session_state: st.session_state.selected_dataset = None
    if 'dataset_preview' not in st.session_state: st.session_state.dataset_preview = None
    if 'explorer_result' not in st.session_state: st.session_state.explorer_result = None
    if 'nql_response' not in st.session_state: st.session_state.nql_response = None

    main()
