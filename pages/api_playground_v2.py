# pages/api_playground_v2.py

import streamlit as st
import streamlit.components.v1 as components

# Import from the shared utils for pages
try:
    from pages.pages_shared_utils import (
        load_css as load_shared_css,
        API_BASE_URL, # Use this instead of TENSORUS_API_URL
        get_api_status
    )
except ImportError:
    st.error("Critical Error: Could not import `pages_shared_utils`. Page cannot function.")
    def load_shared_css(): pass
    API_BASE_URL = "http://127.0.0.1:8000" # Fallback
    def get_api_status(): 
        st.error("`get_api_status` unavailable.")
        return False, {"error": "Setup issue"} # Simulate API down
    st.stop()

st.set_page_config(page_title="API Playground (V2)", layout="wide")
load_shared_css() # Load common CSS from shared utilities

# Custom CSS for API Playground page
# These styles are specific to this page and enhance the shared theme.
st.markdown("""
<style>
    /* API Playground specific styles */
    /* Using classes for titles and captions allows for more specific styling if needed,
       while still inheriting base styles from shared CSS (h1, .stCaption etc.) */
    .main-title { 
        color: #e0e0ff !important; /* Ensure it uses the desired dashboard title color */
        text-align: center;
        margin-bottom: 0.5rem; 
        font-weight: bold;
    }
    .main-caption { 
        color: #c0c0ff !important; /* Consistent caption color */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Status message styling - these could be potentially moved to shared_utils if used elsewhere */
    /* For now, keeping them page-specific as they are styled slightly differently from .status-indicator */
    .status-message-shared { /* Base for status messages on this page */
        padding: 0.75rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center; /* Center align text in status messages */
    }
    .status-success { /* Success status message, inherits from .status-message-shared */
        color: #4CAF50; /* Green text */
        background-color: rgba(76, 175, 80, 0.1); /* Light green background */
        border: 1px solid rgba(76, 175, 80, 0.3); /* Light green border */
    }
    .status-error { /* Error status message, inherits from .status-message-shared */
        color: #F44336; /* Red text */
        background-color: rgba(244, 67, 54, 0.1); /* Light red background */
        border: 1px solid rgba(244, 67, 54, 0.3); /* Light red border */
    }
    .status-error code, .status-success code { /* Style <code> tags within status messages */
        background-color: rgba(0,0,0,0.1);
        padding: 0.1em 0.3em;
        border-radius: 3px;
    }

    /* Styling for the Streamlit Tabs component */
    .tab-container .stTabs [data-baseweb="tab-list"] { 
        background-color: transparent; /* Remove default Streamlit tab bar background if any */
        gap: 5px; /* Increased gap between tab headers */
        border-bottom: 1px solid #2a3f5c; /* Match other dividers */
    }
    .tab-container .stTabs [data-baseweb="tab"] { 
        background-color: #18223f; /* Match common-card background for inactive tabs */
        color: #c0c0ff; /* Standard text color for inactive tabs */
        border-radius: 5px 5px 0 0; /* Rounded top corners */
        padding: 0.75rem 1.5rem;
        border: 1px solid #2a3f5c; /* Border for inactive tabs */
        border-bottom: none; /* Remove bottom border as it's handled by tab-list */
        margin-bottom: -1px; /* Overlap with tab-list border */
    }
    .tab-container .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background-color: #2a2f4c; /* Slightly lighter for active tab, similar to nav hover */
        color: #ffffff; /* White text for active tab */
        font-weight: bold;
        border-color: #3a6fbf; /* Accent color for active tab border */
    }

    /* Container for the iframes displaying Swagger/ReDoc */
    .iframe-container {
        border: 1px solid #2a3f5c; /* Consistent border */
        border-radius: 0 5px 5px 5px; /* Rounded corners, except top-left to align with tabs */
        overflow: hidden; /* Ensures iframe respects border radius */
        margin-top: -1px; /* Align with tab-list border */
        height: 800px; /* Default height */
    }
</style>
""", unsafe_allow_html=True)

# Page Title and Caption, using custom classes for styling
st.markdown('<h1 class="main-title">🚀 API Playground & Documentation Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-caption">Explore and interact with the Tensorus REST API directly.</p>', unsafe_allow_html=True)


# Check if API is running using the shared utility function.
api_ok, api_info = get_api_status()

# Display API status message.
if not api_ok:
    # If API is not reachable, display an error message and stop page execution.
    st.markdown(
        f"""
        <div class="status-message-shared status-error">
            <strong>API Connection Error:</strong> The Tensorus API backend does not seem to be running or reachable at <code>{API_BASE_URL}</code>.
            <br>Please ensure the backend (<code>uvicorn api:app --reload</code>) is active to use the API Playground.
        </div>
        """, unsafe_allow_html=True
    )
    st.stop() # Halt further rendering of the page.
else:
    # If API is reachable, display a success message with API version.
    api_version = api_info.get("version", "N/A") # Get API version from status info.
    st.markdown(
        f"""
        <div class="status-message-shared status-success">
            Successfully connected to Tensorus API v{api_version} at <code>{API_BASE_URL}</code>.
        </div>
        """, unsafe_allow_html=True
    )

# Introductory text for the API Playground.
st.markdown(
    """
    This section provides live, interactive documentation for the Tensorus API,
    powered by FastAPI's OpenAPI integration. You can explore endpoints,
    view schemas, and even try out API calls directly in your browser.
    """
)

# Use tabs to embed Swagger UI and ReDoc for API documentation.
# Wrapping tabs in a div to apply specific tab styling if needed.
st.markdown('<div class="tab-container">', unsafe_allow_html=True) 
tab1, tab2 = st.tabs(["Swagger UI", "ReDoc"])
st.markdown('</div>', unsafe_allow_html=True)


# Construct the URLs for Swagger and ReDoc based on the API_BASE_URL from shared utils.
swagger_url = f"{API_BASE_URL}/docs"
redoc_url = f"{API_BASE_URL}/redoc"

# Swagger UI Tab
with tab1:
    st.subheader("Swagger UI") # Styled by shared CSS
    st.markdown(f"Explore the API interactively. [Open Swagger UI in new tab]({swagger_url})")
    # Embed Swagger UI using an iframe within a styled container.
    st.markdown('<div class="iframe-container">', unsafe_allow_html=True)
    components.iframe(swagger_url, height=800, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ReDoc Tab
with tab2:
    st.subheader("ReDoc") # Styled by shared CSS
    st.markdown(f"View the API documentation. [Open ReDoc in new tab]({redoc_url})")
    # Embed ReDoc using an iframe within a styled container.
    st.markdown('<div class="iframe-container">', unsafe_allow_html=True)
    components.iframe(redoc_url, height=800, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider() # Visual separator.
st.caption("Note: The API backend must be running and accessible to fully utilize the interactive documentation.")

```
