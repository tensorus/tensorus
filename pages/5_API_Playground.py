# pages/5_API_Playground.py

import streamlit as st
import streamlit.components.v1 as components
from ui_utils import TENSORUS_API_URL, get_api_status # Import base URL and status check

st.set_page_config(page_title="API Playground", layout="wide")

st.title("🚀 API Playground & Documentation Hub")
st.caption("Explore and interact with the Tensorus REST API.")

# Check if API is running
api_running = get_api_status()

if not api_running:
    st.error(
        f"The Tensorus API backend does not seem to be running at {TENSORUS_API_URL}. "
        "Please start the backend (`uvicorn api:app --reload`) to use the API Playground."
    )
    st.stop() # Stop execution if API is not available
else:
    st.success(f"Connected to API backend at {TENSORUS_API_URL}")

st.markdown(
    f"""
    This section provides live, interactive documentation for the Tensorus API,
    powered by FastAPI's OpenAPI integration. You can explore endpoints,
    view schemas, and even try out API calls directly in your browser.

    * **Swagger UI:** A graphical interface for exploring and testing API endpoints.
    * **ReDoc:** Alternative documentation format, often preferred for reading.

    Select a view below:
    """
)

# Use tabs to embed Swagger and ReDoc
tab1, tab2 = st.tabs(["Swagger UI", "ReDoc"])

# Construct the documentation URLs based on the API base URL
swagger_url = f"{TENSORUS_API_URL}/docs"
redoc_url = f"{TENSORUS_API_URL}/redoc"

with tab1:
    st.subheader("Swagger UI")
    st.markdown(f"Explore the API interactively. [Open in new tab]({swagger_url})")
    # Embed Swagger UI using an iframe
    components.iframe(swagger_url, height=800, scrolling=True)

with tab2:
    st.subheader("ReDoc")
    st.markdown(f"View the API documentation. [Open in new tab]({redoc_url})")
    # Embed ReDoc using an iframe
    components.iframe(redoc_url, height=800, scrolling=True)

st.divider()
st.caption("Note: Ensure the Tensorus API backend is running to interact with the playground.")
