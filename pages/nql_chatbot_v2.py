# pages/nql_chatbot_v2.py

import streamlit as st
import pandas as pd

# Import from the shared utils for pages
try:
    from pages.pages_shared_utils import (
        load_css as load_shared_css,
        post_nql_query
    )
except ImportError:
    st.error("Critical Error: Could not import `pages_shared_utils`. Page cannot function.")
    def load_shared_css(): pass
    def post_nql_query(query: str): 
        st.error("`post_nql_query` unavailable.")
        return {"query": query, "response_text": "Error: NQL processing unavailable.", "error": "Setup issue.", "results": None}
    st.stop()

st.set_page_config(page_title="NQL Chatbot (V2)", layout="wide")
load_shared_css() # Load common CSS from shared utilities

# Custom CSS for NQL Chatbot page
# These styles are specific to the chat interface elements.
st.markdown("""
<style>
    /* Chatbot specific styles */
    .stChatMessage { /* Base style for chat messages */
        border-radius: 10px; /* Rounded corners */
        padding: 0.85rem 1.15rem; /* Comfortable padding */
        margin-bottom: 0.75rem; /* Space between messages */
        box-shadow: 0 2px 5px rgba(0,0,0,0.15); /* Subtle shadow */
        border: 1px solid #2a3f5c; /* Consistent border with other elements */
    }
    /* User messages styling */
    .stChatMessage[data-testid="stChatMessageContent"]:has(.user-avatar) {
        background-color: #2a2f4c; /* Darker blue, similar to nav hover, for user messages */
        border-left: 5px solid #3a6fbf; /* Accent blue border, consistent with active nav link */
    }
    /* Assistant messages styling */
    .stChatMessage[data-testid="stChatMessageContent"]:has(.assistant-avatar) {
        background-color: #18223f; /* Slightly lighter blue-gray, similar to common-card */
        border-left: 5px solid #7070ff; /* Muted purple accent for assistant messages */
    }
    .user-avatar, .assistant-avatar { /* Styling for user/assistant avatars */
        font-size: 1.5rem; /* Avatar size */
        margin-right: 0.5rem; /* Space between avatar and message content */
    }

    /* Styling for the chat input area at the bottom */
    /* The .chat-input-container class is a convention, Streamlit might not add this by default.
       Targeting based on Streamlit's internal structure is more robust. */
    div[data-testid="stChatInput"] { /* Main container for chat input */
        background-color: #0a0f2c; /* Match page background */
        border-top: 1px solid #2a3f5c; /* Separator line */
        padding: 0.75rem;
    }
    div[data-testid="stChatInput"] textarea { /* The actual text input field */
        border: 1px solid #3a3f5c !important;
        background-color: #1a1f3c !important; /* Dark input background */
        color: #e0e0e0 !important; /* Light text color */
        border-radius: 5px !important;
    }
    div[data-testid="stChatInput"] button { /* The send button */
        border: none !important;
        background-color: #3a6fbf !important; /* Accent blue, consistent with primary buttons */
        color: white !important;
        border-radius: 5px !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background-color: #4a7fdc !important; /* Lighter blue on hover */
    }

    /* Styling for DataFrames displayed within chat messages */
    .stDataFrame { /* General DataFrame styling from shared_utils will apply */
        margin-top: 0.5rem; /* Space above DataFrame in chat */
    }
    /* .results-dataframe and .results-dataframe .col-header are less reliable than generic .stDataFrame */
</style>
""", unsafe_allow_html=True)


st.title("💬 Natural Query Language (NQL) Chatbot")
st.caption("Query Tensorus datasets using natural language. Powered by Tensorus NQL Agent.")

# Initialize chat history in session state if it doesn't exist.
# Use a unique key for this page's messages to avoid conflicts with other potential chat interfaces.
if "nql_messages_v2" not in st.session_state: 
    st.session_state.nql_messages_v2 = []

# Display chat messages from history on app rerun.
# This ensures that the conversation persists during the session.
for message in st.session_state.nql_messages_v2:
    avatar = "👤" if message["role"] == "user" else "🤖" # Assign avatar based on role
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"]) # Display the text content of the message.
        # If the message includes a DataFrame (results from NQL query), display it.
        if "results_df" in message and message["results_df"] is not None and not message["results_df"].empty:
            st.dataframe(message["results_df"], use_container_width=True, hide_index=True)
        # If the message includes an error, display it in an error box.
        elif "error" in message and message["error"]:
            st.error(message["error"])


# React to user input from the chat interface.
if prompt := st.chat_input("Enter your query (e.g., 'show 5 tensors from my_dataset')"):
    # Add user's message to chat history and display it in the chat interface.
    st.session_state.nql_messages_v2.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get assistant's response by processing the query.
    with st.spinner("Tensorus NQL Agent is thinking..."): # Show a loading spinner.
        # Call the NQL processing function from shared utilities.
        nql_api_response = post_nql_query(prompt) 

        # Extract relevant information from the API response.
        response_text = nql_api_response.get("response_text", "Sorry, I encountered an issue processing your request.")
        results_data = nql_api_response.get("results") # Expected to be a list of dicts (records).
        error_message = nql_api_response.get("error") # Any error message from the backend.
        
        # Prepare data for storing the assistant's message in session state.
        assistant_message_data = {"role": "assistant", "content": response_text}

        # Display assistant's response in the chat interface.
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response_text) # Display the textual response.

            if results_data: # If the API returned 'results'.
                try:
                    # Process results into a Pandas DataFrame for structured display.
                    # Assumes results_data is a list of records (dictionaries).
                    # Each record might have 'id', 'shape', 'dtype', 'metadata'.
                    processed_for_df = []
                    for record in results_data:
                        row = {"tensor_id": record.get("id")} # Start with tensor_id.
                        # Flatten metadata fields into the main row for the DataFrame.
                        if isinstance(record.get("metadata"), dict):
                            row.update(record["metadata"])
                        row["shape"] = str(record.get("shape")) # Ensure shape is a string for display.
                        row["dtype"] = record.get("dtype")
                        processed_for_df.append(row)
                    
                    if processed_for_df:
                        results_df = pd.DataFrame(processed_for_df)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        assistant_message_data["results_df"] = results_df # Store DataFrame for history.
                    elif not error_message: # If no data and no error, it might be a query that doesn't return records.
                        st.caption("Query processed, no specific records returned.")

                except Exception as e:
                    # Handle errors during DataFrame processing.
                    st.error(f"Error formatting results for display: {e}")
                    assistant_message_data["error"] = f"Error formatting results: {e}"
            
            elif error_message: # If there's a specific error message from the API.
                st.error(error_message)
                assistant_message_data["error"] = error_message
            
            # If no results_data and no explicit error_message, the response_text itself is the primary message.

        # Add assistant's response (including any processed data or errors) to chat history.
        st.session_state.nql_messages_v2.append(assistant_message_data)

else: # This block runs when the page loads or if the chat input is empty.
    # Show a welcome/instruction message if the chat history is empty.
    if not st.session_state.nql_messages_v2: 
        st.info("Ask me anything about your data! For example: 'list datasets' or 'show tensors from dataset XYZ limit 5'.")
